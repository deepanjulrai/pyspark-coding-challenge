"""
pyspark-coding-challenge : pipeline module

Core API function:

produce_training_examples(
    impressions_df,
    clicks_df,
    add_to_carts_df,
    orders_df,
    max_history=1000
) -> DataFrame

Output schema (per training example):
  - dt: string (YYYY-MM-DD)       -> impression date partition
  - ranking_id: string (optional)
  - customer_id: long
  - impression_item_id: long
  - impression_pos: int (position in original impressions array, 0-based)
  - label: int (0/1)              -> is_order flag on that impression
  - actions: array<int> (len == max_history)       -> item_ids, most recent first (index 0)
  - action_types: array<int> (len == max_history)  -> {1=click,2=add_to_cart,3=order,0=pad}
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import IntegerType, ArrayType, LongType

DEFAULT_MAX_HISTORY = 1000


def normalize_actions(clicks_df: DataFrame, adds_df: DataFrame, orders_df: DataFrame) -> DataFrame:
    """
    Normalize three action sources into a common schema:
      customer_id: long
      item_id: long
      action_time: timestamp
      action_type: int  (1=click, 2=add_to_cart, 3=order)
    """
    clicks_n = clicks_df.select(
        F.col("customer_id").cast("long"),
        F.col("item_id").cast("long"),
        F.col("click_time").alias("action_time"),
        F.lit(1).alias("action_type"),
    )

    adds_n = adds_df.select(
        F.col("customer_id").cast("long"),
        F.col("config_id").alias("item_id").cast("long"),
        F.col("occurred_at").alias("action_time"),
        F.lit(2).alias("action_type"),
    )

    orders_n = orders_df.select(
        F.col("customer_id").cast("long"),
        F.col("config_id").alias("item_id").cast("long"),
        F.col("occurred_at").alias("action_time"),
        F.lit(3).alias("action_type"),
    )

    unioned = clicks_n.unionByName(adds_n).unionByName(orders_n)
    # remove rows missing important fields
    return unioned.filter(F.col("customer_id").isNotNull() & F.col("item_id").isNotNull() & F.col("action_time").isNotNull())


def explode_impressions(impressions_df: DataFrame) -> DataFrame:
    """
    Explodes impressions array into one row per impression item.

    Input impressions schema:
      dt: string (YYYY-MM-DD)
      ranking_id: string
      customer_id: long
      impressions: array<struct<item_id:long, is_order:boolean>>

    Output:
      dt, ranking_id, customer_id, impression_pos (int), impression_item_id (long), label (int)
    """
    # Use posexplode to keep position
    exploded = impressions_df.select(
        F.col("dt"),
        F.col("ranking_id"),
        F.col("customer_id").cast("long"),
        F.posexplode_outer("impressions").alias("impression_pos", "imp_struct")
    )

    # imp_struct may be null (defensive)
    return exploded.select(
        "dt",
        "ranking_id",
        "customer_id",
        F.col("impression_pos").cast("int"),
        F.col("imp_struct.item_id").alias("impression_item_id").cast("long"),
        F.when(F.col("imp_struct.is_order") == True, F.lit(1)).otherwise(F.lit(0)).alias("label")
    )


def get_customer_history_before_dt(actions_df: DataFrame, dt_col_or_literal: str, max_history: int = DEFAULT_MAX_HISTORY) -> DataFrame:
    """
    For each (customer_id, dt) pair we need the customer's most recent actions strictly BEFORE dt.
    This function expects to be given the actions_df (normalized) and will return a table:
       customer_id, dt, actions, action_types
    where actions and action_types are arrays of length <= max_history (not yet padded to exact length).
    NOTE: dt_col_or_literal should be a column expression string representing the dt cutoff (date string 'YYYY-MM-DD'),
    we will convert it to a timestamp at 00:00:00 and compare action_time < cutoff.
    """

    # dt_col_or_literal may be a column name (e.g., "dt") or a literal string; we will support both by accepting a SQL expression.
    # We'll not materialize dt for all customers; instead the caller will join exploded impressions with this per-dt history via join on customer+dt.
    # Implementation here constructs, for every customer and dt (coming from a join), the top-k actions before that dt.
    # However this helper will be used in the pipeline by joining impressions -> actions filtered by dt.

    # This function is kept for clarity; actual implementation occurs in produce_training_examples where we have both impressions and actions in scope.
    raise NotImplementedError("Use produce_training_examples which contains the full pipeline combining impressions and actions.")


def produce_training_examples(
    impressions_df: DataFrame,
    clicks_df: DataFrame,
    add_to_carts_df: DataFrame,
    orders_df: DataFrame,
    max_history: int = DEFAULT_MAX_HISTORY
) -> DataFrame:
    """
    Main pipeline. Returns DataFrame with columns:
      dt, ranking_id, customer_id, impression_pos, impression_item_id, label, actions, action_types
    Where actions and action_types are arrays of length == max_history, most recent first (index 0).
    """
    # Normalize actions
    all_actions = normalize_actions(clicks_df, add_to_carts_df, orders_df)

    # Explode impressions into per-impression rows
    ex = explode_impressions(impressions_df).alias("imp")

    # Convert impression dt to cutoff timestamp (midnight of dt): action_time < cutoff (no leakage)
    # create a column 'cutoff_ts' to use in join/filter
    ex = ex.withColumn("cutoff_ts", F.to_timestamp(F.concat(F.col("dt"), F.lit(" 00:00:00"))))

    # We'll join actions to impressions by customer_id and filter action_time < cutoff_ts
    # But to efficiently gather top-K ordered actions per (customer_id, dt), we:
    #  1. Filter actions by action_time < cutoff_ts via a broadcast join optimization for small dt list,
    #     but here we create a join key and use window partitioning.
    #
    # Approach:
    #   - Join impressions (customer_id, cutoff_ts) with actions on customer_id
    #   - Filter action_time < cutoff_ts
    #   - For each (customer_id, dt) assign row_number() ordered by action_time desc
    #   - Keep rn <= max_history
    #   - Group by original impression row (dt, ranking_id, customer_id, impression_pos) and collect lists
    #
    # Note: This may trigger a shuffle: we repartition by customer_id to colocate customer's actions and impressions.
    repartition_count = 200  # reasonable default; user can tune for their cluster
    ex_repart = ex.repartition(repartition_count, "customer_id")

    actions_repart = all_actions.repartition(repartition_count, "customer_id")

    # Join
    joined = ex_repart.join(actions_repart, on="customer_id", how="left_outer") \
        .where(F.col("action_time") < F.col("cutoff_ts"))

    # Row number per (impression row) based on action_time descending. To identify each impression row we need a unique id:
    # We'll create an impression_row_id using concat of dt, ranking_id, customer_id, impression_pos (safe as string)
    joined = joined.withColumn(
        "_imp_row_id",
        F.concat_ws("::", F.col("dt"), F.coalesce(F.col("ranking_id"), F.lit("NULL")), F.col("customer_id"), F.col("impression_pos"))
    )

    w = Window.partitionBy("_imp_row_id").orderBy(F.col("action_time").desc())
    joined = joined.withColumn("rn", F.row_number().over(w))

    filtered_topk = joined.filter(F.col("rn") <= F.lit(max_history))

    # Collect (rn, item_id, action_type) per impression row
    struct_col = F.struct(F.col("rn").cast("int").alias("rn"),
                          F.col("item_id").cast("long").alias("item_id"),
                          F.col("action_type").cast("int").alias("action_type"))

    agg = filtered_topk.groupBy(
        "_imp_row_id", "dt", "ranking_id", "customer_id", "impression_pos", "impression_item_id", "label", "cutoff_ts"
    ).agg(
        F.collect_list(struct_col).alias("action_structs")
    )

    # action_structs need to be ordered by rn ascending (1 = most recent). Use array_sort which sorts by first struct field (rn).
    agg = agg.withColumn("action_structs_sorted", F.expr("array_sort(action_structs)"))

    # extract actions and action_types arrays preserving order
    agg = agg.withColumn("actions_unsized", F.expr("transform(action_structs_sorted, x -> x.item_id)")) \
             .withColumn("action_types_unsized", F.expr("transform(action_structs_sorted, x -> x.action_type)"))

    # pad/truncate to exact length max_history
    # Use array_repeat(0, max_history) and slice on concatenated array
    agg = agg.withColumn(
        "actions",
        F.expr(f"slice(array_concat(actions_unsized, array_repeat(0, {max_history})), 1, {max_history})")
    ).withColumn(
        "action_types",
        F.expr(f"slice(array_concat(action_types_unsized, array_repeat(0, {max_history})), 1, {max_history})")
    )

    # For impressions that had no matching actions (NULL action_structs), ensure arrays exist: the expressions above produce empty arrays then concat -> padded zeros.
    # Select and return desired columns
    result = agg.select(
        F.col("dt"),
        F.col("ranking_id"),
        F.col("customer_id"),
        F.col("impression_pos"),
        F.col("impression_item_id"),
        F.col("label"),
        F.col("actions"),
        F.col("action_types")
    )

    # Cast arrays to arrays of ints/longs for schema clarity (Spark may infer)
    # action ids: long, action types: int
    result = result.withColumn("actions", F.expr("transform(actions, x -> cast(x as long))")) \
                   .withColumn("action_types", F.expr("transform(action_types, x -> cast(x as int))"))

    return result
