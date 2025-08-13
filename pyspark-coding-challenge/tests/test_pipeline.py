import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from src.pipeline import produce_training_examples
import datetime

@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder \
        .master("local[2]") \
        .appName("pyspark-coding-challenge-tests") \
        .getOrCreate()
    yield spark
    spark.stop()

def make_impressions_df(spark):
    # Sample impressions: one customer, dt=2025-08-14, two impressions
    data = [
        {
            "dt": "2025-08-14",
            "ranking_id": "r1",
            "customer_id": 1,
            "impressions": [
                {"item_id": 100, "is_order": False},
                {"item_id": 200, "is_order": True}
            ]
        },
        {
            "dt": "2025-08-14",
            "ranking_id": "r2",
            "customer_id": 2,
            "impressions": [
                {"item_id": 300, "is_order": False}
            ]
        }
    ]
    return spark.read.json(spark.sparkContext.parallelize([F.json.dumps(x) for x in data]))


def make_actions_dfs(spark):
    # Create clicks, adds, orders with timestamps.
    # Customer 1 has 3 actions before dt and 1 action on dt (which should be excluded).
    clicks = [
        {"dt": "2025-08-13", "customer_id": 1, "item_id": 10, "click_time": "2025-08-13 12:00:00"},
        {"dt": "2025-08-14", "customer_id": 1, "item_id": 999, "click_time": "2025-08-14 01:00:00"},  # same day -> exclude
    ]
    adds = [
        {"dt": "2025-07-01", "customer_id": 1, "config_id": 20, "simple_id": 1, "occurred_at": "2025-07-01 09:00:00"},
    ]
    orders = [
        {"order_date": "2024-12-01", "customer_id": 1, "config_id": 30, "simple_id": 1, "occurred_at": "2024-12-01 17:00:00"},
    ]

    clicks_df = spark.createDataFrame(clicks).withColumn("click_time", F.to_timestamp("click_time"))
    adds_df = spark.createDataFrame(adds).withColumn("occurred_at", F.to_timestamp("occurred_at"))
    orders_df = spark.createDataFrame(orders).withColumn("occurred_at", F.to_timestamp("occurred_at"))

    # For customer 2, no actions -> arrays should be all zeros
    return clicks_df, adds_df, orders_df


def test_basic_history_extraction(spark):
    # Build impressions and actions
    impressions = spark.createDataFrame([
        {
            "dt": "2025-08-14",
            "ranking_id": "r1",
            "customer_id": 1,
            "impressions": [
                {"item_id": 100, "is_order": False},
                {"item_id": 200, "is_order": True}
            ]
        },
        {
            "dt": "2025-08-14",
            "ranking_id": "r2",
            "customer_id": 2,
            "impressions": [
                {"item_id": 300, "is_order": False}
            ]
        }
    ])

    clicks_df = spark.createDataFrame([
        {"dt": "2025-08-13", "customer_id": 1, "item_id": 10, "click_time": "2025-08-13 12:00:00"}
    ]).withColumn("click_time", F.to_timestamp("click_time"))

    adds_df = spark.createDataFrame([
        {"dt": "2025-07-01", "customer_id": 1, "config_id": 20, "simple_id": 1, "occurred_at": "2025-07-01 09:00:00"}
    ]).withColumn("occurred_at", F.to_timestamp("occurred_at"))

    orders_df = spark.createDataFrame([
        {"order_date": "2024-12-01", "customer_id": 1, "config_id": 30, "simple_id": 1, "occurred_at": "2024-12-01 17:00:00"}
    ]).withColumn("occurred_at", F.to_timestamp("occurred_at"))

    out = produce_training_examples(impressions, clicks_df, adds_df, orders_df, max_history=5)
    rows = out.collect()

    # Two impressions for customer 1 (r1) and one for customer 2 (r2) => total 3 rows
    assert len(rows) == 3

    # Find row for customer 1, impression_pos 0 (first impression)
    r1_pos0 = [r for r in rows if r["customer_id"] == 1 and r["impression_pos"] == 0][0]
    # Most recent actions before dt (descending) for customer 1 should be:
    # click on 2025-08-13 item 10 (most recent), add 20 (2025-07-01), order 30 (2024-12-01)
    expected_actions = [10, 20, 30, 0, 0]  # padded to length 5
    assert r1_pos0["actions"][:5] == expected_actions
    assert r1_pos0["action_types"][:3] == [1, 2, 3]  # types for those items

    # customer 2 should have actions all zeros (no history)
    r2 = [r for r in rows if r["customer_id"] == 2][0]
    assert r2["actions"][:5] == [0, 0, 0, 0, 0]
    assert r2["action_types"][:5] == [0, 0, 0, 0, 0]


def test_truncation_and_padding(spark):
    # Customer 99 has 7 actions and we ask for max_history=5 so should truncate to 5 most recent.
    impressions = spark.createDataFrame([
        {"dt": "2025-08-14", "ranking_id": "rX", "customer_id": 99, "impressions": [{"item_id": 1, "is_order": False}]}
    ])

    # create 7 clicks with increasing timestamps; later timestamps are more recent
    clicks = []
    base = datetime.datetime(2025, 8, 1, 0, 0, 0)
    for i in range(7):
        t = (base + datetime.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
        clicks.append({"dt": "2025-08-0"+str(i+1), "customer_id": 99, "item_id": 1000 + i, "click_time": t})
    clicks_df = spark.createDataFrame(clicks).withColumn("click_time", F.to_timestamp("click_time"))
    adds_df = spark.createDataFrame([], schema="dt string, customer_id long, config_id long, simple_id int, occurred_at timestamp")
    orders_df = spark.createDataFrame([], schema="order_date string, customer_id long, config_id long, simple_id int, occurred_at timestamp")

    out = produce_training_examples(impressions, clicks_df, adds_df, orders_df, max_history=5)
    row = out.collect()[0]
    # most recent 5 items (highest timestamps)
    # clicks have item_ids 1000..1006; most recent are 1006,1005,1004,1003,1002 (descending)
    assert row["actions"][:5] == [1006, 1005, 1004, 1003, 1002]
    assert row["action_types"][:5] == [1, 1, 1, 1, 1]

