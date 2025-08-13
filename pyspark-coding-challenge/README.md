# pyspark-coding-challenge

This repository implements a PySpark pipeline that produces training inputs for a transformer-style model from impression and action data.

## What this repo contains

- `src/pipeline.py` - core PySpark pipeline functions. The main entrypoint is `produce_training_examples(...)`.
- `tests/test_pipeline.py` - pytest unit tests that validate correctness of history extraction, padding, and truncation.
- `requirements.txt` - minimal requirements: `pyspark` and `pytest`.

## Training input shape

Each produced training example corresponds to a **single impression item** (one item shown to a customer in a carousel on a date). For each impression we output:

- `dt` (string) - the day of the impression (YYYY-MM-DD)
- `ranking_id` (string) - original ranking id (if present)
- `customer_id` (long)
- `impression_pos` (int) - position in the original impression array
- `impression_item_id` (long) - item id of the impression (this will be mapped to embedding index by the data scientists)
- `label` (int) - `is_order` flag for that impression (0/1)
- `actions` (array<long>, length == MAX_HISTORY) - customer's most recent action item_ids strictly **before** the impression day, ordered most-recent-first. Padding with 0s if fewer than MAX_HISTORY.
- `action_types` (array<int>, length == MAX_HISTORY) - parallel array with values {1=click, 2=add_to_cart, 3=order, 0=pad}.

This shape aligns with the PyTorch forward signature described in the prompt:
forward(impressions: Tensor, actions: Tensor, action_types: Tensor) -> Tensor

where `impressions` is one integer per example, and `actions` & `action_types` are fixed-length sequences.

## Important design choices

- **No leakage**: actions included for an impression are strictly before the impression day (`action_time < dt 00:00:00`).
- **Fixed-length sequences**: sequences are truncated to `max_history` (default 1000) and padded with zeros. This enables efficient batching on GPU.
- **Ordering**: action index 0 is the most recent action.
- **Scalability**:
  - The pipeline uses Spark window functions and grouping to compute per-impression top-K actions.
  - Repartitioning by `customer_id` is applied before join/window to co-locate data and reduce shuffle — this parameter should be tuned for your cluster.
  - Outputs are row-wise per impression and partitionable by `dt` for efficient per-day training reads.
- **Storage**: parquet with snappy compression is recommended for output in production.

## How the pipeline works (high level)

1. **Normalize** click, add-to-cart and order sources into a single `actions` table with columns `(customer_id, item_id, action_time, action_type)`.
2. **Explode** impressions to one row per impression item.
3. **Join** impressions with actions by `customer_id`, filtering to `action_time < impression_day` (no leakage).
4. **Window** by `(_imp_row_id)` (unique per-impression) ordered by `action_time desc` and keep top-K actions (`row_number <= max_history`).
5. **Aggregate** those actions into arrays, order them by recency, then pad/truncate to the fixed length.
6. **Return** the final per-impression row with arrays `actions` and `action_types`.

## Running locally / tests
1. Create a venv and install deps:
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Run tests:
pytest -q


## Deploying or running on Databricks
- The code is written using standard PySpark APIs and should run on Databricks with minimal changes.
- Recommendations for Databricks:
- Use cluster-scoped configuration for shuffle partitions.
- Tune `repartition_count` near `spark.sparkContext.defaultParallelism`.
- Persist `all_actions` if multiple days of impressions will be processed in the same job.
- Write outputs partitioned by `dt` to S3/ADLS for fast per-day reads during training.

## Performance notes & further improvements
- For high-throughput production, consider:
- Precomputing per-customer rolling action arrays (e.g., last 1000) and periodically persisting them as materialized tables keyed by customer_id and cutoffs, then slicing per day — this avoids reprocessing large action volumes per dt.
- Handling skew (very active customers) by splitting them across more partitions or capping stored actions for the hottest users.
- Using columnar formats (Parquet/ORC) and proper partitioning by `dt` and optionally `customer_hash_mod_n`.
- If memory/latency are critical, consider a two-step approach: (1) compute per-customer ordered actions and store them compactly, (2) when producing per-day impressions join to the compact store, which is cheaper than joining raw Kafka-level data every time.
- The pipeline favors correctness and clarity; you can micro-optimize further for your cluster and data volume.

## Notes
- This repo implements the core transformation functions and tests to validate correctness. It is purposely minimal and written to be understandable and extensible.
