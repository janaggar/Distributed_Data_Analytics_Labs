# Distributed Data Analysis — Labs 1, 2 & 3

**Faculty of Computers and Data Science, Alexandria University**

This repository contains three progressive PySpark labs: an introduction to Spark DataFrames, a hands-on RDD-based distributed data analysis project, and a full distributed machine learning pipeline for spam detection.

## Contents

| Lab | File | Focus |
|---|---|---|
| Lab 1 | `Lab1.ipynb` | Spark fundamentals with the Titanic dataset (DataFrame API) |
| Lab 2 | *(this assignment)* | RDD transformations, key-value operations, and performance tuning |
| Lab 3 | `Spam_Detection_Lab3_Distributed_Run_All.ipynb` | End-to-end distributed spam classification pipeline (Spark MLlib) |

## Requirements

- Python 3.x
- PySpark (`pip install pyspark`)
- pandas (for the Lab 1 performance comparison)
- Datasets: `Titanic-Dataset.csv` (Lab 1), `spam.csv` (Lab 3)

---

## Lab 1 — Spark Fundamentals (Titanic Dataset)

Introduces core Spark concepts using the DataFrame API.

**Steps covered:**

1. Install and import PySpark, create a `SparkSession`
2. Load `Titanic-Dataset.csv` with schema inference
3. Inspect the data: `printSchema()`, `show()`, `count()`, `describe()`
4. Filter records (e.g. passengers over 30, surviving female passengers)
5. Aggregate with `groupBy()` — passenger counts and average age by class, average fare by survival status
6. Compare load and filter performance between pandas and Spark

**Observation:** pandas is faster than Spark on this small dataset (~0.016s vs ~0.25-0.37s), since Spark's distributed overhead only pays off at larger scale.

---

## Lab 2 — RDD-Based Distributed Data Analysis

Builds a full RDD pipeline (no DataFrame API) across five phases.

### Phase 1: Data Loading & Inspection
- Create a Spark session
- Load the dataset with `sc.textFile()`
- Remove the header row
- Split each row into columns
- Display sample records and total record count
- Check partition count with `getNumPartitions()`
- **Required calls:** `map()`, `filter()`, `take()`, `getNumPartitions()`

### Phase 2: Data Cleaning (RDD transformations only)
- Remove missing values
- Remove duplicate records
- Fix malformed rows
- Trim whitespace
- Convert numeric columns to proper types

### Phase 3: Core RDD Transformations (at least 6, chosen from)
`map()`, `flatMap()`, `filter()`, `distinct()`, `union()`, `intersection()`, `subtract()`, `sample()`, `mapPartitions()`, `coalesce()`, `repartition()`

For each transformation used: show the code, explain its purpose, and show a sample output.

### Phase 4: Key-Value Operations (at least 4, chosen from)
`reduceByKey()`, `groupByKey()`, `countByKey()`, `sortByKey()`, `combineByKey()`, `aggregateByKey()`

### Phase 5: Partitioning & Performance
- Analyze the number of partitions
- Apply `cache()` or `persist()`
- Compare execution time before and after caching
- Apply `repartition()` or `coalesce()`
- Discuss performance observations

---

## Lab 3 — Distributed Spam Detection Pipeline

An end-to-end SMS spam classifier built entirely on Spark MLlib, structured as nine phases.

### Phase 1: Data Ingestion
Load `spam.csv` with schema inference and inspect the raw schema and sample rows.

### Phase 2: Data Cleaning
- Keep and rename only the relevant columns (`v1` -> `label`, `v2` -> `message`)
- Drop rows with null or empty labels/messages
- Encode labels: `spam` -> 1, `ham` -> 0 (unexpected values become null and are dropped)

### Phase 3: Exploratory Data Analysis
- Spam vs. ham class distribution
- Message length statistics (mean, standard deviation) by class
- Message length buckets (ASCII histogram)
- Class imbalance ratio and warning if ham:spam exceeds 3:1

### Phase 4: Feature Engineering
- `Tokenizer` — split messages into words
- `StopWordsRemover` — remove stopwords
- `HashingTF` — compute term frequency
- `IDF` — compute inverse document frequency weighting

### Phase 5 & 6: Model & Pipeline
- `LogisticRegression` classifier on the `features` column
- Assemble a `Pipeline` (tokenizer -> remover -> TF -> IDF -> logistic regression)
- Split data into train/test sets and fit the pipeline

### Phase 7: Evaluation
Evaluate the trained model with `MulticlassClassificationEvaluator`: accuracy, precision, and recall.

### Phase 8: Save & Load
Persist the trained `PipelineModel` to disk and reload it to confirm it can be reused without retraining.

### Phase 9: New Predictions
Run the loaded model on new, unseen SMS messages to demonstrate inference on fresh data.

---

## How to Run

Open each notebook in Jupyter or Google Colab and run all cells in order:

```bash
jupyter notebook Lab1.ipynb
jupyter notebook Spam_Detection_Lab3_Distributed_Run_All.ipynb
```

For Lab 2, follow the phase structure above, ensuring each required transformation/action is demonstrated with code, an explanation, and sample output as specified in the assignment brief.
