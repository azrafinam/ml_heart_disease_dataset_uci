# Heart Disease Prediction Using Machine Learning

**Full submission write-up:** [ASMICORE_SUBMISSION_REPORT.md](ASMICORE_SUBMISSION_REPORT.md)  
**Repository:** https://github.com/azrafinam/ml_heart_disease_dataset_uci

## Project Overview

This project builds a machine-learning workflow for the UCI Heart Disease dataset. The `num` target supports two supervised formulations:

1. **Multiclass severity (0-4):** `0` = no heart disease, `1-4` = increasing severity.
2. **Binary presence:** `0` = no disease, `1` = disease (original `num` values `1-4` collapsed to `1`).

Both formulations use the same leakage-safe preprocessing and model comparison workflow.

The project is organized as a six-day notebook workflow and a matching command-line pipeline. The final version is leakage-safe: train/test splitting happens before imputation, encoding, scaling, model fitting, and model evaluation.

## Problem Statement

Heart disease risk assessment can support earlier clinical investigation and preventive care. This project compares multiple supervised classification algorithms using patient medical attributes such as age, cholesterol, blood pressure, chest pain type, ECG results, maximum heart rate, exercise-induced angina, and related clinical measurements.

This is an educational machine-learning project and is not intended for clinical deployment without external validation.

## Dataset

- Dataset: UCI Heart Disease dataset
- File: `data/heart_disease_uci.csv`
- Rows: 920
- Columns: 16
- Target column: `num`
- Target formulations:
  - **Multiclass:** five severity classes (`0`, `1`, `2`, `3`, `4`)
  - **Binary:** two classes (`0` = no disease, `1` = disease with severity `1-4` mapped to `1`)
- Derived column (optional): `num_binary` for EDA and processed exports

The dataset contains numeric and categorical features with missing values. Missing values are handled inside a preprocessing pipeline fitted on training data only.

## Models Compared

The project currently trains and evaluates 22 models:

- Dummy baseline
- Logistic Regression
- Ridge Classifier
- SGD Classifier
- Perceptron
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Linear SVM
- RBF SVM
- Decision Tree
- Random Forest
- Extra Trees
- Bagging
- AdaBoost
- Gradient Boosting
- HistGradientBoosting
- Linear Discriminant Analysis
- Nearest Centroid
- XGBoost
- LightGBM
- CatBoost
- RandomizedSearchCV-tuned Random Forest

## Current Best Results

| Target mode | Best model | F1 | ROC-AUC | Bundle |
|-------------|------------|-----|---------|--------|
| Multiclass (0–4) | `hist_gradient_boosting` | 0.6096 | 0.8293 | `hist_gradient_boosting_final_bundle.pkl` |
| Binary (0 vs 1) | `gaussian_nb` | 0.8694 | 0.9174 | `gaussian_nb_final_bundle_binary.pkl` |

Selection metric: weighted F1 on the held-out test set. Multiclass class 4 is rare (3% of rows), so severity-wise metrics should be read carefully. Binary collapses `num` values 1–4 into disease present (`1`).

Re-run `python scripts/main.py` to regenerate all metrics and artifacts for both modes.

## Project Structure

```text
.
├── data/
│   ├── heart_disease_uci.csv
│   └── processed/
│       ├── X_train_processed.csv
│       ├── X_test_processed.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── y_train_binary.csv
│       ├── y_test_binary.csv
│       ├── X_processed.csv
│       └── y_processed.csv
├── notebooks/
│   ├── day1_data_exploration_IMPROVED.ipynb
│   ├── day2_eda_IMPROVED.ipynb
│   ├── day3_feature_engineering.ipynb
│   ├── day4_model_training.ipynb
│   ├── day5_evaluation.ipynb
│   └── day6_final_pipeline_IMPROVED.ipynb
├── outputs/
│   ├── models/
│   │   ├── preprocessor.pkl
│   │   ├── preprocessor_binary.pkl
│   │   ├── *_trained.pkl
│   │   ├── *_trained_binary.pkl
│   │   ├── hist_gradient_boosting_final_bundle.pkl
│   │   └── gaussian_nb_final_bundle_binary.pkl
│   ├── reports/
│   │   ├── day1_data_summary.json
│   │   ├── day2_eda_report.json
│   │   ├── day3_preprocessing_metadata.json
│   │   ├── day4_model_manifest.json
│   │   ├── day4_model_manifest_binary.json
│   │   ├── day5_evaluation_report.json
│   │   ├── day5_evaluation_report_binary.json
│   │   ├── day6_final_pipeline_report.json
│   │   ├── day6_final_pipeline_report_binary.json
│   │   ├── model_comparison_table.csv
│   │   ├── model_comparison_table_binary.csv
│   │   ├── MODEL_USAGE_GUIDE.txt
│   │   └── MODEL_USAGE_GUIDE_binary.txt
│   └── visualizations/
├── scripts/
│   ├── main.py
│   └── utils.py
├── ASMICORE_SUBMISSION_REPORT.md
├── environment.yml
├── requirements.txt
└── README.md
```

## Workflow

```text
Raw data
  -> Data exploration
  -> EDA and visualizations
  -> Train/test split
  -> Fit preprocessing on training data only
  -> Transform train/test data
  -> Train 22 models (multiclass 0-4)
  -> Evaluate on held-out test set
  -> Select best model by weighted F1
  -> Save final model + preprocessor bundle
  -> Repeat for binary target (0 vs 1)
```

## Notebook Summary

### Day 1: Data Exploration

- Loads raw dataset with `?` treated as missing values.
- Checks shape, data types, missing values, duplicates, and target distribution.
- Saves `outputs/reports/day1_data_summary.json`.

### Day 2: Exploratory Data Analysis

- Creates target distribution, numeric distributions, categorical distributions, correlation heatmap, and boxplots.
- Saves EDA visualizations in `outputs/visualizations/`.
- Saves `outputs/reports/day2_eda_report.json`.

### Day 3: Feature Engineering and Preprocessing

- Splits data into train/test sets before preprocessing.
- Fits numeric imputation and scaling on training data only.
- Fits categorical imputation and one-hot encoding on training data only.
- Saves train/test processed files for multiclass and binary targets (`y_train.csv`, `y_train_binary.csv`, and matching test files).
- Saves `outputs/models/preprocessor.pkl` (multiclass) and `preprocessor_binary.pkl` when using the CLI pipeline.

### Day 4: Model Training

- Loads processed training data from Day 3.
- Trains the full 22-model zoo.
- Saves each trained model to `outputs/models/`.
- Saves `outputs/reports/day4_model_manifest.json`.

### Day 5: Model Evaluation

- Loads only models listed in the Day 4 manifest.
- Evaluates models on the held-out test set (multiclass).
- Saves `outputs/reports/model_comparison_table.csv` and `day5_evaluation_report.json`.
- Binary metrics: run `python scripts/main.py` or use `*_binary` report files.

### Day 6: Final Pipeline

- Loads the best multiclass model from Day 5.
- Loads the fitted preprocessor from Day 3.
- Saves `outputs/models/hist_gradient_boosting_final_bundle.pkl`.
- Saves binary bundle when `day5_evaluation_report_binary.json` exists (see Day 6 notebook Section 5B).

## Running The Project

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate ml_1
```

Run the full script pipeline:

```bash
python scripts/main.py
```

This runs both target modes in one pass: multiclass severity (`0-4`) and binary presence (`0` vs `1`).

Or run the notebooks in order:

```text
Day 1 -> Day 2 -> Day 3 -> Day 4 -> Day 5 -> Day 6
```

## Using The Final Models

**Multiclass severity (0–4):**

```python
import joblib
import pandas as pd

bundle = joblib.load("outputs/models/hist_gradient_boosting_final_bundle.pkl")
model = bundle["model"]
preprocessor = bundle["preprocessor"]

raw_rows = pd.DataFrame([...])  # original feature columns, excluding id and num
X_new_array = preprocessor.transform(raw_rows)
X_new = pd.DataFrame(X_new_array, columns=bundle["feature_names"])
severity = model.predict(X_new)  # 0-4
```

**Binary presence (0=no disease, 1=disease):**

```python
bundle = joblib.load("outputs/models/gaussian_nb_final_bundle_binary.pkl")
# Same transform/predict pattern; predictions are 0 or 1
```

## Important Methodology Notes

- The test set is not used to fit preprocessing.
- Model comparison uses the same held-out test split for every model.
- Weighted F1 is used for model selection because the target classes are imbalanced.
- ROC-AUC is reported where model scores are available; models without usable probability or decision scores may show missing ROC-AUC.
- The final artifact includes preprocessing so inference can start from raw feature rows.

## Main Outputs

- [ASMICORE_SUBMISSION_REPORT.md](ASMICORE_SUBMISSION_REPORT.md) — full ASMICORE submission narrative
- Multiclass: `model_comparison_table.csv`, `day5_evaluation_report.json`, `hist_gradient_boosting_final_bundle.pkl`
- Binary: `model_comparison_table_binary.csv`, `day5_evaluation_report_binary.json`, `gaussian_nb_final_bundle_binary.pkl`
- Shared: `preprocessor.pkl` / `preprocessor_binary.pkl`, `MODEL_USAGE_GUIDE.txt` / `MODEL_USAGE_GUIDE_binary.txt`

## Submission Summary

| Item | Value |
|------|-------|
| Dataset | 920 × 16 (UCI Heart Disease) |
| Features | 14 raw → 29 processed |
| Models | 22 per target mode (44 training runs via CLI) |
| Multiclass best | HistGradientBoosting, F1 0.6096 |
| Binary best | Gaussian NB, F1 0.8694 |
| JSON reports | 12+ under `outputs/reports/` |
| Preprocessing | Leakage-free (split before fit) |

## Conclusion

The project is a complete, reproducible submission for heart disease prediction with multiclass severity (0–4) and binary presence (0 vs 1). It includes six-day notebooks, dual-target CLI pipeline, structured JSON reports, visualizations, and deployable model bundles for both tasks. See [ASMICORE_SUBMISSION_REPORT.md](ASMICORE_SUBMISSION_REPORT.md) for the full section-by-section submission document.
