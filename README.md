# Heart Disease Prediction Using Machine Learning

## Project Overview

This project builds a machine-learning workflow for the UCI Heart Disease dataset. The goal is to predict the `num` target, where `0` means no heart disease and `1-4` represent increasing heart disease severity.

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
- Target type: multiclass classification
- Target labels:
  - `0`: no heart disease
  - `1-4`: increasing heart disease severity

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

## Current Best Result

After the latest verified run, the best model is:

- Best model: `hist_gradient_boosting`
- Selection metric: weighted F1 score
- Accuracy: `0.6141`
- Weighted precision: `0.6072`
- Weighted recall: `0.6141`
- Weighted F1 score: `0.6096`
- Weighted multiclass ROC-AUC: `0.8293`

The rare severity classes have fewer samples, so class-wise performance should be interpreted carefully.

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
│   │   ├── *_trained.pkl
│   │   └── hist_gradient_boosting_final_bundle.pkl
│   ├── reports/
│   │   ├── day3_preprocessing_metadata.json
│   │   ├── day4_model_manifest.json
│   │   ├── day4_training_summary.json
│   │   ├── day5_evaluation_report.json
│   │   ├── day6_final_pipeline_report.json
│   │   ├── model_comparison_table.csv
│   │   └── MODEL_USAGE_GUIDE.txt
│   └── visualizations/
├── scripts/
│   ├── main.py
│   └── utils.py
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
  -> Train 22 models
  -> Evaluate on held-out test set
  -> Select best model by weighted F1
  -> Save final model + preprocessor bundle
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
- Saves train/test processed files and `outputs/models/preprocessor.pkl`.

### Day 4: Model Training

- Loads processed training data from Day 3.
- Trains the full 22-model zoo.
- Saves each trained model to `outputs/models/`.
- Saves `outputs/reports/day4_model_manifest.json`.

### Day 5: Model Evaluation

- Loads only models listed in the Day 4 manifest.
- Evaluates models on the held-out test set.
- Saves `outputs/reports/model_comparison_table.csv`.
- Saves `outputs/reports/day5_evaluation_report.json`.

### Day 6: Final Pipeline

- Loads the best model from Day 5.
- Loads the fitted preprocessor from Day 3.
- Saves the final model bundle:

```text
outputs/models/hist_gradient_boosting_final_bundle.pkl
```

- Saves final metadata and usage guide.

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

Or run the notebooks in order:

```text
Day 1 -> Day 2 -> Day 3 -> Day 4 -> Day 5 -> Day 6
```

## Using The Final Model

```python
import joblib
import pandas as pd

bundle = joblib.load("outputs/models/hist_gradient_boosting_final_bundle.pkl")
model = bundle["model"]
preprocessor = bundle["preprocessor"]

raw_rows = pd.DataFrame([...])  # original feature columns, excluding id and num
X_new_array = preprocessor.transform(raw_rows)
X_new = pd.DataFrame(X_new_array, columns=bundle["feature_names"])
predictions = model.predict(X_new)
```

## Important Methodology Notes

- The test set is not used to fit preprocessing.
- Model comparison uses the same held-out test split for every model.
- Weighted F1 is used for model selection because the target classes are imbalanced.
- ROC-AUC is reported where model scores are available; models without usable probability or decision scores may show missing ROC-AUC.
- The final artifact includes preprocessing so inference can start from raw feature rows.

## Main Outputs

- `outputs/reports/model_comparison_table.csv`
- `outputs/reports/day5_evaluation_report.json`
- `outputs/reports/day6_final_pipeline_report.json`
- `outputs/reports/MODEL_USAGE_GUIDE.txt`
- `outputs/models/preprocessor.pkl`
- `outputs/models/hist_gradient_boosting_final_bundle.pkl`

## Conclusion

The final project is a complete, reproducible machine-learning submission for multiclass heart disease prediction. It includes data exploration, EDA, leakage-safe preprocessing, broad model comparison, evaluation reports, visualizations, and a reusable final model bundle.
