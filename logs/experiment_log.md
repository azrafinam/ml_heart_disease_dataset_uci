> First-version log from the early draft (multiclass target turned into binary, label encoding). The current pipeline replaced label encoding with one-hot encoding and keeps both the multiclass and binary tasks; see `ASMICORE_SUBMISSION_REPORT.md`.

# Experiment Log

## Day 1
- Loaded heart disease dataset
- Analyzed dataset structure
- Generated summary statistics

## Day 2
- Performed EDA
- Created correlation heatmaps
- Analyzed class distribution

## Day 3
- Converted multiclass target into binary classification
- Applied Label Encoding
- Standardized numerical features
- Saved processed datasets

## Day 4
Trained multiple models:

- Logistic Regression
- KNN
- Decision Tree
- Random Forest
- XGBoost

Tested multiple preprocessing strategies:

- Median imputation
- Drop-NaN strategy
- IQR outlier removal

## Day 5
Evaluated models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Performed cross-validation and confusion matrix analysis.

## Day 6
Built complete ML pipeline:

- Automated preprocessing
- Automated training
- Automated evaluation
- Model saving
- JSON report generation