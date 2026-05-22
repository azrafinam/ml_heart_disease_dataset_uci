# ASMICORE SUBMISSION REPORT

## Heart Disease Prediction Using UCI Dataset

**Repository:** [azrafinam/ml_heart_disease_dataset_uci](https://github.com/azrafinam/ml_heart_disease_dataset_uci)  
**Language composition:** Jupyter Notebook (~96.8%), Python (~3.2%)

---

## Key Statistics at a Glance

| Metric | Multiclass (severity 0–4) | Binary (disease 0 vs 1) |
|--------|---------------------------|-------------------------|
| Dataset size | 920 rows × 16 columns | 920 rows × 16 columns |
| Features used | 14 (post-ID/target removal) | 14 (post-ID/target removal) |
| Processed features | 29 (post-encoding) | 29 (post-encoding) |
| Classes | 5 (0–4 severity) | 2 (0=no disease, 1=disease) |
| Train/test split | 80/20 (stratified) | 80/20 (stratified) |
| Training samples | 736 | 736 |
| Test samples | 184 | 184 |
| Models trained | 22 | 22 |
| Best model | HistGradientBoosting | Gaussian Naive Bayes |
| Best F1 score | 0.6096 | 0.8694 |
| Best ROC-AUC | 0.8293 | 0.9174 |
| JSON outputs | 12 structured reports | 12 structured reports (8 shared + 4 binary-specific) |
| Git commits | 1 (initial implementation) | — |
| Preprocessing safety | Leakage-free | Leakage-free |


---

## 1. LEARNING & UNDERSTANDING

### Dataset & Problem Understanding

- **Dataset:** UCI Heart Disease dataset with 920 rows × 16 columns
- **Source:** Kaggle UCI Heart Disease dataset
- **Target variable:** `num` (heart disease indicator)
- **Clinical relevance:** Supports early clinical investigation and preventive care

### Dual Target Formulations

The project models `num` in two complementary ways:

1. **Multiclass severity (primary notebook workflow):** five classes `0–4`  
   - `0` = no heart disease  
   - `1–4` = increasing disease severity  

2. **Binary presence (CLI pipeline + derived targets):** two classes  
   - `0` = no disease  
   - `1` = disease (original `num` values `1`, `2`, `3`, `4` collapsed to `1`)

Both formulations share the same features, leakage-safe preprocessing, and 22-model comparison zoo.

### Feature Composition

- **Numeric features (6):** age, trestbps (resting BP), chol (cholesterol), thalch (max heart rate), oldpeak (ST depression), ca (major vessels)
- **Categorical features (8):** sex, dataset (source location), cp (chest pain type), fbs (fasting blood sugar), restecg (resting ECG), exang (exercise-induced angina), slope (ST slope), thal (thalassemia)

### Data Quality Issues Identified

- 1,759 total missing values across the dataset
- Significant missing values in: ca (611), thal (486), slope (309)
- No duplicate rows detected

### Class Distribution — Multiclass (Imbalanced)

| Class | Label | Count | % |
|-------|-------|-------|---|
| 0 | No disease | 411 | 44.67% |
| 1 | Mild | 265 | 28.80% |
| 2 | Moderate | 109 | 11.85% |
| 3 | Moderate-severe | 107 | 11.63% |
| 4 | Severe | 28 | 3.04% |

### Class Distribution — Binary (Collapsed 1–4)

| Class | Label | Count | % |
|-------|-------|-------|---|
| 0 | No disease | 411 | 44.67% |
| 1 | Disease (num 1–4) | 509 | 55.33% |

**Train split (binary, stratified):** 329 no disease / 407 disease  
**Test split (binary, stratified):** 82 no disease / 102 disease

---

## 2. EXPLORATION & EXPERIMENTAL WORK

### Exploratory Data Analysis (EDA)

**Completed activities:**

- Dataset shape, dtypes, and missing value mapping
- Multiclass and binary target distribution analysis
- Numeric statistics (mean, median, std, min, max)
- Categorical value distributions
- Correlation heatmap analysis
- Boxplots and distribution histograms
- Feature importance via correlation with target

### Key Findings

1. **Numeric feature ranges:** Age 28–77, BP 0–200, cholesterol 0–603
2. **Missing value impact:** Requires careful handling before modeling
3. **Feature correlations:** Relationships between cardiac indicators and disease severity
4. **Class imbalance:** Multiclass severity class 4 is rare (3%); binary task is more balanced (~55% disease)

### Experiment Logs Generated

- `outputs/reports/day1_data_summary.json` — dataset overview
- `outputs/reports/day2_eda_report.json` — comprehensive EDA statistics
- `outputs/visualizations/` — distribution plots and heatmaps

---

## 3. IMPLEMENTATION & PREPROCESSING

### Feature Engineering Workflow

**Preprocessing steps (leakage-safe):**

1. **Train-test split (80/20):** Stratified by target, `random_state=42`  
   - Training: 736 samples × 29 features  
   - Testing: 184 samples × 29 features  

2. **Numeric imputation:** Median (fitted on training data only)  
   - Features: age, trestbps, chol, thalch, oldpeak, ca  

3. **Categorical imputation:** Most frequent value (fitted on training data only)  
   - Features: sex, dataset, cp, fbs, restecg, exang, slope, thal  

4. **Categorical encoding:** One-hot encoding (fitted on training data only)  
   - Produces 29 final processed features  

5. **Numeric scaling:** StandardScaler (fitted on training data only)  

### Why This Approach?

- **Data leakage prevention:** Preprocessing fitted only on training data
- **Real-world simulation:** Test set transformed with training preprocessor
- **Class balance preservation:** Stratified split maintains distribution per target mode

### Output Artifacts

**Multiclass:**

- `outputs/models/preprocessor.pkl`
- `data/processed/X_train_processed.csv`, `X_test_processed.csv`
- `data/processed/y_train.csv`, `y_test.csv`
- `outputs/reports/day3_preprocessing_metadata.json`

**Binary (same features; target encoded 0/1):**

- `outputs/models/preprocessor_binary.pkl`
- `data/processed/y_train_binary.csv`, `y_test_binary.csv`
- `outputs/reports/day4_training_summary_binary.json` (via CLI)

---

## 4. MODEL SELECTION & IMPLEMENTATION

### 22 Models Trained (Per Target Mode)

**Baseline:** Dummy Classifier  

**Linear models:** Logistic Regression, Ridge, SGD, Perceptron, LDA, Nearest Centroid  

**Distance-based:** K-Nearest Neighbors  

**Probabilistic:** Gaussian Naive Bayes  

**SVMs:** Linear SVM, RBF SVM  

**Tree-based:** Decision Tree, Random Forest, Extra Trees, Bagging, AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost  

**Hyperparameter tuning:** RandomizedSearchCV-tuned Random Forest  

`python scripts/main.py` trains all 22 models for **both** multiclass and binary targets in one reproducible run.

---

## 5. EVALUATION & PERFORMANCE METRICS

### Best Model — Multiclass Severity (0–4)

**HistGradientBoosting**

| Metric | Value |
|--------|-------|
| Accuracy | 0.6141 |
| Weighted precision | 0.6072 |
| Weighted recall | 0.6141 |
| Weighted F1 score | 0.6096 |
| Weighted ROC-AUC | 0.8293 |

#### Model Comparison Summary — Multiclass (Top 5)

| Rank | Model | F1 | Accuracy | ROC-AUC |
|------|-------|-----|----------|---------|
| 1 | HistGradientBoosting | 0.6096 | 0.6141 | 0.8293 |
| 2 | Gradient Boosting | 0.5926 | 0.6141 | 0.8322 |
| 3 | XGBoost | 0.5797 | 0.5924 | 0.8445 |
| 4 | Bagging | 0.5790 | 0.5924 | 0.8323 |
| 5 | Random Forest (RandomizedSearch) | 0.5784 | 0.6033 | 0.8572 |

---

### Best Model — Binary Presence (0 vs 1)

**Gaussian Naive Bayes**

| Metric | Value |
|--------|-------|
| Accuracy | 0.8696 |
| Weighted precision | 0.8694 |
| Weighted recall | 0.8696 |
| Weighted F1 score | 0.8694 |
| ROC-AUC | 0.9174 |

#### Model Comparison Summary — Binary (Top 5)

| Rank | Model | F1 | Accuracy | ROC-AUC |
|------|-------|-----|----------|---------|
| 1 | Gaussian Naive Bayes | 0.8694 | 0.8696 | 0.9174 |
| 2 | CatBoost | 0.8636 | 0.8641 | 0.9179 |
| 3 | LightGBM | 0.8631 | 0.8641 | 0.9140 |
| 4 | RBF SVM | 0.8624 | 0.8641 | 0.9218 |
| 5 | Random Forest (RandomizedSearch) | 0.8580 | 0.8587 | 0.9333 |

*Note: HistGradientBoosting remains strong on binary (F1 0.8474) but Gaussian NB ranks first on weighted F1 for this run.*

### Evaluation Artifacts

**Multiclass:**

- `outputs/reports/day5_evaluation_report.json`
- `outputs/reports/model_comparison_table.csv`
- `outputs/reports/day4_model_manifest.json`
- `outputs/models/hist_gradient_boosting_final_bundle.pkl`

**Binary:**

- `outputs/reports/day5_evaluation_report_binary.json`
- `outputs/reports/model_comparison_table_binary.csv`
- `outputs/reports/day4_model_manifest_binary.json`
- `outputs/models/gaussian_nb_final_bundle_binary.pkl`

### Selection Rationale

- **Weighted F1:** Accounts for class imbalance (especially multiclass class 4)
- **ROC-AUC:** Separation across probability thresholds (binary uses positive-class scores)
- **Held-out test set:** Identical split for fair comparison within each target mode
- **Reproducibility:** `random_state=42` throughout

---

## 6. GIT REPOSITORY STRUCTURE

```text
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

---

## 7. JSON OUTPUTS & STRUCTURED RESULTS

All results are stored in structured JSON (and CSV where noted):

| Artifact | Content |
|----------|---------|
| `day1_data_summary.json` | Shape, dtypes, missing values, target distribution |
| `day2_eda_report.json` | EDA statistics and target percentages |
| `day3_preprocessing_metadata.json` | Feature types, 29 processed features, split config |
| `day4_model_manifest.json` | 22 multiclass model paths |
| `day4_model_manifest_binary.json` | 22 binary model paths |
| `day4_training_summary.json` / `*_binary.json` | Training logs per target mode |
| `day5_evaluation_report.json` / `*_binary.json` | Full metrics; best model per mode |
| `day6_final_pipeline_report.json` / `*_binary.json` | Final bundle metadata per mode |

---

## 8. CHALLENGES & SOLUTIONS

| Challenge | Solution |
|-----------|----------|
| Class 4 severely underrepresented (3%) | Stratified split + weighted F1 (multiclass) |
| Two valid clinical views of `num` | Multiclass severity + binary presence pipelines |
| Missing values >50% in some columns | Median/mode imputation on training only |
| Ridge/Perceptron lack probability scores | ROC-AUC omitted where scores unavailable |
| Test set contamination risk | Split before preprocessing; fit only on train |
| 22 models × 2 targets | Modular `utils.py` + `main.py` with `target_mode` |

### Assumptions

1. All 14 features used after removing `id` and `num`  
2. Missing-at-random (MAR) for imputation  
    

### What Worked Well

- Leakage-safe preprocessing  
- Dual-target comparison with shared feature pipeline  
- 22-model zoo per formulation  
- JSON-structured reports and separate binary artifacts  

### What Could Be Improved

- Hyperparameter tuning beyond Random Forest  
- Cross-validation for robust estimates  
- Feature importance from tree models  
- Ensembles of top models per target mode  
- Threshold tuning for precision/recall tradeoffs  

---

## 9. DEPTH OF UNDERSTANDING

1. **Data understanding** — 920-row UCI dataset; 6 numeric + 8 categorical features; dual target encoding  
2. **EDA mastery** — Descriptive stats, imbalance analysis (multiclass and binary), correlations, visualizations  
3. **Preprocessing rigor** — Stratified 80/20 split; train-only fit for imputation, encoding, scaling  
4. **Model selection** — 22 algorithms per target; fair test-set comparison; best model per mode  
5. **Evaluation methodology** — Accuracy, precision, recall, F1, ROC-AUC; weighted F1 for imbalance  
6. **Pipeline integration** — Final bundles with preprocessor; usage guides for both modes  

---

## 10. REPRODUCIBILITY & REUSABILITY

```bash
conda env create -f environment.yml
conda activate ml_1
python scripts/main.py
```

**Key files:**

- `environment.yml` / `requirements.txt` — dependencies  
- `scripts/main.py` — full dual-target pipeline  
- `scripts/utils.py` — `encode_target()`, preprocessing, model zoo  

**Production artifacts:**

| Task | Bundle | Guide |
|------|--------|-------|
| Multiclass 0–4 | `outputs/models/hist_gradient_boosting_final_bundle.pkl` | `MODEL_USAGE_GUIDE.txt` |
| Binary 0/1 | `outputs/models/gaussian_nb_final_bundle_binary.pkl` | `MODEL_USAGE_GUIDE_binary.txt` |

---

## 11. CONCLUSION

This project delivers a complete, reproducible machine-learning workflow for heart disease prediction on the UCI dataset with **two supervised formulations**:

| Phase | Multiclass (0–4) | Binary (0 vs 1) |
|-------|------------------|-----------------|
| Learning | Severity levels as five classes | Disease presence with 1–4 collapsed |
| Exploration | EDA + imbalance on five classes | Binary distribution (~55% disease) |
| Implementation | 29 leakage-safe features | Same features, binary `y` |
| Model building | 22 models; best HistGradientBoosting | 22 models; best Gaussian NB |
| Evaluation | F1 **0.6096**, ROC-AUC **0.8293** | F1 **0.8694**, ROC-AUC **0.9174** |
| Integration | `hist_gradient_boosting_final_bundle.pkl` | `gaussian_nb_final_bundle_binary.pkl` |

**Best multiclass model:** HistGradientBoosting — F1 = 0.6096, ROC-AUC = 0.8293  
**Best binary model:** Gaussian Naive Bayes — F1 = 0.8694, ROC-AUC = 0.9174  

Documentation: this report, `README.md`, JSON reports under `outputs/reports/`, and six-day notebooks.
