# Learning Guide: Heart Disease Prediction, Step by Step

This guide teaches the project from the beginning. It follows the same six days as the notebooks, explains the idea behind each step, points to the file where the step happens, and ends each part with questions to check your understanding. Answers are hidden under each question; try to answer before opening them.

Read it next to the code. The notebooks in `notebooks/` show each step interactively; `scripts/utils.py` and `scripts/main.py` hold the final, reusable version of the same logic.

## Contents

0. [Before you start: the tools](#0-before-you-start-the-tools)
1. [Day 1: Know the data](#1-day-1-know-the-data)
2. [Day 2: Look at the data (EDA)](#2-day-2-look-at-the-data-eda)
3. [Day 3: Prepare the data without leaking](#3-day-3-prepare-the-data-without-leaking)
4. [Day 4: Train many models](#4-day-4-train-many-models)
5. [Day 5: Measure the models honestly](#5-day-5-measure-the-models-honestly)
6. [Day 6: Package the final model](#6-day-6-package-the-final-model)
7. [Reading the results critically](#7-reading-the-results-critically)
8. [Mistakes made along the way](#8-mistakes-made-along-the-way)
9. [Exercises to go further](#9-exercises-to-go-further)
10. [Glossary](#10-glossary)

---

## 0. Before you start: the tools

| Tool | Role in this project |
|------|----------------------|
| Conda | Creates the isolated `ml_1` environment so package versions do not clash with other projects |
| Jupyter notebooks | Run the code cell by cell and see tables and plots immediately |
| pandas, NumPy | Load and manipulate the table of patients |
| matplotlib, seaborn | Draw the EDA and evaluation plots |
| scikit-learn | Preprocessing, most models, metrics, train/test split |
| XGBoost, LightGBM, CatBoost | Three extra gradient boosting libraries |
| joblib | Saves trained models to `.pkl` files and loads them back |
| Git and GitHub | Version history and sharing |

Set up once:

```bash
conda env create -f environment.yml
conda activate ml_1
```

Why an environment file? Machine learning code depends on exact library versions. `environment.yml` records the full Conda environment; `requirements.txt` records the main pip packages with pinned versions. Anyone can rebuild the same setup and get the same results.

More detail: `logs/learning_notes.md`, sessions 01 and 02.

<details><summary>Check: what is the difference between pip and conda?</summary>

pip installs Python packages only. Conda manages whole environments, including the Python version itself and non-Python libraries. In this project Conda creates the environment and installs the core packages from conda-forge; pip is used only for packages Conda does not provide.
</details>

---

## 1. Day 1: Know the data

**Notebook:** `notebooks/day1_data_exploration_IMPROVED.ipynb`
**Output:** `outputs/reports/day1_data_summary.json`

### What happens

The CSV is loaded with `pd.read_csv(path, na_values="?")`. The `?` symbol marks unknown values in the original UCI files, so it is turned into a proper missing value (`NaN`).

Then the basic facts are collected:

| Fact | Value |
|------|-------|
| Rows (patients) | 920 |
| Columns | 16 |
| Identifier | `id` (not a feature) |
| Target | `num` |
| Numeric features | age, trestbps (resting blood pressure), chol (cholesterol), thalch (maximum heart rate), oldpeak (ST depression), ca (number of major vessels) |
| Categorical features | sex, dataset (hospital), cp (chest pain type), fbs (fasting blood sugar), restecg (resting ECG), exang (exercise angina), slope (ST slope), thal |
| Missing values | 1,759 cells; `ca` 611, `thal` 486, `slope` 309 |
| Duplicate rows | none |

### The idea

Before any modelling you need to know four things: how many rows, what each column means, which column you predict, and what is missing. Every later decision depends on these.

### The target

| `num` | Meaning | Patients | Share |
|-------|---------|----------|-------|
| 0 | No disease | 411 | 44.7 % |
| 1 | Mild | 265 | 28.8 % |
| 2 | Moderate | 109 | 11.8 % |
| 3 | Moderate to severe | 107 | 11.6 % |
| 4 | Severe | 28 | 3.0 % |

Merging 1 to 4 gives the binary target: 411 without disease, 509 with disease. This is done by `encode_target()` in `scripts/utils.py`.

<details><summary>Check: why is <code>id</code> removed before training?</summary>

`id` is only a row number. It carries no medical information. If kept, a model could memorise it and appear to learn patterns that do not exist.
</details>

<details><summary>Check: two thirds of <code>ca</code> is missing. Why not just delete the rows with missing values?</summary>

Deleting every row with any missing value would remove most of the dataset, because `ca`, `thal` and `slope` are missing for hundreds of patients. Imputation (filling in values) keeps all 920 patients. The cost is that filled values are guesses, so the model must not rely on them too much.
</details>

---

## 2. Day 2: Look at the data (EDA)

**Notebook:** `notebooks/day2_eda_IMPROVED.ipynb`
**Outputs:** `outputs/visualizations/day2_*.png`, `outputs/reports/day2_eda_report.json`

### What happens

| Plot | Question it answers |
|------|---------------------|
| Target distribution | Are the classes balanced? (No: class 4 has only 3 %) |
| Numeric distributions | What range and shape does each measurement have? |
| Categorical distributions | How common is each category? |
| Correlation heatmap | Which numeric features move together, and with the target? |
| Boxplots | Where are the outliers? |

### The idea

Exploratory data analysis (EDA) means looking before modelling. Plots reveal problems that summary numbers hide, such as impossible values, skewed distributions and rare classes.

### A finding worth noticing

The cholesterol distribution has a spike at exactly 0. A living patient cannot have zero cholesterol: these 172 zeros are missing values recorded as 0. They come from two hospitals (all 123 Switzerland patients and 49 VA Long Beach patients). The current pipeline does not treat them as missing; see section 7.

<details><summary>Check: what does a correlation of -0.4 between two features mean?</summary>

When one feature goes up, the other tends to go down, with moderate strength. Correlation measures only straight-line relationships, and it does not prove that one causes the other.
</details>

<details><summary>Check: why does class imbalance matter before you train anything?</summary>

A model can score well on accuracy by mostly predicting the common classes and ignoring the rare ones. Knowing the imbalance early decides two things later: the split must be stratified, and the selection metric must not be plain accuracy.
</details>

---

## 3. Day 3: Prepare the data without leaking

**Notebook:** `notebooks/day3_feature_engineering.ipynb`
**Code:** `create_preprocessor()` and `prepare_train_test_data()` in `scripts/utils.py`
**Outputs:** `data/processed/`, `outputs/models/preprocessor.pkl`, `outputs/reports/day3_preprocessing_metadata.json`

This is the most important day. A mistake here makes every later number wrong.

### What happens, in order

1. **Split first.** `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` gives 736 training and 184 test patients. `stratify=y` keeps the same class shares in both parts.
2. **Fit the preprocessor on the training part only.**

| Feature type | Step 1 | Step 2 |
|--------------|--------|--------|
| Numeric | Fill missing with the median | Standardise to mean 0, standard deviation 1 |
| Categorical | Fill missing with the most frequent value | One-hot encode (one 0/1 column per category) |

3. **Apply the same fitted preprocessor to the test part.** 14 raw features become 29 processed features.

### The idea: data leakage

Leakage happens when information from the test set influences training. Example: if the median cholesterol were computed on all 920 patients and then used to fill gaps, the training data would contain a number partly derived from test patients. The test score would then be slightly too optimistic, and the model would look better than it will be on new patients.

The rule: **the test set is a sealed envelope.** Everything that is learned from data (medians, scaling factors, category lists, model weights) is learned from the training set only.

The first version of this project preprocessed the whole dataset before splitting. The final version fixes this.

### Why one-hot encoding and not label encoding?

Label encoding turns categories into 0, 1, 2, 3. A linear model then reads "typical angina = 0, asymptomatic = 3" as if asymptomatic were three times something, which is meaningless. One-hot encoding gives each category its own 0/1 column, so no false order is created. `handle_unknown="ignore"` keeps prediction working if a new category appears later.

### Why scale?

Cholesterol is in the hundreds; oldpeak is around 0 to 6. Distance-based models (KNN, SVM) and gradient-based models (logistic regression, SGD) would treat cholesterol as more important only because its numbers are larger. Scaling puts all numeric features on the same footing. Tree models do not need scaling, but it does not hurt them.

<details><summary>Check: why median and not mean for numeric imputation?</summary>

The median is not pulled by extreme values. Medical measurements often have outliers (for example very high cholesterol), and the median gives a more typical filler value.
</details>

<details><summary>Check: what goes wrong if you call <code>fit_transform</code> on the test set?</summary>

The preprocessor would learn new medians, scaling factors and categories from the test data. The test set would then be processed differently from the training set, and information from the test set would leak into the pipeline. On the test set you only call `transform`.
</details>

---

## 4. Day 4: Train many models

**Notebook:** `notebooks/day4_model_training.ipynb`
**Code:** `build_model_zoo()` in `scripts/utils.py`
**Outputs:** `outputs/models/*_trained.pkl`, `outputs/reports/day4_model_manifest.json`

### What happens

22 classifiers are trained on the same processed training data. Each is saved, and a manifest file lists them so Day 5 loads exactly the models that were trained.

### The idea: model families

| Family | How it decides | Models here | Strength | Weakness |
|--------|----------------|-------------|----------|----------|
| Baseline | Always predicts the most common class | Dummy | Shows the score you get for free | Learns nothing |
| Linear | Weighted sum of features, then a boundary | Logistic Regression, Ridge, SGD, Perceptron, Linear SVM, LDA | Fast, easy to interpret | Cannot capture curved boundaries without extra features |
| Distance | Looks at the closest training patients | KNN, Nearest Centroid | Simple idea | Sensitive to scaling and to irrelevant features |
| Kernel | Maps data to a space where a boundary is easier | RBF SVM | Handles curved boundaries | Slow on large data, needs tuning |
| Probabilistic | Bayes' rule, assumes features are independent | Gaussian Naive Bayes | Very fast, works on small data | Independence assumption is rarely true |
| Single tree | A chain of yes/no questions | Decision Tree | Readable | Overfits easily |
| Bagging | Many trees on random samples, then a vote | Random Forest, Extra Trees, Bagging | Stable, strong default | Less interpretable |
| Boosting | Trees built one after another, each fixing the previous errors | AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost | Often the best on tables | More settings to tune, can overfit |

The tuned Random Forest uses `RandomizedSearchCV`: it tries 8 random combinations of settings and scores each with 3-fold stratified cross-validation on the training data, then keeps the best.

### Why always include a dummy baseline?

A score means nothing without a reference. The dummy model reaches 0.554 accuracy on the binary task just by always saying "disease" (102 of the 184 test patients have disease). A real model must beat this clearly.

<details><summary>Check: what is overfitting, and which model here is most at risk?</summary>

Overfitting means the model learns the noise of the training set and performs worse on new data. The single Decision Tree is most at risk because it can keep splitting until it memorises the training patients. Here it is limited with `max_depth=10` and `min_samples_split=10`, and it still ranks near the bottom on both tasks.
</details>

<details><summary>Check: why might Gaussian Naive Bayes win on a small dataset?</summary>

It has very few parameters to learn (a mean and variance per feature per class), so it needs little data and does not overfit easily. Complex models need more data to show their advantage. With only 736 training patients, a simple model can match or beat a complex one.
</details>

---

## 5. Day 5: Measure the models honestly

**Notebook:** `notebooks/day5_evaluation.ipynb`
**Code:** `evaluate_classifier()`, `get_model_scores()`, `create_model_comparison_report()` in `scripts/utils.py`
**Outputs:** `outputs/reports/model_comparison_table*.csv`, `day5_evaluation_report*.json`, `outputs/visualizations/day5_*.png`

### The metrics

Think of the binary task: "disease" is the positive class.

| Metric | Question it answers |
|--------|---------------------|
| Accuracy | What share of all predictions is correct? |
| Precision | Of the patients the model calls sick, how many are really sick? |
| Recall | Of the patients who are really sick, how many does the model find? |
| F1 | Balance of precision and recall (their harmonic mean) |
| ROC-AUC | If you pick one sick and one healthy patient at random, how often does the model score the sick one higher? 0.5 is guessing, 1.0 is perfect |

**Weighted** precision, recall and F1 compute the metric for each class and average them, weighted by class size. This project selects the best model by weighted F1.

For a medical screening tool, recall usually matters most: missing a sick patient (false negative) is worse than sending a healthy patient for another test (false positive).

### The confusion matrix

A table of true class against predicted class. The diagonal holds correct predictions; everything else shows which classes are confused with which. See `outputs/visualizations/day5_best_model_confusion_matrix.png`.

### Why some models have no ROC-AUC

ROC-AUC needs a score, not just a label. Ridge, Linear SVM and Perceptron give no probabilities, so `get_model_scores()` falls back to their `decision_function()` scores. That works for the binary task. Multiclass ROC-AUC needs class probabilities that add up to 1, which raw decision scores are not, so ROC-AUC is left empty for these three in the multiclass table.

<details><summary>Check: a model has 0.61 accuracy but 0.83 ROC-AUC on the multiclass task. How can both be true?</summary>

ROC-AUC measures ranking: the model usually gives higher scores to the right class. Accuracy measures the final hard choice. The model may often have the right class as its second choice, which hurts accuracy but not ranking. In the multiclass task this happens between neighbouring severity levels, for example 2 and 3.
</details>

<details><summary>Check: why not choose the model by accuracy?</summary>

With imbalanced classes, accuracy rewards ignoring rare classes. Weighted F1 takes precision and recall of each class into account, so it penalises a model that never predicts the small classes.
</details>

---

## 6. Day 6: Package the final model

**Notebook:** `notebooks/day6_final_pipeline_IMPROVED.ipynb`
**Code:** `scripts/main.py`
**Outputs:** `outputs/models/*_final_bundle*.pkl`, `outputs/reports/day6_final_pipeline_report*.json`, `MODEL_USAGE_GUIDE*.txt`

### What happens

The best model and the fitted preprocessor are saved together in one bundle, with the feature names:

```python
bundle = {"model": ..., "preprocessor": ..., "feature_names": [...]}
```

`scripts/main.py` repeats Days 3 to 6 automatically, first for the multiclass target and then for the binary target, and writes every report and model file.

### The idea

A model alone is not usable: it expects 29 scaled, encoded columns, not the 14 raw measurements a doctor would type in. Saving the preprocessor with the model means new data goes through exactly the same steps as the training data.

The script exists so the result can be reproduced with one command. Notebooks are good for learning and exploring; scripts are good for repeating the same run reliably.

<details><summary>Check: what happens if you load only the model and feed it raw patient data?</summary>

It fails or gives wrong answers. The model was trained on 29 processed columns. Raw data has 14 columns, unscaled numbers, text categories and missing values. The preprocessor must transform the data first.
</details>

---

## 7. Reading the results critically

| Task | Best model | Accuracy | Weighted F1 | ROC-AUC | Dummy accuracy |
|------|------------|----------|-------------|---------|----------------|
| Binary | Gaussian Naive Bayes | 0.870 | 0.869 | 0.917 | 0.554 |
| Multiclass | HistGradientBoosting | 0.614 | 0.610 | 0.829 | 0.446 |

### Why binary is so much easier

The features clearly separate sick from healthy patients. They separate severity levels poorly: a patient at level 2 looks much like one at level 3. Class 4 has only 28 patients in total, and about 6 in the test set, too few to learn from.

### How sure is the ranking?

The test set has 184 patients, so one patient changes accuracy by 0.0054. Gaussian NB (0.870) beats CatBoost, LightGBM and RBF SVM (0.864) by exactly one patient. With a different random split the order could change. The honest conclusion is: "several models reach about 86 to 87 % accuracy on the binary task", not "Gaussian NB is the best model".

Stratified k-fold cross-validation on the training data (for example 5 folds) would give each model five scores; the mean and the spread show which differences are real.

### Two hidden problems in the data

1. **Zero cholesterol.** 172 patients have `chol = 0`, which means "not measured". The imputer only fills empty cells, so these zeros are treated as real measurements. They should be replaced by `NaN` before the split so the median imputer handles them.
2. **The hospital column.** `dataset` says which hospital the patient came from. Disease rates differ strongly between hospitals:

| Hospital | Patients | With disease | Zero cholesterol | `ca` missing |
|----------|----------|--------------|------------------|--------------|
| Cleveland | 304 | 45.7 % | 0 | 5 |
| Hungary | 293 | 36.2 % | 0 | 290 |
| Switzerland | 123 | 93.5 % | 123 | 118 |
| VA Long Beach | 200 | 74.5 % | 49 | 198 |

A model can partly learn "Switzerland means disease" instead of learning from the patient's measurements. That works on this dataset but not for a patient from a new hospital. Training once with and once without `dataset` shows how much the models depend on it.

---

## 8. Mistakes made along the way

These come from `logs/issue_log.md`. Each one is a common beginner mistake.

| Mistake | Effect | Fix |
|---------|--------|-----|
| Preprocessing before splitting | Test scores slightly too optimistic (leakage) | Split first, fit preprocessing on training data only |
| Label encoding for categories | False order between categories | One-hot encoding |
| Training on data with `NaN` | Many scikit-learn models crash | Imputation inside the preprocessing pipeline |
| Assuming every model has `predict_proba()` | ROC-AUC calculation crashed | Check for scores, fall back to `decision_function()`, otherwise skip |
| Saving pandas types to JSON | `TypeError: Object of type Int64DType is not JSON serializable` | Convert to plain Python types (`str`, `int`, `float`) before `json.dump` |
| `git init` in the home folder | Git tried to track the whole home directory | Remove that repository; run `git init` inside the project folder |
| Saving plots with `plt.savefig` | Model comparison plots were not saved correctly | Keep the figure object and call `fig.savefig` |
| Judging models by one score only | Unstable models (Decision Tree, KNN) looked fine | Cross-validation with mean and standard deviation |

---

## 9. Exercises to go further

Try these in order; each builds on the previous one.

1. **Fix the zero cholesterol.** Replace `chol == 0` (and the one `trestbps == 0`) with `NaN` in `load_data()`. Rerun `python scripts/main.py` and compare the tables.
2. **Remove the hospital column.** Add `dataset` to the dropped columns and rerun. How much do the scores fall? What does that tell you?
3. **Cross-validate.** For the top five binary models, run `cross_val_score` with `StratifiedKFold(n_splits=5)` on the training data. Report mean and standard deviation of weighted F1. Is the ranking still the same?
4. **Tune a boosting model.** Use `RandomizedSearchCV` on HistGradientBoosting (learning rate, max depth, max iterations), the same way Random Forest is tuned in `build_model_zoo()`.
5. **Favour recall.** For the binary model, lower the decision threshold from 0.5 (for example to 0.35) using `predict_proba`. How do precision and recall change?
6. **Handle class 4.** For the multiclass task, try `class_weight="balanced"` on the models that support it, or merge classes 3 and 4. Does recall on the severe class improve?

---

## 10. Glossary

| Term | Meaning |
|------|---------|
| Feature | An input column used for prediction, such as age |
| Target (label) | The column to predict, here `num` |
| Training set | Data the model learns from |
| Test set | Data kept aside to measure performance on unseen patients |
| Stratified split | A split that keeps the same class shares in each part |
| Imputation | Filling missing values |
| One-hot encoding | One 0/1 column per category |
| Standardisation | Rescaling a feature to mean 0 and standard deviation 1 |
| Data leakage | Test information reaching the training process |
| Overfitting | Learning noise in the training data; good training score, worse test score |
| Cross-validation | Repeating training and testing on several splits to get a more stable estimate |
| Hyperparameter | A setting chosen before training, such as tree depth |
| Ensemble | A model that combines many smaller models |
| Weighted F1 | F1 per class, averaged by class size |
| ROC-AUC | How well the model ranks positive cases above negative ones |
| Bundle | The saved model together with its preprocessor |
