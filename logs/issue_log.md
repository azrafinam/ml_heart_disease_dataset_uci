
## Challenges Faced and Solutions Implemented
Throughout the 6-day machine learning project, several implementation challenges and development mistakes were identified and resolved. 

## Key Fixes

- Fixed data leakage by splitting train/test data before preprocessing.
- Fit `SimpleImputer`, `OneHotEncoder`, and `StandardScaler` on the training set only.
- Replaced categorical `LabelEncoder` with `OneHotEncoder(handle_unknown='ignore')`.
- Restored and expanded model comparison to 22 classifiers.
- Saved the final artifact as a bundle containing both the best model and fitted preprocessor.
- Aligned notebooks, `scripts/main.py`, `scripts/utils.py`, reports, dependencies, and documentation.

## Other Fixes 

### Missing Value Handling During Model Training

During Day 5 model evaluation, a ValueError occurred while running the Logistic Regression model. The issue happened because the feature dataset contained missing values (NaN), and Scikit-learn’s LogisticRegression model does not support missing values directly.

The error message was:
```python
ValueError: Input X contains NaN.
LogisticRegression does not accept missing values encoded as NaN natively.
```
To investigate the issue, missing value analysis was performed using:
```python
print("NaNs in X:", X.isnull().sum().sum())
print("NaNs in y:", y.isnull().sum())

print("\nColumns with NaNs:")
print(X.columns[X.isnull().any()].tolist())
```
The output showed that the dataset contained 817 missing values distributed across the following 

columns:
trestbps
chol
thalch
oldpeak
ca


**SOLVE**
To fix the issue, median imputation was applied before model training:

### JSON Sterilaztion issue
While generating the dataset summary report in Day 1, an error occurred during JSON serialization. 


Initially, Pandas datatype objects were directly passed into Python’s `json.dump()` function, which caused serialization failures. This highlighted the importance of validating data types before exporting reports into JSON format.


The original implementation produced the following error:
```python
TypeError: Object of type Int64DType is not JSON serializable
```
The issue occurred because the summary dictionary contained Pandas datatype objects such as Int64DType, which cannot be automatically converted into JSON format.

To solve this problem, the serialization process was modified by adding ```default=str``` inside json.dump():
```python
with open('../outputs/reports/day1_data_summary.json', 'w') as f:
    json.dump(summary, f, indent=4, default=str)
```
Using ```default=str``` converted unsupported Pandas objects into string format automatically, allowing the JSON report to be generated successfully without data loss.  

**Lesson Learned:**  
Always ensure that complex objects such as Pandas dtypes are converted into serializable formats before saving structured outputs.


### Git Project Setup and Fixes

#### Mistake: Initializing Git in Home Directory

At first, Git was accidentally initialized in the home directory (`~`):

```bash
git init
```
This caused Git to treat the entire system user directory as a repository.

As a result, Git started tracking unrelated system files such as:

- `.bashrc`
- `.ssh/`
- `.conda/`
- `.config/`
- `Downloads/`
- `Desktop/`
####  Fixing the Wrong Git Initialization

To remove incorrect Git tracking from the home directory:

```bash id="p6xq8n"
rm -rf ~/.git
```


**Lesson Learned:**

```bash id="x8m4qv"
mkdir ~/projects/my_project
cd ~/projects/my_project
git init
```
Now Git only tracks files inside this project folder.




###  Underestimating Dataset Validation
At the beginning of the project, insufficient validation checks were performed on processed datasets. Missing values and datatype inconsistencies were discovered later during model evaluation.

**Lesson Learned:**  
Validation should be included after every major preprocessing step to detect issues early in the workflow.


### 9. README Formatting and Documentation Issues
The initial workflow diagram used plain ASCII formatting, which rendered poorly on GitHub and reduced documentation readability.

**Lesson Learned:**  
Project documentation should be designed specifically for the target platform. GitHub Markdown supports Mermaid diagrams, which provide cleaner and more maintainable workflow visualizations.


### 10. ROC-AUC Evaluation Assumptions
ROC-AUC calculations initially assumed that all models supported probability prediction. Some models failed because `predict_proba()` was unavailable.

**Lesson Learned:**  
Evaluation pipelines should include conditional checks and exception handling to support different model behaviors safely.


###  Environment and Dependency Management
Initially, dependency tracking was incomplete, and package versions were not properly documented.

**Lesson Learned:**  
Environment reproducibility is critical for machine learning projects. Using Conda environment files and maintaining `requirements.txt` ensures consistent execution across systems.


###  Project Structure Organization
Some files such as logs, utility scripts, and pipeline entry points were initially empty or unorganized.

**Lesson Learned:**  
A clean and maintainable project structure improves readability, collaboration, debugging, and deployment readiness.


### Notebook Kernel Crash Handling
While working with Jupyter notebooks inside VS Code, kernel crashes occasionally occurred due to memory issues, package conflicts, or notebook execution problems.
##### Troubleshooting Steps Followed
1. Used the **Restart Kernel** button inside the notebook interface.
2. Reloaded or restarted VS Code when kernel restarting alone did not resolve the issue.
This represents a standard troubleshooting workflow commonly used for notebook-based development environments.
---


By the end of the project, a great deal of knowledge was gathered about not only machine learning but also about linux, conda based environments, vscode, cursor, github and git.