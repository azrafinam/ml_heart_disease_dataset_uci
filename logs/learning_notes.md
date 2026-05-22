# Project Learning Timeline:
## Learning Sessions 01 to 09

| Session | Topic | Outcome |
|---------|-------|---------|
| 01 | Conda, pip, environments | Reproducible `ml_1` env |
| 02 | Jupyter, Git, VS Code | Six-day notebook workflow |
| 03 | NumPy, Pandas, dataset shape | 920×16, 14 features, target `num` |
| 04 | Boxplots, heatmaps, imbalance | EDA before modeling |
| 05 | Precision, recall, F1, ROC, CV | How to score classifiers |
| 06 | Preprocessing, leakage | 29 features, safe split |
| 07 | Model types (22 algorithms) | When to use linear / tree / boosting / NB |
| 08 | Dual target, comparison | Multiclass vs binary results |
| 09 | Pipeline code, reproducibility | `main.py` + bundles + reports |

---
---
# LEARNING SESSION 01
---
# ML ENVIRONMENT SETUP
---
---

# 1. CONDA ENVIRONMENT

### Conda Environment Structure

A Conda environment is an isolated workspace that contains:

- A specific Python installation
- Its own pip installer
- Installed Python packages
- System-level dependencies (in some cases)

#### Structure

```
System
 └── Conda
      └── Environment (ml_1)
           └── Python
                └── pip
                     └── packages (numpy, pandas, etc.)
```

---

### What is pip?

- pip is the default Python package installer
- It installs packages from the Python Package Index (PyPI)

🔗 https://pypi.org

##### Example

```bash
pip install numpy
```

##### Key idea

- pip installs packages inside the active Python environment
- If a Conda environment is active, pip installs only inside that environment

---

### What is Conda?

- Conda is an environment and package manager
- It manages:
  - Python versions
  - Libraries
  - System dependencies (C/C++/CUDA, etc.)

#### Example

```bash
conda create -n ml python=3.11.15
conda activate ml_1
```

---

### What is conda-forge?

- conda-forge is a community-maintained Conda package channel
- It provides pre-built packages for scientific computing and ML

#### Why it exists

- More packages available than default channels
- Faster updates
- Better compatibility for scientific libraries

#### Example usage

```bash
conda install -c conda-forge numpy
```

---

### What is a Conda channel?

A channel is a repository where Conda searches for packages.

Common channels:

- defaults (official Anaconda channel)
- conda-forge (community channel)

#### Example

```bash
conda install -c conda-forge pandas
```

---

### Why conda-forge appears automatically

Some Conda distributions are configured to use conda-forge by default.

So even if you don’t specify it, it may already be included in:

```
channels:
  - conda-forge
  - defaults
```

---

### pip vs conda

| Feature | pip | conda |
|--------|-----|-------|
| Source | PyPI | Conda channels |
| Installs | Python packages only | Python + system libraries |
| Dependency solving | Weak | Strong |
| Precompiled binaries | Sometimes | Yes (usually) |
| Best for | Pure Python projects | ML, data science, scientific computing |

---

## Key Relationship

- pip is always tied to a Python installation
- Python is inside a Conda environment
- Therefore pip installs inside the Conda environment (if active)

#### Example

```bash
conda activate ml
pip install flask
```

Installs Flask only inside `ml` environment.

---

### pip vs conda install behavior

#### pip install

- Downloads from PyPI
- May compile locally
- Requires system dependencies sometimes

```bash
pip install numpy
```

### conda install

- Downloads prebuilt binaries
- Handles dependencies automatically
- More stable for scientific packages

```bash
conda install numpy
```

---

### Selecting Best Practice Workflow

#### Step 1: Create environment

```bash
conda create -n ml python=3.12
conda activate ml
```

#### Step 2: Install core packages via conda-forge

```bash
conda install -c conda-forge numpy pandas matplotlib scikit-learn jupyter
```

#### Step 3: Use pip only if needed

```bash
pip install some-package-not-in-conda
```

---

### Channel priority (recommended setup)

To avoid dependency conflicts:

```bash
conda config --set channel_priority strict
```

---

## Conda Command Understanding

An error occurred when running:

```bash
def conda list numpy pandas jupyter matplotlib scikit-learn
```
def does not accept multiple package names.

**Correct Usage:**
- `conda list` → show all packages
- `conda list package_name` → filter single package
##### Other commands: 
```bash
    conda info
```
```bash
    which conda
```
## Core Idea

- Conda = environment manager + package manager
- pip = Python package installer inside that environment
- conda-forge = large community package repository

Conda Environment Setup (ML Environment)

I created and used a dedicated Conda environment called `ml_1` for machine learning work. This environment isolates ML libraries from the base system to avoid dependency conflicts.

### Key Observations
- Installed core ML libraries:
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - jupyter
- Conda automatically handles dependency resolution.
- Verified installation using:

```bash
conda list | grep -E "numpy|pandas|jupyter|matplotlib|scikit-learn"
```
Outcome:
All required ML packages are successfully installed and working inside the `ml_1` environment.




# 2. **PYTHON ENVIRONMENT**

### Understanding `pyenv`, `venv`, and `conda`

During the project setup and machine learning workflow preparation, I learned the differences between Python environment and version management tools.

---

## 1. `pyenv`

`pyenv` is mainly a **Python version manager**.

It allows multiple Python interpreters to exist on the same machine.

Example:

```bash
pyenv install 3.11.9
pyenv install 3.12.4
```

This helps when different projects require different Python versions.

Example:

- Project A → Python 3.9
- Project B → Python 3.12

`pyenv` can switch between them easily:

```bash
pyenv global 3.12.4
```

or inside a specific project:

```bash
pyenv local 3.9.18
```

##### Main Purpose

- Manage Python interpreter versions
- Switch Python versions per project
- Lightweight tool

##### Important Note

`pyenv` does **not** primarily manage scientific libraries or complex dependencies.

---

## 2. `venv`

`venv` is Python’s built-in virtual environment system.

Example:

```bash
python -m venv myenv
```

This creates an isolated Python environment containing:

- Separate Python executable
- Separate pip
- Separate installed packages

Activation:

```bash
source myenv/bin/activate
```

Packages installed afterward remain isolated inside the environment.

Example:

```bash
pip install numpy
```

##### Main Purpose

- Isolate Python packages per project
- Prevent package conflicts
- Lightweight and simple

##### Limitation

`venv` only manages Python-related packages and dependencies.

It does not handle:

- CUDA
- System libraries
- Non-Python binaries
- Scientific compiled dependencies

---

## 3. `conda`

As stated earlier, `conda` is a broader environment and package management system.


A conda environment may contain:

- Python
- pip packages
- Scientific libraries
- CUDA dependencies
- C/C++ libraries
- Numerical computing binaries

##### Main Purpose

- Full environment isolation
- Dependency management
- Scientific computing support
- Machine learning workflows

---

### Key Difference Between `venv` and `conda`

#### `venv`

Acts like:

> A separate Python package room

Only isolates Python packages.

---

#### `conda`

Acts like:

> A complete isolated laboratory

Manages:

- Python
- Packages
- Scientific dependencies
- Non-Python binaries

---

## Typical Usage Patterns

#### Software/Web Development

Common stack:

```text
pyenv + venv + pip
```

Reason:

- Lightweight
- Simple
- Fast

---

#### Machine Learning / Data Science

Common stack:

```text
conda
```

Reason:

- Easier dependency management
- Better scientific package support
- Handles compiled libraries more effectively

---

#### Important Understanding

A `venv` is conceptually similar to a lightweight Conda environment that mainly isolates Python packages only.

Conda environments are broader and can isolate entire scientific software ecosystems.

---

#### Current Workflow Decision

For machine learning and Jupyter-based workflows, Conda is more suitable because the project involves:

- pandas
- numpy
- scikit-learn
- Jupyter
- potentially GPU/scientific dependencies later

Current setup direction:

```text
Conda + Jupyter + VS Code
```

This is a common and scalable ML development workflow.


---
---
# LEARNING SESSION 02
---
# JUPYTER NOTEBOOK, GIT & VS CODE
---
---

# 1. JUPYTER NOTEBOOK 

## What is Jupyter Notebook?
Jupyter Notebook is an interactive computing environment used mainly for:
- Data science
- Machine learning
- Data analysis
- Visualization
- Experimentation

It allows you to write and run code in small sections called **cells**, instead of running an entire script at once.

---

## How it works
Jupyter has two main components:

### 1. Frontend (Interface)
- Runs in a web browser or inside VS Code
- Displays cells, outputs, graphs, and text

### 2. Kernel (Execution Engine)
- Runs Python code in the background
- Stores variables in memory while active
- Executes each cell and returns output to the interface

---


## Cell Execution Concept
Each cell can be executed independently, and output appears immediately below it.

---

## Important Feature: Persistent Memory
Variables remain in memory across cells.

Example:
If `x = 10` is defined in one cell, it can be used in another cell without redefining it.

 This can cause confusion if cells are not run in order.

---

## File Type
Jupyter notebooks use:

- `.ipynb` files

These files store:
- Code
- Output
- Graphs
- Text explanations

---

## Why it is used in Machine Learning
Jupyter Notebook is widely used in ML because it allows:

- Step-by-step experimentation  
- Easy data inspection  
- Instant visualization (graphs, plots)  
- Fast testing of models and parameters  

---

## Typical Workflow in ML
1. Import libraries  
2. Load dataset  
3. Explore data  
4. Preprocess data  
5. Train model  
6. Evaluate model  
7. Export results (e.g., JSON)  

---

## Jupyter in VS Code vs Separate 

### Option 1: Separate Jupyter Notebook
- Runs in browser  
- Started using terminal:
## Setup

Jupyter was started using:

```bash
jupyter notebook
```
**What happens internally:**
- A local server runs at `localhost:8888`
- Browser opens as UI interface
- Python kernel runs inside Conda environment
**Important Concept:**
- Jupyter has two components:
  - Server (Terminal) → executes code
  - Browser → interface only
- Stopping the server (`Ctrl + C`) disconnects execution but does NOT close the browser tab.
---

### Option 2: Inside Visual Studio Code
- Requires Python extension  
- Requires Jupyter extension  
- Opens `.ipynb` files directly inside VS Code  

This provides the same functionality but inside a code editor.

---

## VS Code + Jupyter Setup
To use Jupyter in VS Code:

- Install Python extension  
- Install Jupyter extension  
- Create a `.ipynb` file  
- Select Python kernel when prompted  

---

## Difference between `.py` and `.ipynb`

| Python Script (.py) | Jupyter Notebook (.ipynb) |
|--------------------|--------------------------|
| Runs full file at once | Runs cell-by-cell |
| Better for final programs | Better for experiments |
| Plain code execution | Interactive output + visuals |

---

## Jupyter Notebook Shortcuts

| Shortcut | Action |
|---|---|
| `Shift + Enter` | Run cell and move to next cell |
| `Ctrl + Enter` | Run cell and stay on same cell |
| `Alt + Enter` | Run cell and insert new cell below |
| `Ctrl + /` | Comment or uncomment selected line |
| `Ctrl + Shift + -` | Split current cell |
| `A` | Insert new cell above |
| `B` | Insert new cell below |
| `D D` | Delete selected cell |
| `M` | Change cell type to Markdown |
| `Y` | Change cell type to Code |

---



# 2. GIT 
## What is Git?
Git is a distributed version control system used to track changes in files over time.
#### Git vs GitHub:
- Git → local version control system;
- GitHub → cloud hosting platform for Git repositories.

#### Git Setup 
Git was installed in the system (not inside Conda environment).
#### Configuration:
git config --global user.name "YourName"
git config --global user.email "your@email.com"

## Git Workflow Attempt 
Commands used:
```bash
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:username/repo.git
git push -u origin main
```

## `.gitignore`

A `.gitignore` file is used in Git to specify which files and folders should not be tracked or committed to the repository.



Example:
```gitignore
# Ignore Python cache
__pycache__/
*.pyc

# Ignore logs
*.log

# Ignore environment variables
.env

# Ignore build folders
build/
dist/
```

---



## Key Learnings 
#### Git:

Git takes snapshots of project history; at least one commit is required before pushing; branch must exist before pushing. 

Never initialize Git in `~` (home directory). Always create a dedicated project folder first. 
#### Jupyter:
supports server running independently of browser; kernel executes code inside Conda environment; browser is only UI.
#### Conda:
environment isolation prevents dependency conflicts; always verify installations using `conda list` and other commands.


# 3. GIT & VS CODE

#### Initializing a Git Repository

To start version control for the project:


1. Opened the terminal inside VS Code.
2. Initialized a Git repository using `git init`
3. Added all project files using `git add .`
4. Created the first commit using:

```bash
git commit -m "Initial commit"
```

---

## Using Git Inside VS Code

VS Code provides built-in Git integration through the **Source Control** panel.

#### Steps

1. Click the **Source Control** icon (branch icon) from the left sidebar.
2. Enter a commit message.
3. Click the **✓ checkmark** to commit changes.

---

## Connecting Project to GitHub

#### Steps Followed

1. Opened **Source Control** in VS Code.
2. Clicked **Publish to GitHub**.
3. Completed authentication prompts.
4. Selected repository visibility:

   * Public
   * Private

This allows direct synchronization between the local project and GitHub.

---

## Useful VS Code Productivity Shortcuts

#### Multi-Cursor Editing | Purpose 
| Shortcut           | Purpose                                 |
| ------------------ | --------------------------------------- |
| `Ctrl + D`         | Select next occurrence of selected word |
| `Ctrl + Shift + L` | Select all occurrences
            
These shortcuts help edit multiple similar texts simultaneously.

---

## Code Navigation Shortcuts



#### Ctrl + G
Used to jump directly to a specific line number inside the current file. 
#### Ctrl + F
Used to search for text, variables, or functions within the currently open file.

#### Ctrl + Shift + F
Used to search across the entire project directory. 



---

## Running Shell Commands in VS Code

VS Code provides an integrated terminal that allows shell commands to be executed directly inside the editor environment.

#### Opening the Integrated Terminal

```bash
Ctrl + `
```

#### Example Commands Used

```bash
dd ~/ml-heart-disease-project 
code .
```

### Purpose of Commands

The `cd ~/ml-heart-disease-project` command navigates to the project directory.
The `code .` command opens the current folder directly inside VS Code.


---

## Extension Configuration in VS Code
Installed extensions work automatically using default configurations, but they can also be customized according to development needs.
#### Configuration Process
1. Opened VS Code settings using:
```bash 
Ctrl + , 
```
2. Searched for the required extension name.
3. Modified extension-specific settings as needed.
This allows customization of themes, formatting behavior, notebook support, linting tools, and other productivity features.
---
## Understanding Git File Status Icons
Git displays status indicators beside files inside the VS Code Explorer and Source Control panel.
### U — Untracked
Represents a newly created file that Git has not started tracking yet.
### M — Modified
Indicates that an already tracked file has been changed since the last commit.
### A — Added or Staged
Indicates that a file has been staged and is ready to be committed.
'these indicators help quickly identify the current state of files during software development workflows.

---
# Understanding Revert vs Reset in Git
Two important Git recovery operations were studied: `revert` and `reset`.
## Git Revert 
The `revert` operation creates a new commit that safely undoes changes introduced by a previous commit.### Characteristics* Preserves commit history.* Safer for collaborative projects.* Recommended after code has already been pushed to GitHub.This method is useful when mistakes must be corrected without rewriting project history.---
## Git ResetThe 
`reset` operation moves Git history backwards locally by changing the repository pointer to an earlier commit.### Characteristics* Can remove commits from visible history.* More powerful but potentially destructive.* Best used before sharing code with others.This method is commonly used for local cleanup and experimentation.-
kKey Learning OutcomeA major learning outcome from this workflow was understanding the trade-off between safety and flexibility in Git operations.* `Revert` prioritizes safety and collaboration.* `Reset` prioritizes flexibility and local history modification.Understanding when to use each operation is essential for effective version control management and collaborative software development.

---
---

# LEARNING SESSION 03
---
# NUMPY, PANDA, DATASET STRUCTURE
---

---

# 1. NUMPY

## What is NumPy?

NumPy stands for Numerical Python.

It is mainly used for:

- Fast numerical operations
- Arrays and matrices
- Scientific computing
- Mathematical operations on datasets

Machine learning models internally work with vectors, matrices, and tensors, so NumPy is neccessary.

---

#### Importing NumPy

```python
import numpy as np
```


#### NumPy Arrays (Comparison with python list):

##### Python List

```python
a = [1, 2, 3]
```

##### NumPy Array

```python
a = np.array([1, 2, 3])
```

#### Advantages of NumPy Arrays
- Faster computation compared to Python lists
- Lower memory usage
- Supports vectorized operations
- Optimized for mathematical processing

---

### Two-Dimensional Arrays

```python
arr2 = np.array([
    [1, 2],
    [3, 4]
])
```

Two-dimensional arrays represent matrix-like data structures commonly used in machine learning datasets.

---

### Array Shape

```python
print(arr2.shape)
```

Output:

```python
(2, 2)
```

Interpretation:
- 2 rows
- 2 columns

The shape attribute is critical in machine learning because models require input data in specific dimensions.

---

### Array Indexing

##### One-Dimensional Indexing

```python
print(arr[0])
```

##### Two-Dimensional Indexing

```python
print(arr2[1][0])
```

Used to access individual elements from arrays.

---

### Mathematical Operations

```python
a = np.array([1, 2, 3])

print(a + 5)
print(a * 2)
```

NumPy performs operations element-wise, enabling efficient vectorized computation.

---

### Common NumPy Functions

##### zeros()

Creates an array initialized with zeros.

```python
np.zeros((2, 3))
```


##### ones()

Creates an array initialized with ones.

```python
np.ones((2, 2))
```


##### arange()

Generates a sequence of values.

```python
np.arange(0, 10)
```


##### mean()

Calculates the arithmetic mean of array elements.

```python
np.mean(a)
```

---

# 2. PANDAS

### Introduction

Pandas is a data analysis library used for:
- Reading datasets
- Cleaning and preprocessing data
- Filtering and transforming data
- Statistical analysis
- Tabular data manipulation

The primary data structure in pandas is the DataFrame.


#### Importing Pandas

```python
import pandas as pd
```


#### Reading CSV Files

```python
df = pd.read_csv("data.csv")
```

Loads CSV-formatted datasets into a DataFrame structure.

---

### Dataset Inspection

#### Viewing Initial Rows

```python
print(df.head())
```

Displays the first few rows of the dataset for quick inspection.


#### Dataset Information

```python
print(df.info())
```

Provides:
- Column names
- Data types
- Non-null counts
- Missing value overview


#### Statistical Summary

```python
print(df.describe())
```

Generates descriptive statistics for numerical columns, including:
- Mean
- Minimum
- Maximum
- Standard deviation


#### Column Selection

##### Single Column

```python
df["Age"]
```

##### Multiple Columns

```python
df[["Name", "Age"]]
```

Used to retrieve specific features from the dataset.

#### Data Filtering

```python
young = df[df["Age"] < 30]
```

Filters rows based on conditional expressions.


#### Multiple Conditional Filtering

```python
df[(df["Age"] > 20) & (df["Cholesterol"] > 200)]
```

Notes:
- `&` represents logical AND
- Parentheses are required around conditions


#### GroupBy Operations

```python
df.groupby("Sex")["Age"].mean()
```

Groups records by a categorical feature and computes aggregate statistics.

Applications:
- Exploratory Data Analysis (EDA)
- Feature analysis
- Statistical summarization

---

# 3. DATASET STRUCTURE


### Dataset Dimensions

```python
print(df.shape)
```

Example Output:

```python
(303, 14)
```

Interpretation:
- 303 samples (rows)
- 14 features (columns)

---


### Features and Target Variables

##### Features
Input variables used for prediction.

##### Target Variable
Output variable the model aims to predict.

Example:

| age | chol | target |
|---|---|---|
| 45 | 230 | 1 |

Features:
- age
- chol

Target:
- target

---

### Data Types

```python
print(df.dtypes)
```

##### Numerical Data
Examples:
- age
- salary
- temperature

##### Categorical Data
Examples:
- Male/Female
- Yes/No
- City names

Categorical variables often require encoding before machine learning.

---

### Missing Value Analysis

```python
print(df.isnull().sum())
```

Used to identify incomplete or missing data within columns.

---

### Basic Data Analysis Workflow

#### Step 1 — Load Dataset

```python
df = pd.read_csv("data.csv")
```



#### Step 2 — Inspect Dataset

```python
print(df.head())
print(df.info())
print(df.shape)
```


#### Step 3 — Understand Features

Key questions:
- What does each column represent?
- Which column is the target variable?


#### Step 4 — Data Cleaning

Common preprocessing tasks:
- Handling missing values
- Removing duplicate records
- Correcting data types

#### Step 5 — Exploratory Data Analysis

Performed using:
- Filtering
- GroupBy operations
- Statistical summaries

---
---

# LEARNING SESSION 04
---
# BOXPLOT, CORELATION HEATMAP, CLASS IMBALANCE
---
---



## Exploratory Data Analysis (EDA) Concepts Learned

### Overview

In this phase of the machine learning project, I learned several important Exploratory Data Analysis (EDA) techniques used before model training. These techniques help understand the dataset structure, detect problems, and identify relationships between features.

The main topics learned were:

- Boxplots
- Correlation Heatmaps
- Class Imbalance

---

## 1. Boxplots

#### What is a Boxplot?

A boxplot is a statistical visualization used to understand the distribution of numerical data.

It helps identify:

- Median
- Spread of data
- Quartiles
- Outliers
- Skewness

---

#### Why Boxplots are Important

Boxplots are mainly used to detect outliers.

Outliers are abnormal values that are far away from the rest of the data.

Example:

```python
marks = [40, 42, 43, 45, 47, 48, 50, 95]
```

Here, `95` behaves like an outlier because it is much larger than the other values.

---

### Effect of Outliers in Machine Learning

Outliers can:

- Distort averages
- Reduce model accuracy
- Affect scaling
- Mislead algorithms

Algorithms highly sensitive to outliers include:

- Linear Regression
- KNN
- Clustering algorithms

---

### Boxplot Visualization Example

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

sns.boxplot(x=df["chol"])
plt.show()
```

This visualizes the cholesterol distribution and helps detect abnormal values.

---

# 2. CORELATION HEATMAP

## What is Correlation?

Correlation measures the relationship between two numerical variables.

Correlation values range from:

\[
-1 \rightarrow 1
\]

- `+1` = strong positive relationship
- `-1` = strong negative relationship
- `0` = no relationship

---

#### Example of Positive Correlation

| Age | Blood Pressure |
|------|----------------|
| 20 | 110 |
| 40 | 130 |
| 60 | 150 |

As age increases, blood pressure also increases.

This indicates positive correlation.

---

## What is a Correlation Heatmap?

A correlation heatmap is a visual representation of correlations between numerical features.

It uses colors to show relationship strength.

---

## Why Heatmaps are Important

Heatmaps help identify:

#### i. Important Features

Features strongly related to the target variable.

#### ii. Redundant Features

If two features are extremely correlated:

```python
corr = 0.99
```

then one feature may be unnecessary.

Removing redundant features can:

- Reduce multicollinearity
- Improve model performance
- Reduce overfitting
- Simplify computation

---

#### Correlation Heatmap Example

```python
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()
```

---

# 3. CLASS IMBALANCE


Class imbalance occurs when one class appears much more frequently than another class in a classification dataset.

Example:

| Class | Count |
|------|------|
| Healthy | 950 |
| Diseased | 50 |

This dataset is highly imbalanced.

---

## Why Class Imbalance is Dangerous

A model may predict every patient as healthy and still achieve:

\[
\frac{950}{1000} = 95\%
\]

accuracy.

However, the model completely fails to detect diseased patients.

Therefore, accuracy alone becomes misleading.

---

## Better Evaluation Metrics

For imbalanced datasets, important metrics include:

- Precision
- Recall
- F1-score
- Confusion Matrix

---

## How to Check Class Imbalance

##### Using Value Counts

```python
df["target"].value_counts()
```

##### Using Countplot

```python
sns.countplot(x=df["target"])
plt.show()
```

---

## Solutions to Class Imbalance

#### 1. Oversampling

Increase minority class samples.

Example methods:

- Random duplication
- SMOTE



#### 2. Undersampling

Reduce majority class samples.



#### 3. Class Weights

Assign higher importance to minority class errors.

Example:

```python
class_weight="balanced"
```
---
---
#   LEARNING SESSION 05
---
# DATASET ANALYSIS
---
---
## EDA Workflow Learned

Typical EDA workflow before machine learning training:

```python
df.head()
df.info()
df.describe()
```

Then:

1. Check missing values
2. Detect outliers using boxplots
3. Analyze feature relationships using heatmaps
4. Check class balance
5. Perform preprocessing
6. Train machine learning models

## Precision

Precision measures how accurate the model’s positive predictions are. It evaluates how many patients predicted as having heart disease actually have the disease. A high precision value indicates that the model produces fewer false positive predictions.

- **TP (True Positive):** Model correctly predicts heart disease.
- **FP (False Positive):** Model predicts heart disease when the patient is actually healthy.

#### Formula

\[
Precision = \frac{TP}{TP + FP}
\]

Precision is important in medical diagnosis because it reduces unnecessary treatments, tests, and patient anxiety caused by incorrect positive predictions.

---

## Recall

Recall measures the model’s ability to correctly identify patients who truly have heart disease. It focuses on minimizing false negatives, which is critical in healthcare applications.

- **TP (True Positive):** Model correctly predicts heart disease.
- **FN (False Negative):** Model predicts a patient as healthy even though the patient has heart disease.

#### Formula

\[
Recall = \frac{TP}{TP + FN}
\]

A high recall score means the model successfully detects most disease cases and minimizes missed diagnoses.

---

## F1 Score

F1 Score is a combined evaluation metric that balances both Precision and Recall. It is especially useful when both false positives and false negatives are important.

#### Formula

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

#### Example

Suppose the model achieves:

- Precision = 80%
- Recall = 90%

Then:

\[
F1 = 2 \times \frac{0.8 \times 0.9}{0.8 + 0.9} \approx 0.85
\]

This indicates that the model maintains a good balance between identifying disease cases and avoiding incorrect predictions.

---

## ROC Curve

ROC (Receiver Operating Characteristic) Curve is a graphical evaluation method used to measure the classification performance of a machine learning model across different threshold values.

The ROC Curve compares:

- **True Positive Rate (TPR)** → Ability to correctly detect positive cases.
- **False Positive Rate (FPR)** → Frequency of incorrectly predicting healthy patients as diseased.

#### TPR Formula

\[
TPR = \frac{TP}{TP + FN}
\]

#### FPR Formula

\[
FPR = \frac{FP}{FP + TN}
\]

Where:

- **TN (True Negative):** Model correctly predicts healthy patients.

A better ROC Curve stays closer to the upper-left corner, indicating stronger classification performance.

---

## AUC (Area Under the Curve)

AUC represents the area under the ROC Curve and measures the model’s overall ability to distinguish between positive and negative classes.

### Interpretation of AUC Values

| AUC Score | Performance |
|---|---|
| 0.90 – 1.00 | Excellent |
| 0.80 – 0.90 | Very Good |
| 0.70 – 0.80 | Good |
| 0.50 | Random Prediction |

A higher AUC value indicates better discrimination capability between heart disease and non-heart disease patients.

---

## Cross Validation

Cross Validation is a model evaluation technique used to test the stability and reliability of a machine learning model. Instead of using a single train-test split, the dataset is divided into multiple subsets called folds.

In this project, K-Fold Cross Validation was used to evaluate model consistency.

---

### K-Fold Cross Validation

In K-Fold Cross Validation:

1. The dataset is divided into **K equal folds**.
2. The model is trained using **K-1 folds**.
3. The remaining fold is used for testing.
4. This process repeats until every fold has been used once as the testing set.
5. The final score is calculated as the average of all iterations.

#### Example of 5-Fold Cross Validation

| Iteration | Training Folds | Testing Fold |
|---|---|---|
| 1 | Fold 2,3,4,5 | Fold 1 |
| 2 | Fold 1,3,4,5 | Fold 2 |
| 3 | Fold 1,2,4,5 | Fold 3 |
| 4 | Fold 1,2,3,5 | Fold 4 |
| 5 | Fold 1,2,3,4 | Fold 5 |

#### Formula

\[
Cross\ Validation\ Score = \frac{Score_1 + Score_2 + ... + Score_k}{k}
\]

Cross Validation helps reduce overfitting and provides a more reliable estimate of real-world model performance. In the implementation code, cross-validation was applied to evaluate the stability and generalization capability of the trained machine learning models.

---
---

# LEARNING SESSION 06
---
# PREPROCESSING, FEATURE ENGINEERING & DATA LEAKAGE
---
---

## Why Preprocessing Comes After EDA

EDA tells us *what is wrong* with the data (missing values, outliers, imbalance). Preprocessing fixes those issues in a repeatable way before any model sees the full dataset.

In the heart disease project, the order was:

1. Explore raw CSV (`heart_disease_uci.csv`)
2. Split train / test (80/20, stratified)
3. Fit preprocessing only on training data
4. Transform train and test with that same fitted pipeline
5. Train models on processed training features

##### Key idea

If we impute, encode, or scale using the whole dataset before splitting, information from the test set leaks into training. The model looks better on paper but fails in production.

---

## Train-Test Split (Stratified)

```python
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

| Parameter | Meaning in this project |
|-----------|-------------------------|
| `test_size=0.2` | 736 train / 184 test rows |
| `stratify=y` | Keeps class proportions similar in train and test |
| `random_state=42` | Same split every run (reproducibility) |

Stratification matters because class `4` (severe disease) has only 28 rows in the full dataset. A random split without stratify could put almost all severe cases in one fold by accident.

---

## Feature Types in the Dataset

| Type | Count | Examples |
|------|-------|----------|
| Numeric | 6 | age, trestbps, chol, thalch, oldpeak, ca |
| Categorical | 8 | sex, cp, thal, slope, exang, restecg, fbs, dataset |
| Dropped | 2 | id (identifier), num (target) |

Raw features: **14** → after one-hot encoding: **29** processed columns.

---

## Preprocessing Pipeline Steps

### 1. Numeric path

```
SimpleImputer(strategy="median") → StandardScaler()
```

- **Median imputation:** Robust when outliers exist (e.g. cholesterol spikes).
- **StandardScaler:** Converts each numeric column to mean ≈ 0, std ≈ 1 so distance-based and linear models are not dominated by large-scale features like `chol`.

### 2. Categorical path

```
SimpleImputer(strategy="most_frequent") → OneHotEncoder(handle_unknown="ignore")
```

- **Mode imputation:** Fills missing categories with the most common value seen in training.
- **One-hot encoding:** Each category becomes its own binary column (e.g. `cp_typical angina`, `cp_asymptomatic`).

### 3. ColumnTransformer

Both paths are combined with `ColumnTransformer` so numeric and categorical columns are processed in parallel, then concatenated into one matrix for sklearn models.

---

## Data Leakage Checklist (What I Learned to Avoid)

| Mistake | Why it is leakage |
|---------|-------------------|
| Fit imputer on full `df` before split | Test missing patterns influence train imputation |
| Fit scaler on train + test together | Test value ranges leak into scaling parameters |
| Tune model on test set | Test set is no longer “unseen” |
| Use target distribution from test to choose metrics | Should be decided from train/validation only |

Correct pattern used in `scripts/utils.py`:

```python
preprocessor.fit(X_train_raw)      # learn only from train
X_train = preprocessor.transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)  # apply train rules
```

---

## Saving the Fitted Preprocessor

The preprocessor is saved with `joblib` so inference uses the same imputation values, category mappings, and scaling as training:

```python
joblib.dump(preprocessor, "outputs/models/preprocessor.pkl")
```

This is bundled with the final model so a new patient row (raw features only) can be scored without re-running the notebook.

---
---

# LEARNING SESSION 07
---
# MACHINE LEARNING MODEL TYPES (CLASSIFICATION)
---
---

## High-Level Taxonomy

Supervised learning for classification means: learn a mapping from features **X** to label **y**.

```
                    Supervised Classification
                              |
        +---------------------+---------------------+
        |                     |                     |
   Parametric            Non-parametric         Ensemble
   (fixed form)          (flexible memory)      (many models)
        |                     |                     |
   Linear / LDA          KNN, trees            Forests, boosting
   Logistic, SVM         Naive Bayes           Bagging, stacking
```

##### Key idea

- **Parametric:** Assumes a functional form (e.g. linear boundary). Few parameters, fast, interpretable.
- **Non-parametric:** Complexity grows with data (e.g. KNN stores all training points).
- **Ensemble:** Combines many weak learners into one stronger predictor.

---

## Model Families Used in This Project (22 Models)

| Family | Models in project | Base idea |
|--------|-------------------|-----------|
| Baseline | DummyClassifier | Predict most frequent class — sanity check |
| Linear | Logistic Regression, Ridge, SGD, Perceptron | Linear decision boundary (or hyperplane) |
| Discriminant | LDA, Nearest Centroid | Assume class-specific distributions or centroids |
| Distance | KNN | Class = majority vote of nearest training points |
| Probabilistic | Gaussian Naive Bayes | Bayes rule + feature independence assumption |
| Margin-based | Linear SVM, RBF SVM | Maximize margin between classes |
| Tree | Decision Tree | Recursive splits on features |
| Bagging | Bagging (trees), Random Forest, Extra Trees | Many trees on bootstrap samples |
| Boosting | AdaBoost, Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost | Sequentially correct previous errors |
| Tuning | RandomizedSearchCV + RF | Random hyperparameter search with CV |

---

## 1. Baseline Models

### DummyClassifier

Always predicts the majority class (or stratified random guess).

**Purpose in learning flow:**

- If a complex model cannot beat the dummy, something is wrong (features, leakage, or labels).
- In our multiclass run, dummy accuracy ≈ 0.45 (matches largest class share).

---

## 2. Linear Models

### Logistic Regression

Models class probability with a sigmoid (binary) or softmax (multiclass) over a linear combination of features.

\[
P(y=1|x) = \sigma(w^T x + b)
\]

| Strength | Weakness |
|----------|----------|
| Fast, interpretable coefficients | Cannot capture non-linear interactions alone |
| Works well after scaling | Sensitive to correlated features |

**Heart disease context:** Good baseline when relationships are roughly monotonic (e.g. higher `oldpeak` ↔ more disease risk).

### Ridge Classifier

Linear classifier with L2 penalty on weights. Similar spirit to logistic but optimizes a different loss.

**Learning note:** In our evaluation, Ridge sometimes matched high accuracy but had no `predict_proba`, so ROC-AUC was missing for that row in the comparison table.

### SGDClassifier & Perceptron

- **SGD:** Stochastic gradient descent on linear models — scalable to large data.
- **Perceptron:** Early linear classifier; updates weights on misclassified points only.

Both teach that many “different” sklearn names are still linear decision boundaries under the hood.

---

## 3. Discriminant & Prototype Models

### Linear Discriminant Analysis (LDA)

Assumes each class has a Gaussian distribution with a shared covariance matrix. Finds linear separators that maximize class separation.

**When useful:** Moderate feature count, roughly normal numeric features after scaling.

### Nearest Centroid

Each class is represented by the mean feature vector (centroid). New samples go to the nearest centroid.

**Learning note:** Simple, fast, and surprisingly competitive on tabular medical data in our comparison.

---

## 4. Distance-Based: K-Nearest Neighbors (KNN)

No explicit training phase — the model stores all training points.

Prediction = majority class among the **k** closest points (Euclidean distance after scaling).

| Strength | Weakness |
|----------|----------|
| No training time | Slow prediction on large datasets |
| Flexible boundaries | Curse of dimensionality (29 features still OK here) |
| Intuitive | Sensitive to feature scale → why StandardScaler mattered |

**Parameter learned:** `n_neighbors=5` in our zoo.

---

## 5. Probabilistic: Gaussian Naive Bayes

Applies Bayes’ theorem:

\[
P(y|x) \propto P(y) \prod_i P(x_i|y)
\]

**“Naive”** = assumes features are independent given the class (often false, but works surprisingly well on tabular data).

| Strength | Weakness |
|----------|----------|
| Very fast train/predict | Independence assumption is strict |
| Good on binary tasks with mixed features | Poor if features are highly correlated |

**Project result:** Gaussian NB was best on binary target (F1 ≈ 0.87) — disease vs no-disease is a simpler boundary than 5 severity levels.

---

## 6. Support Vector Machines (SVM)

### LinearSVC

Finds the hyperplane with maximum margin between classes.

### SVC (RBF kernel)

Kernel trick maps features into higher dimensions so non-linear boundaries are possible.

| Kernel | Boundary shape |
|--------|----------------|
| Linear | Straight line / hyperplane |
| RBF | Smooth curved regions |

**Learning note:** SVMs need scaling; we used `probability=True` on RBF SVM to enable ROC-AUC via Platt-style probabilities.

---

## 7. Decision Trees

Splits data recursively on one feature at a time to minimize impurity (Gini or entropy).

```
                    [age <= 55?]
                   /            \
              Yes /              \ No
                /                \
        [chol <= 240?]      [cp = asymptomatic?]
              ...                  ...
```

| Strength | Weakness |
|----------|----------|
| No scaling required | Overfits easily |
| Easy to visualize | Unstable — small data change → different tree |

**Our settings:** `max_depth=10`, `min_samples_split=10` to limit overfitting.

---

## 8. Ensemble — Bagging

### BaggingClassifier

Trains many decision trees on bootstrap samples (random rows with replacement) and votes.

### RandomForestClassifier

Bagging + feature subsampling at each split → trees are less correlated → better generalization.

### ExtraTreesClassifier

Like Random Forest but splits are more random (extremely randomized trees) → often lower variance.

| Concept | Meaning |
|---------|---------|
| `n_estimators=200` | 200 trees vote |
| `max_depth=15` | Cap tree depth |
| `n_jobs=-1` | Use all CPU cores |

**Heart disease project:** Tree ensembles were strong on both targets; Random Forest + RandomizedSearch had among the highest ROC-AUC on binary.

---

## 9. Ensemble — Boosting

Boosting builds trees sequentially, each one focusing on mistakes of the previous ensemble.

| Algorithm | Core idea |
|-----------|-----------|
| AdaBoost | Up-weight misclassified samples |
| GradientBoosting | Fit new trees to pseudo-residuals (gradient of loss) |
| HistGradientBoosting | Histogram-based splits — fast on medium tabular data |
| XGBoost / LightGBM / CatBoost | Optimized gradient boosting implementations |

**Project result (multiclass):** **HistGradientBoosting** won best weighted F1 (0.6096) — severity prediction is harder than binary presence.

**Boosting vs bagging (mental model):**

- **Bagging:** Parallel trees, reduce variance.
- **Boosting:** Serial trees, reduce bias.

---

## 10. Hyperparameter Search

### RandomizedSearchCV

- Samples random combinations from a parameter grid.
- Uses K-fold CV on training data to score each combination.
- Picks the best estimator and refits it.

In this project, only Random Forest used this in the 22-model zoo (`n_iter=8`, `f1_weighted` scoring).

**Learning gap identified:** Tuning more models could improve multiclass severity class 4 recall.

---

## Choosing a Model Type 

| Situation | Start with | Why |
|-----------|------------|-----|
| Need interpretability | Logistic Regression, Decision Tree (shallow) | Coefficients or rules |
| Small/medium tabular data | Random Forest, HistGradientBoosting | Strong default |
| Binary yes/no, fast baseline | Gaussian NB, Logistic Regression | Simple, often strong |
| Many classes, imbalance | Boosting + weighted F1 | Handles hard multiclass better than plain accuracy |
| High-dimensional sparse text | Linear SVM, SGD | — (not this dataset) |
| Production latency critical | Naive Bayes, shallow tree, linear | Fast inference |

---
---

# LEARNING SESSION 08
---
# DUAL TARGET, CONFUSION MATRIX & MODEL COMPARISON
---
---

## Two Ways to Treat the Same `num` Column

The UCI target `num` originally has values 0–4.

| Formulation | Classes | Clinical meaning |
|-------------|---------|------------------|
| Multiclass | 5 | Severity: none → mild → severe |
| Binary | 2 | 0 = healthy, 1 = any disease (1+2+3+4 → 1) |

**Why both matter:**

- **Binary** answers: “Does this patient have heart disease?”
- **Multiclass** answers: “How severe is it?”

Same features and preprocessor; only **y** changes:

```python
# Multiclass: keep 0,1,2,3,4
y_multiclass = df["num"]

# Binary: collapse 1-4 into 1
y_binary = (df["num"] > 0).astype(int)
```

---

## Binary vs Multiclass — What Changed in Metrics

| Target | Best model (our run) | F1 | Insight |
|--------|----------------------|-----|---------|
| Multiclass 0–4 | HistGradientBoosting | 0.6096 | Hard task — rare class 4 |
| Binary 0/1 | Gaussian Naive Bayes | 0.8694 | Easier separation |

**Learning takeaway:** Good binary accuracy does not mean severity levels are equally predictable. Always state which target formulation was used.

---

## Confusion Matrix (Multiclass)

Rows = true class, columns = predicted class.

For 5 classes, diagonals are correct predictions; off-diagonals show *which* severities get confused (e.g. predicting `2` when truth is `3`).

**What I learned to read:**

- Model strong on class `0` and `1` (more training samples).
- Class `4` often misclassified — only 28 training examples.

---

## Weighted vs Macro F1

| Average | How it treats classes |
|---------|----------------------|
| **Macro** | Each class counts equally (punishes neglect of rare class 4) |
| **Weighted** | Weighted by support — closer to overall accuracy feel |

We used weighted F1 for model selection because it reflects overall performance while still penalizing some imbalance effects — documented in the submission report.

---

## ROC-AUC: Binary vs Multiclass

| Setting | sklearn approach |
|---------|------------------|
| Binary | AUC on positive class probability (`predict_proba[:, 1]`) |
| Multiclass | `multi_class="ovr"` (one-vs-rest), weighted average |

Some models (Ridge, LinearSVC, Perceptron) lack `predict_proba` → ROC-AUC stored as NaN — not a failure, a capability gap to note in reports.

---

## Model Comparison Workflow (End-to-End Learning Flow)

```
Day 1  → Understand rows, columns, missing values, target counts
Day 2  → EDA plots, correlations, imbalance (multiclass + binary)
Day 3  → Split → preprocess → save X_train, y_train, y_train_binary
Day 4  → Train 22 models (notebooks) or scripts/main.py (both targets)
Day 5  → Evaluate all models on same test set → comparison CSV
Day 6  → Pick best → save bundle (model + preprocessor + metadata)
```

**Fair comparison rules I applied:**

1. Same `random_state=42` split.
2. Same 29 processed features per target mode.
3. Same metrics functions in `evaluate_classifier()`.
4. Best model chosen only by held-out test weighted F1 (not train score).

---

## Results Snapshot 

### Multiclass — top 3 by F1

| Model | F1 | ROC-AUC |
|-------|-----|---------|
| HistGradientBoosting | 0.6096 | 0.8293 |
| Gradient Boosting | 0.5926 | 0.8322 |
| XGBoost | 0.5797 | 0.8445 |

### Binary — top 3 by F1

| Model | F1 | ROC-AUC |
|-------|-----|---------|
| Gaussian Naive Bayes | 0.8694 | 0.9174 |
| CatBoost | 0.8636 | 0.9179 |
| LightGBM | 0.8631 | 0.9140 |

The multiclass winner (HistGradientBoosting) was not the binary winner — different algorithms fit different problem shapes.

---

## Production Bundle (Final Model)

```python
bundle = {
    "model": trained_classifier,
    "preprocessor": fitted_column_transformer,
    "feature_names": [...],
    "target_mode": "multiclass" or "binary",
    "target_description": "...",
}
```

Inference path:

1. Raw patient row (14 features, no `id`/`num`)
2. `preprocessor.transform()`
3. `model.predict()` → class label

Two bundles in this project:

- `hist_gradient_boosting_final_bundle.pkl` — severity 0–4
- `gaussian_nb_final_bundle_binary.pkl` — disease yes/no


---
---

# LEARNING SESSION 09
---
# SKLEARN PIPELINE PATTERNS & REPRODUCIBILITY
---
---

## scripts/utils.py — Functions Learned 

| Function | Role in learning flow |
|----------|----------------------|
| `load_data()` | `na_values="?"` for UCI missing markers |
| `split_features_target()` | Drop `id`, separate `num` |
| `encode_target(mode=)` | Switch multiclass ↔ binary |
| `create_preprocessor()` | Leakage-safe ColumnTransformer |
| `prepare_train_test_data()` | Split + fit + transform in one call |
| `build_model_zoo()` | 22 classifiers + target-specific boosting params |
| `evaluate_classifier()` | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix |
| `get_best_model()` | Rank by weighted F1 |

---

## scripts/main.py 

```bash
python scripts/main.py
```

Runs `run_pipeline("multiclass")` then `run_pipeline("binary")`.


## Reproducibility Checklist

| Item | Value in project |
|------|------------------|
| Random seed | 42 |
| Environment | `environment.yml` / conda `ml_1` |
| Data file | `data/heart_disease_uci.csv` |
| Reports | JSON under `outputs/reports/` |
| Submission doc | `ASMICORE_SUBMISSION_REPORT.md` |

---

---

