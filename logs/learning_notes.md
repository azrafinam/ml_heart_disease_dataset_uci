
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

In this project, **K-Fold Cross Validation** was used to evaluate model consistency.

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