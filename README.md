# Semiconductor Manufacturing Quality Prediction

**Machine Learning Mini Project**  
University of Ruhuna

## 1. Project Overview

This project applies machine learning techniques to predict semiconductor manufacturing quality using the **UCI SECOM dataset**.

The objective is to classify manufacturing samples into two categories:

- **Pass (-1):** Semiconductor sample passes quality inspection.
- **Fail (+1):** Semiconductor sample fails quality inspection.

The dataset contains a significant class imbalance, making the identification of defective samples particularly challenging. Therefore, the project focuses on evaluating models using metrics that reflect their ability to detect the minority Fail class.

## 2. Dataset

Dataset: [SECOM — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/179/secom)

| Property | Description |
|---|---|
| Observations | 1,567 |
| Original features | 590 |
| Target classes | Pass (-1), Fail (+1) |
| Pass samples | 1,463 |
| Fail samples | 104 |

The dataset contains missing values and a highly imbalanced target distribution.

Place the downloaded dataset files inside the `data/` directory:

- `secom.data`
- `secom_labels.data`

## 3. Data Preprocessing

The following preprocessing operations are incorporated into the machine learning pipeline:

1. **Missing-value filtering:** Remove features with more than 50% missing values.
2. **Missing-value imputation:** Replace missing values using median imputation.
3. **Variance filtering:** Remove features with zero variance.
4. **Feature selection:** Apply ANOVA F-test feature selection, retaining 15 features in the final model pipeline.
5. **Feature scaling:** Apply StandardScaler.

Preprocessing is performed within the machine learning pipeline so that transformations are fitted using training data within each cross-validation fold, reducing data leakage.

## 4. Machine Learning Models

Five classification algorithms are investigated:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Support Vector Machine (SVM)
5. Gradient Boosting

Class imbalance is considered during model training where applicable.

## 5. Model Evaluation

The project uses the following evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Balanced Accuracy
- Average Precision (AP)

Because the Fail class is the minority class, accuracy alone is not sufficient to evaluate model performance.

### Cross-validation

Repeated Stratified 5-Fold Cross-Validation with 3 repetitions is used to compare the five classification algorithms on the training split.

### Nested Cross-Validation

Nested Stratified Cross-Validation is performed separately to evaluate Logistic Regression with hyperparameter selection.

- Outer cross-validation: 5 folds
- Inner cross-validation: 3 folds

### Hyperparameter Tuning

RandomizedSearchCV is used to tune Logistic Regression, SVM, and Gradient Boosting.

Average Precision is used as the optimization scoring metric.

### Threshold Optimization

Classification thresholds are investigated using out-of-fold predictions from the training data. The threshold is selected to maximize the Fail-class F1-score.

## 6. Results and Visualizations

The project generates visualizations for:

- Repeated cross-validation model comparison.
- Fail-class precision, recall, and F1-score.
- Tuned model evaluation.
- Default versus optimized classification thresholds.
- Nested cross-validation performance.

Generated plots and experiment results are stored in the `results/` directory.

**Important evaluation note:** The test split was used during initial model comparisons and subsequent evaluation. Therefore, its results should be interpreted as exploratory rather than as an independent final estimate of generalization performance.

## 7. Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Jupyter Notebook

## 8. Project Structure

```text
semiconductor-quality-prediction/
│
├── data/
│   └── (SECOM dataset files - not tracked by Git)
│
├── models/
│   ├── decision_tree.py
│   ├── evaluate_tuned_models.py
│   ├── gradient_boosting.py
│   ├── hyperparameter_tuning.py
│   ├── logistic_regression.py
│   ├── model_pipeline.py
│   ├── random_forest.py
│   ├── svm.py
│   ├── cross_validation.py
│   ├── threshold_optimization.py
│   ├── create_final_summary.py
│   └── visualize_final_results.py
│
├── preprocessing/
│   ├── preprocessing.py
│   └── preprocessing_analysis.ipynb
│
├── notebooks/
│   └── (Currently empty)
│
├── results/
│   ├── tuned_models/
│   ├── cross-validation results
│   ├── hyperparameter tuning results
│   ├── threshold optimization results
│   ├── nested cross-validation results
│   └── performance visualizations
│
├── cross_validation.py
├── explore_data.py
├── logistic_regression.py
├── nested_cv.py
├── visualize_results.py
├── requirements.txt
├── .gitignore
└── README.md
```
## 9. Installation and Execution

### Step 1: Clone the repository

```bash
git clone <repository-url>
cd semiconductor-quality-prediction
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 4: Run experiments

Run repeated cross-validation:

```bash
python -m models.cross_validation
```

Run nested cross-validation:

```bash
python nested_cv.py
```

Run hyperparameter tuning:

```bash
python -m models.hyperparameter_tuning
```

Evaluate tuned models:

```bash
python -m models.evaluate_tuned_models
```

Run threshold optimization:

```bash
python -m models.threshold_optimization
```

Generate final experiment summaries and visualizations:

```bash
python -m models.create_final_summary
python -m models.visualize_final_results
```

## 10. Group Information

**Group 29**  
University of Ruhuna  
Machine Learning Mini Project
