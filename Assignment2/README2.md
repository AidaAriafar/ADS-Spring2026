# Regression & Classification on Alzheimer’s Disease Data

A comprehensive data science assignment that builds and evaluates regression (MMSE score prediction), binary classification (Alzheimer’s diagnosis), and multiclass classification (education level) models. The goal is to apply classical algorithms, tune hyperparameters, and interpret results – even when performance is poor.

## Tools

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Models-F7931E?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-FF6600?logo=xgboost)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient_Boosting-00A1D6?logo=lightgbm)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-388E8E?logo=seaborn)

## Dataset

Synthetic Alzheimer’s dataset (`data.csv`) with 2149 patients and 35 features (demographics, clinical measurements, lifestyle factors, cognitive assessments). Targets:
- **MMSE** (Mini‑Mental State Exam, 0‑30) – regression  
- **Diagnosis** (0 = No Alzheimer’s, 1 = Alzheimer’s) – binary classification  
- **EducationLevel** (0,1,2,3) – multiclass classification  

## What I Did

### 1. Preprocessing  
- Dropped non‑informative columns (`PatientID`, `DoctorInCharge`).  
- Standardised all numeric features (StandardScaler).  
- Confirmed no missing values (dataset already clean).  
- Same train/test split (80/20) used for all tasks.

### 2. Regression (MMSE)  
Models: Linear, Ridge, LASSO, Kernel Ridge (plus later SVR, RandomForest, GradientBoosting).  
**Key findings:**  
- All models gave **negative R²** (≈ -0.02 to -0.55), meaning they perform worse than predicting the mean.  
- MAPE was inflated (>400%) due to MMSE values near zero – I switched to Median Absolute Error for robustness.  
- Conclusion: The given features have very weak linear or even non‑linear correlation with MMSE; regression is not feasible on this dataset.

### 3. Binary Classification (Diagnosis)  
Models: Logistic Regression, Linear SVM, Kernel SVM (RBF), KNN (tuned k=15), Decision Tree (tuned depth=4), Random Forest.  
**Results:**  
- Accuracy: 79–86%  
- AUC: 0.86–0.89  
- Best: Decision Tree (depth=4) with F1 = 0.817, AUC = 0.887  
- ROC curves and confusion matrix provided.  
- **No imbalance issue** – Diagnosis classes are reasonably balanced.

### 4. Multiclass Classification (EducationLevel)  
Models: SVM (OvR), Multinomial Logistic, KNN (k=20), Decision Tree (depth=1), XGBoost, LightGBM, AdaBoost.  
**Results:**  
- Accuracy: ~33–39% (only slightly better than random baseline ~25% for 4 classes).  
- F1‑macro as low as 0.14 – model struggles with minority classes (especially class 3 with only 41 test samples).  
- Log loss high (>1.2).  
- **Class imbalance** is the main issue; adding class weights or SMOTE would be needed for improvement.

### 5. Model Evaluation & Discussion  
- For regression: **MAE** is most interpretable (MMSE points); MAPE unreliable when true values are near zero.  
- For binary classification: **F1‑score** and **AUC** are best (imbalance not severe but still important).  
- For multiclass: **F1‑weighted** accounts for class support.  
- Provided thorough answers to all discussion questions (bias‑variance, L1 vs L2, kernel trick, regularising trees, micro/macro F1, multi‑label vs multiclass, etc.).

### 6. Error Analysis  
- Examined misclassified binary cases: most borderline patients have MMSE scores near the diagnostic threshold.  
- Feature importance (Random Forest) shows top predictors for Diagnosis: `FunctionalAssessment`, `ADL`, `MMSE` (logical).

## Issues Encountered & Lessons Learned

| Problem | Why it happened | What I learned |
|---------|----------------|----------------|
| **Negative R² for MMSE regression** | Features have almost no correlation with MMSE (non‑linear or no relationship). Even tree‑based models failed. | Regression requires features with genuine predictive power; not every dataset is suited for every task. |
| **Low multiclass accuracy (≤39%)** | Severe class imbalance (class 3 has 41 test samples vs 162 for class 1) and weak feature‑class relationships. | Always check class distribution; use F1‑weighted or macro, not accuracy. |
| **MAPE > 400%** | Division by very small true MMSE values (0–30). | Use alternative metrics like Median Absolute Error for bounded targets. |
| **Kernel Ridge performed worse than linear** | Default `gamma=0.1` was suboptimal; tuning `alpha` and `gamma` slightly improved it but still negative R². | Kernel methods are not magic – they need proper hyperparameter search. |

## How to Run

1. Place `data.csv` in the same folder as the notebook.  
2. Run all cells in order (the notebook installs `lightgbm` automatically).  
3. Results tables and plots will be displayed inline.

## Submission Files

- `assignment2.ipynb` – main notebook with all code, outputs, and explanations  
- `README2.md` – this file  
- `data.md` – the dataset I used
---

*Note: Poor regression results are **not** a bug in the code, they actually reflect the actual (lack of) predictive relationship in the dataset. The implementation, evaluation, and analysis have been implemented correctly*