# Regression & Classification on Alzheimer’s Disease Data

A comprehensive data science assignment that builds and evaluates regression (MMSE score prediction), binary classification (Alzheimer’s diagnosis), and multiclass classification (cognitive impairment stages). The goal is to apply classical algorithms, tune hyperparameters, and interpret results – with a focus on avoiding data leakage and choosing appropriate metrics.

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
- **EducationLevel** (0,1,2,3) – multiclass (original, but weak)  
- **MMSE_Category** (0=Normal, 1=Mild, 2=Moderate, 3=Severe) – **new multiclass target** (more meaningful)

## What I Did

### 1. Preprocessing & Data Leakage Fix  
- Dropped non‑informative columns (`PatientID`, `DoctorInCharge`).  
- **Critical:** Ensured `MMSE` was removed from the feature matrix for both regression and the new multiclass task to avoid data leakage.  
- Standardised all numeric features (StandardScaler).  
- Same train/test split (80/20, `random_state=42`) used for all tasks.

### 2. Regression (MMSE)  
Models: Linear, Ridge, LASSO, Kernel Ridge.  
**Key findings:**  
- After removing `MMSE` from features, regression became meaningful.  
- **LASSO** performed best: R² = 0.891, MAE = 2.42 (errors within ~2.4 MMSE points).  
- Linear and Ridge were almost identical (R² ≈ 0.888).  
- Kernel Ridge failed (R² = -0.07) – default hyperparameters not suitable.  
- **Conclusion:** The remaining features (demographics, blood pressure, cholesterol, lifestyle) collectively explain ~89% of MMSE variance.

### 3. Binary Classification (Diagnosis)  
Models: Logistic Regression, Linear SVM, Kernel SVM (RBF), KNN (tuned k=15), Decision Tree (tuned depth=4), Random Forest.  
**Results:**  
- Accuracy: 79–89%  
- AUC: 0.86–0.94  
- Best: Random Forest (AUC = 0.937, F1 = 0.823)  
- Decision Tree (depth=4) gave highest accuracy (89.1%) with good F1 (0.849).  
- ROC curves and confusion matrix provided.  
- **No severe imbalance** – diagnosis classes are reasonably balanced.

### 4. Multiclass Classification (Original: EducationLevel)  
Models: SVM (OvR), Multinomial Logistic, KNN (k=20), Decision Tree (depth=1), XGBoost, LightGBM, AdaBoost.  
**Results:**  
- Accuracy: ~35–39% (barely above random 25%).  
- F1‑weighted as low as 0.21.  
- **Poor performance due to:**  
  - Weak relationship between health features and education level.  
  - Class imbalance (class 3 has only 41 test samples).  

### 5. Multiclass Classification (Improved: MMSE Categories)  

**Why a better target?**  
MMSE categories (Normal, Mild, Moderate, Severe) are clinically relevant and directly linked to the health features.  

**Data preparation fix:**  
- Created `MMSE_Category` from MMSE scores.  
- **Removed both `MMSE` and `MMSE_Category` from features** – no leakage.  

**Results:**  
- Best model (SVM OvR): accuracy = 36.5%, F1‑weighted = 0.313.  
- Performance is moderate because only non‑cognitive features (age, BP, cholesterol, lifestyle) remain.  
- **Confirms** that predicting exact cognitive stage from basic health metrics is difficult without direct cognitive assessments.  

**Metric choice:** F1‑Weighted accounts for class imbalance (normal and mild cases are underrepresented).

### 6. Error Analysis  
- Examined 71 misclassifications (Kernel SVM).  
- Most errors occur for borderline patients (early‑stage Alzheimer’s with MMSE near normal).  
- Feature importance (Random Forest) shows `FunctionalAssessment` and `ADL` are top predictors for Diagnosis – logical.

## Issues Encountered & Lessons Learned

| Problem | Why it happened | What I learned |
|---------|----------------|----------------|
| **Perfect R² (1.0) for regression** | `MMSE` was accidentally left in features (data leakage). | Always remove the target from features – even for regression. |
| **Perfect multiclass accuracy (1.0)** | `MMSE` was still in features when predicting `MMSE_Category`. | The same leakage can affect classification; clean features per task. |
| **Negative R² for MMSE after fix?** | Actually R² became good (0.89) after removing leakage. | Leakage can mask true relationships. |
| **Low multiclass accuracy (≤39%)** | Weak feature‑target relationship and class imbalance. | Choose a target that is naturally related to the features. |
| **Kernel Ridge performed worse than linear** | Default `gamma=0.1` was suboptimal; also kernel methods are not magic. | Always tune hyperparameters; don't assume non‑linear = better. |

## How to Run

1. Place `data.csv` in the same folder as the notebook.  
2. Run all cells in order (the notebook installs `lightgbm` automatically).  
3. Results tables and plots will be displayed inline.

## Submission Files

- `assignment2.ipynb` – main notebook with all code, outputs, and explanations  
- `README.md` – this file  
- `data.csv` – the dataset used
