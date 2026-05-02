# Healthcare Data Cleaning & Preprocessing

A practical data wrangling assignment that takes a raw, messy healthcare dataset and prepares it for machine learning. The final goal: predict a patient's medical condition (Asthma, Diabetes, Heart Disease, Hypertension) from age, gender, cholesterol, blood pressure, and visit history.

## Tools

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458?logo=pandas)
![NumPy](https://img.shields.io/badge/NumPy-Numerical_Computing-013243?logo=numpy)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Preprocessing-F7931E?logo=scikit-learn)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557c?logo=matplotlib)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical_Plots-388E8E?logo=seaborn)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75?logo=plotly)
![BeautifulSoup](https://img.shields.io/badge/BeautifulSoup-Web_Scraping-3776AB?logo=python)

## Dataset

A simulated healthcare CSV with 1000 rows and 10 columns. It contains real‑world problems:
- missing values
- inconsistent date formats
- text‑based age (e.g., "forty")
- categorical variables (Gender, Condition, Medication)

## What I Did

1. **Cleaning**  
   - Converted age words to numbers and imputed missing values with median.  
   - Parsed messy dates into `Days Since First Visit`.  
   - Split blood pressure into systolic and diastolic, imputed missing.  
   - Imputed `Condition` with mode and `Cholesterol` with median.  
   - Dropped identifiers (email, phone, patient name).  

2. **Encoding & Scaling**  
   - One‑hot encoded `Gender` and `Medication`.  
   - Standardised all numeric features (Z‑score).

3. **Visualisation**  
   - Static: pie, box, line, bar, scatter, bubble, error bar charts.  
   - Interactive: Plotly scatter plot with hover details.

4. **Feature Engineering**  
   - Created `Days Since First Visit`, `Systolic`, `Diastolic`, dummy variables.  
   - Used Mutual Information for feature selection.  
   - Applied PCA for dimensionality reduction (visualisation).

5. **Baseline Model**  
   - Logistic regression to predict `Condition`.  
   - Accuracy: 41% (random baseline is 25%).  
   - Confirms that cleaning preserved predictive signal.

6. **Bonus – Web Scraping**  
   - Scraped 50 “Samand” cars (post‑1385) from bama.ir.  
   - Extracted price, mileage, colour, year, transmission, description.  
   - Files: `samand_scraper.py` and `samand_cars.xlsx`.

## How to Run

1. Place `healthcare_messy_data.csv` in the same folder as the notebook.  
2. Run the notebook cells in order.  
3. Cleaned data will be saved as `healthcare_cleaned.csv`.  
4. For the scraper: run `samand_scraper.py` (internet required) → generates `samand_cars.xlsx`.

## Submission Files

- `assignment1.ipynb` – main notebook  
- `healthcare_cleaned.csv` – output after cleaning  
- `samand_scraper.py` + `samand_cars.xlsx` – bonus web scraping  
- `README.md` – this file  

---

*For details, see the comments and markdown cells inside the notebook.*
