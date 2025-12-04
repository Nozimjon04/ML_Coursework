# World Bank Contract Awards — Machine Learning Coursework

This project analyzes the **World Bank Contract Awards dataset** and builds machine learning models to **predict contract amounts**, supported by a complete **EDA→Preprocessing→Model Training→Evaluation→Deployment** workflow.

The solution includes:

- Jupyter Notebook for analysis  
- Data preprocessing pipeline  
- ML model training (3 models)  
- Model evaluation  
- Streamlit multi-page web application  
- Online deployment (Streamlit Cloud)  

---

#  Project Structure

ML_Coursework/
│
├── notebooks/
│ ├── data_exploration.ipynb
│ ├── data_preprocessing.ipynb
│ └── evaluation.ipynb
  └── model_training.ipynb
│
├── streamlit_app/
│ ├── app.py
│ └── pages/
│ ├── EDA.py
│ ├── Preprocessing.py
│ ├── Train_Model.py
│ └── Evaluation.py
│
├── data/
│ ├── raw/
│ └── processed/
│
├── models/ # ignored by Git
│
├── requirements.txt
├── .gitignore
└── README.md



---

#  Dataset

The dataset used:

**World Bank – Contract Awards in Investment Project Financing (Since FY 2020)**  
Source: https://financesone.worldbank.org/contract-awards-in-investment-project-financing-since-fy-2020/DS00005

The dataset contains fields such as:

- Region  
- Borrower Country  
- Procurement Category  
- Fiscal Year  
- Supplier  
- Contract Signing Date  
- Contract Amount (Target Variable)
---

# Exploratory Data Analysis (EDA)

Performed using:

- Summary statistics (mean, std, min, max, percentiles)
- Correlation matrix
- Histograms & KDE plots
- Boxplots for outlier inspection
- Scatter plots between numeric features
- Categorical frequency charts
- Missing values heatmap  
- Duplicate row detection

EDA is available in:

- Streamlit page: `/EDA`

---

#  Data Preprocessing

Steps included:

### Handling missing values
- Numeric → median replacement  
- Categorical → `"Unknown"`  

### Outlier removal  
- IQR (Interquartile Range) filtering on numeric features  

### Feature engineering  
- Extract year and month from `Contract Signing Date`  
- Create `Is_Recent` feature based on fiscal year  

### Encoding categorical features  
- One-hot encoding (low cardinality)
- Ordinal encoding (high cardinality)

### Scaling  
- StandardScaler applied to numeric columns

### Train-test split  
Handled in the training page (configurable split %).

File:  
- Streamlit: `/Preprocessing`

---
# Model Training

Three machine learning models:

1. **Linear Regression**  
2. **Random Forest Regressor**  
3. **XGBoost Regressor**

### Hyperparameter tuning (via Streamlit sliders):
- `n_estimators`, `max_depth`, `min_samples_split` (Random Forest)  
- `learning_rate`, `n_estimators`, `max_depth` (XGBoost)

### Target transformation
- Optional `log1p()` transformation for improved performance on skewed data.

### Outputs:
- RMSE  
- MAE  
- R² score  
- Saved trained model (`.joblib`)  
- Downloadable model file  

Files:
- `notebooks/Model_Training.ipynb`
- Streamlit: `/Train Model`

---

# Model Evaluation

Evaluation metrics:

- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- R² Score

Visualizations include:

- Actual vs Predicted scatter plot  
- Distribution of residuals  
- Feature importance for tree models  

File:
- Streamlit: `/Evaluation`

---

# Deployment (Streamlit Cloud)

The application is deployed at:

👉 **LIVE APP:** *https://mlcoursework-hycmb7wemglsee9hqeaj9h.streamlit.app/*  

Start the app locally:

```bash
streamlit run streamlit_app/app.py




