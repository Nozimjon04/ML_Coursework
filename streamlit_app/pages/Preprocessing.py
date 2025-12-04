import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Data Preprocessing", layout="wide")
st.title("Data Preprocessing")

uploaded_file = st.file_uploader("Upload the RAW CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.success("Raw dataset loaded!")

    st.subheader("Preview of Raw Data")
    st.write(df.head())

    # 1. HANDLE MISSING VALUES 
    st.subheader("Handling Missing Values")
    missing_before = df.isnull().sum()

    # Numerical → fill with median
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Categorical → fill with "Unknown"
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    missing_after = df.isnull().sum()

    st.write("Missing values fixed:")
    st.write(pd.DataFrame({
        "Before": missing_before,
        "After": missing_after
    }))

    # 2. CORRECTING ERROR DATA (Outlier Removal)
    st.subheader("Correcting Error Data (Outlier Removal)")

    # Remove extreme outliers using IQR method
    def remove_outliers(df, columns):
        df_clean = df.copy()
        for col in columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        return df_clean

    before_rows = len(df)
    df = remove_outliers(df, numeric_cols)
    after_rows = len(df)

    st.write(f"Outlier rows removed: **{before_rows - after_rows}**")

    # 3. FEATURE ENGINEERING 
    st.subheader("Feature Engineering")

    # Example engineered features (COURSEWORK REQUIREMENT)
    if "Contract Signing Date" in df.columns:
        df["Signing_Year"] = pd.to_datetime(df["Contract Signing Date"], errors='coerce').dt.year
        df["Signing_Month"] = pd.to_datetime(df["Contract Signing Date"], errors='coerce').dt.month
    else:
        df["Signing_Year"] = 0
        df["Signing_Month"] = 0

    if "Fiscal Year" in df.columns:
        df["Is_Recent"] = (df["Fiscal Year"] >= df["Fiscal Year"].median()).astype(int)
    else:
        df["Is_Recent"] = 0

    # Label Encoding categorical columns
    st.subheader("Encoding Categorical Columns")
    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    st.write("Categorical columns encoded.")
    st.write(df.head())

    # 4. SCALING (StandardScaler) 
    st.subheader("Scaling Numeric Features")

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    st.success("Numeric columns scaled using StandardScaler.")

    st.write(df.head())

    # SAVE PROCESSED DATA
    st.session_state["cleaned_df"] = df
    st.session_state["encoder"] = encoder
    st.session_state["scaler"] = scaler

    st.success("Data preprocessing completed successfully!")
    st.info("➡ Now go to **Train Model** to continue.")

else:
    st.info("Please upload a CSV file to begin preprocessing.")
