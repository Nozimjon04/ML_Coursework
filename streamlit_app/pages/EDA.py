import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Exploratory Data Analysis")

uploaded_file = st.file_uploader("Upload contract awards CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    st.subheader("Dataset Preview")
    st.write(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Numeric vs Categorical")
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    st.write("Numeric Columns:", numeric_cols)
    st.write("Categorical Columns:", categorical_cols)

    st.subheader("Correlation Heatmap (Numeric Only)")
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")
