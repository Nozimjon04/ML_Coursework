import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

st.title("Data Preprocessing")

uploaded_file = st.file_uploader("Upload the RAW CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Raw dataset loaded!")

    st.subheader("Before Cleaning")
    st.write(df.head())

    st.subheader("Handling Missing Values")
    df = df.fillna("Unknown")
    st.write("Missing values have been replaced with 'Unknown'")

    st.subheader("Encoding Categorical Columns")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    encoder = LabelEncoder()

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col].astype(str))

    st.write("Categorical columns encoded successfully!")
    st.write(df.head())

    # Save cleaned data in session state so Train Model page can use it
    st.session_state["cleaned_df"] = df
    st.session_state["encoder"] = encoder

    st.success("Data preprocessing completed! You can now go to 'Train Model'")
