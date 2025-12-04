import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

st.set_page_config(page_title="Train Model", layout="wide")
st.title("Train Machine Learning Model")

# Load preprocessed data
if "cleaned_df" not in st.session_state:
    st.warning("Please complete preprocessing first!")
    st.stop()

df = st.session_state["cleaned_df"]
st.write("### Cleaned Dataset")
st.dataframe(df.head())

# Target column selection
st.subheader("Select Target Variable (What you want to predict)")
target_column = st.selectbox("Choose the target column:", df.columns)

if not target_column:
    st.stop()

X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
test_size = st.slider("Test Size (%)", 10, 50, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

st.write("Training samples:", X_train.shape)
st.write("Testing samples:", X_test.shape)

# Model selection
st.subheader("Choose Machine Learning Model")

model_name = st.selectbox(
    "Select Model:",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

params = {}
st.write("---")

# Hyperparameters for each model
if model_name == "Linear Regression":
    st.info("Linear Regression has no major hyperparameters.")
    model = LinearRegression()

elif model_name == "Random Forest":
    st.subheader("Random Forest Hyperparameters")

    n_estimators = st.slider("Number of Trees", 50, 500, 150)
    max_depth = st.slider("Max Depth", 3, 30, 12)
    min_samples_split = st.slider("Min Samples Split", 2, 20, 5)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )

elif model_name == "XGBoost":
    st.subheader("XGBoost Hyperparameters")

    n_estimators = st.slider("Number of Trees", 50, 500, 200)
    learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
    max_depth = st.slider("Max Depth", 3, 20, 6)

    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42
    )

st.write("---")

# Train model button
if st.button("Train Model"):
    with st.spinner("Training the model..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Save values for Evaluation page
    st.session_state["model"] = model
    st.session_state["y_test"] = y_test
    st.session_state["y_pred"] = y_pred
    st.session_state["X_columns"] = list(X.columns)

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.success("Model trained successfully!")

    st.subheader("Model Performance Metrics")
    st.write(f"**RMSE:** {rmse:,.3f}")
    st.write(f"**MAE:** {mae:,.3f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # Save model file
    os.makedirs("../models", exist_ok=True)
    model_path = f"../models/{model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, model_path)

    st.success(f"Model saved at `{model_path}`")
    st.info("➡ Go to **Evaluation** page next!")
