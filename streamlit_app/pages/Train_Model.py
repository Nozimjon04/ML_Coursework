import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

st.title("Train Machine Learning Model")

# Check if data exists from preprocessing
if "cleaned_df" not in st.session_state:
    st.warning("Please complete preprocessing first!")
    st.stop()

df = st.session_state["cleaned_df"]

st.subheader("Cleaned Dataset Ready for Training")
st.write(df.head())

# Select Target Column 
st.subheader("Select Target Column (Prediction Column)")
target_column = st.selectbox("Choose the target variable:", df.columns)

if target_column:
    st.info(f"Your model will predict: **{target_column}**")

    # Prepare Data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train/Test Split
    test_size = st.slider("Test Size (%)", 10, 50, 20) / 100

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    st.write("Training samples:", X_train.shape)
    st.write("Testing samples:", X_test.shape)

    # Train button
    if st.button("Train Model"):
        with st.spinner("Training model... Please wait"):
            rf = RandomForestRegressor(
                n_estimators=80,
                max_depth=15,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )

            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

        # Save predictions for evaluation page
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["model"] = rf

        # Compute metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.success("Model trained successfully!")

        st.subheader("Model Performance Metrics")
        st.write(f"**RMSE:** {rmse:,.2f}")
        st.write(f"**MAE:** {mae:,.2f}")
        st.write(f"**R² Score:** {r2:.4f}")

        # --- Save Model ---
        model_path = "../models/best_model.pkl"

        os.makedirs("../models", exist_ok=True)
        joblib.dump(rf, model_path)

        st.success(f"Model saved to `{model_path}`")

        st.info("➡ Now go to **Evaluation** page to visualize results!")
