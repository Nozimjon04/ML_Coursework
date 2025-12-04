import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.title("Model Evaluation")

# Check if the model & predictions exist
if "model" not in st.session_state or "y_pred" not in st.session_state:
    st.warning("Please train a model first on the 'Train Model' page.")
    st.stop()

model = st.session_state["model"]
y_test = st.session_state["y_test"]
y_pred = st.session_state["y_pred"]

# Evaluation Metrics
st.subheader("Evaluation Metrics")

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"**RMSE:** {rmse:,.2f}")
st.write(f"**MAE:** {mae:,.2f}")
st.write(f"**R² Score:** {r2:.4f}")

# Scatter Plot: Actual vs Predicted
st.subheader("Actual vs Predicted Values")

fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax1)
ax1.set_xlabel("Actual Contract Amount")
ax1.set_ylabel("Predicted Contract Amount")
ax1.set_title("Actual vs Predicted Values")
st.pyplot(fig1)

