import streamlit as st
import numpy as np
import pandas as pd

# ---------------------------------
# App configuration
# ---------------------------------
st.set_page_config(
    page_title="Pure NumPy Data Science App",
    layout="centered"
)

st.title("📊 Data Science Regression App")
st.write("Linear regression implemented from scratch using NumPy")

# ---------------------------------
# Data generation
# ---------------------------------
np.random.seed(42)
n = 300

df = pd.DataFrame({
    "Study_Hours": np.random.normal(6, 1.5, n),
    "Attendance": np.random.uniform(50, 100, n),
    "Sleep_Hours": np.random.normal(7, 1, n),
    "Previous_Score": np.random.normal(65, 10, n)
})

df["Final_Score"] = (
    5 * df["Study_Hours"]
    + 0.3 * df["Attendance"]
    + 2 * df["Sleep_Hours"]
    + 0.5 * df["Previous_Score"]
    + np.random.normal(0, 5, n)
)

# ---------------------------------
# EDA
# ---------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Summary Statistics")
st.dataframe(df.describe())

# ---------------------------------
# Features and target
# ---------------------------------
features = ["Study_Hours", "Attendance", "Sleep_Hours", "Previous_Score"]
target = "Final_Score"

X = df[features].values
y = df[target].values.reshape(-1, 1)

# ---------------------------------
# Train-test split (manual)
# ---------------------------------
test_size = st.slider("Test set size", 0.1, 0.5, 0.2)

split_idx = int((1 - test_size) * n)

X_train = X[:split_idx]
X_test = X[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

# ---------------------------------
# Add bias term
# ---------------------------------
X_train_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# ---------------------------------
# Linear Regression (Normal Equation)
# θ = (XᵀX)⁻¹ Xᵀy
# ---------------------------------
theta = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

y_pred = X_test_b @ theta

# ---------------------------------
# Evaluation metrics (manual)
# ---------------------------------
mse = np.mean((y_test - y_pred) ** 2)

ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_residual = np.sum((y_test - y_pred) ** 2)
r2 = 1 - (ss_residual / ss_total)

st.subheader("Model Performance")
st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
st.metric("R² Score", f"{r2:.3f}")

# ---------------------------------
# Coefficients
# ---------------------------------
coef_df = pd.DataFrame({
    "Feature": ["Intercept"] + features,
    "Coefficient": theta.flatten()
})

st.subheader("Model Coefficients")
st.dataframe(coef_df)

# ---------------------------------
# Actual vs Predicted (Streamlit chart)
# ---------------------------------
st.subheader("Actual vs Predicted")

comparison_df = pd.DataFrame({
    "Actual": y_test.flatten(),
    "Predicted": y_pred.flatten()
})

st.scatter_chart(comparison_df)

# ---------------------------------
# Residual analysis
# ---------------------------------
st.subheader("Residual Analysis")

residuals = y_test.flatten() - y_pred.flatten()
residuals_df = pd.DataFrame({"Residuals": residuals})

st.line_chart(residuals_df)

st.success("Pure NumPy regression analysis completed successfully.")
