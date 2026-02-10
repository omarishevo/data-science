import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------
# App configuration
# ---------------------------------
st.set_page_config(
    page_title="Auto Data Science App",
    layout="centered"
)

st.title("📊 Data Science Regression App")
st.write("End-to-end regression analysis using an automatically generated dataset")

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
# Data preview & EDA
# ---------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Summary Statistics")
st.dataframe(df.describe())

# ---------------------------------
# Features & target
# ---------------------------------
features = ["Study_Hours", "Attendance", "Sleep_Hours", "Previous_Score"]
target = "Final_Score"

X = df[features]
y = df[target]

# ---------------------------------
# Train-test split
# ---------------------------------
test_size = st.slider("Test set size", 0.1, 0.5, 0.2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# ---------------------------------
# Model training
# ---------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------------------------
# Evaluation
# ---------------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("Model Performance")
st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
st.metric("R² Score", f"{r2:.3f}")

# ---------------------------------
# Coefficient interpretation
# ---------------------------------
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.subheader("Model Coefficients")
st.dataframe(coef_df)

# ---------------------------------
# Actual vs Predicted (Streamlit chart)
# ---------------------------------
st.subheader("Actual vs Predicted")

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

st.scatter_chart(comparison_df)

# ---------------------------------
# Residual analysis (Streamlit chart)
# ---------------------------------
st.subheader("Residual Analysis")

residuals_df = pd.DataFrame({
    "Residuals": y_test.values - y_pred
})

st.bar_chart(residuals_df["Residuals"].value_counts().sort_index())

st.success("Regression analysis completed successfully.")
