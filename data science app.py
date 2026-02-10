import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# Data generation (controlled + realistic)
# ---------------------------------
np.random.seed(42)
n = 300

df = pd.DataFrame({
    "Study_Hours": np.random.normal(6, 1.5, n),
    "Attendance": np.random.uniform(50, 100, n),
    "Sleep_Hours": np.random.normal(7, 1, n),
    "Previous_Score": np.random.normal(65, 10, n)
})

# Target variable with noise
df["Final_Score"] = (
    5 * df["Study_Hours"]
    + 0.3 * df["Attendance"]
    + 2 * df["Sleep_Hours"]
    + 0.5 * df["Previous_Score"]
    + np.random.normal(0, 5, n)
)

# ---------------------------------
# Preview data
# ---------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

st.subheader("Dataset Summary")
st.dataframe(df.describe())

# ---------------------------------
# Feature selection
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
st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
st.write(f"**R² Score:** {r2:.3f}")

# ---------------------------------
# Coefficients interpretation
# ---------------------------------
coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

st.subheader("Model Coefficients")
st.dataframe(coef_df)

# ---------------------------------
# Visualization: Actual vs Predicted
# ---------------------------------
st.subheader("Actual vs Predicted")

fig1, ax1 = plt.subplots()
ax1.scatter(y_test, y_pred)
ax1.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
ax1.set_xlabel("Actual Final Score")
ax1.set_ylabel("Predicted Final Score")
ax1.set_title("Model Fit")

st.pyplot(fig1)

# ---------------------------------
# Residual analysis
# ---------------------------------
st.subheader("Residual Distribution")

residuals = y_test - y_pred

fig2, ax2 = plt.subplots()
ax2.hist(residuals, bins=25)
ax2.set_xlabel("Residual")
ax2.set_ylabel("Frequency")
ax2.set_title("Residuals")

st.pyplot(fig2)

st.success("Regression analysis completed successfully.")
