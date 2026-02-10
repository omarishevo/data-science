import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import statsmodels.api as sm

# ---------------------------------
# App configuration
# ---------------------------------
st.set_page_config(
    page_title="Statsmodels Regression App",
    layout="centered"
)

st.title("📊 Regression Analysis with Statsmodels")
st.write("Linear regression using Statsmodels and interactive plots with Altair")

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
# Data preview & summary
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
split_idx = int((1 - test_size) * n)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# Add constant for intercept
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# ---------------------------------
# Statsmodels Linear Regression
# ---------------------------------
model = sm.OLS(y_train, X_train_sm)
results = model.fit()

y_pred = results.predict(X_test_sm)

# ---------------------------------
# Evaluation metrics
# ---------------------------------
mse = np.mean((y_test - y_pred) ** 2)
r2 = results.rsquared

st.subheader("Model Performance")
st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
st.metric("R² Score", f"{r2:.3f}")

# ---------------------------------
# Coefficients
# ---------------------------------
coef_df = pd.DataFrame({
    "Feature": ["Intercept"] + features,
    "Coefficient": results.params.values
})

st.subheader("Model Coefficients")
st.dataframe(coef_df)

# ---------------------------------
# Actual vs Predicted (Altair scatter)
# ---------------------------------
st.subheader("Actual vs Predicted")

comparison_df = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred.values
})

scatter_chart = alt.Chart(comparison_df).mark_circle(size=60).encode(
    x='Actual',
    y='Predicted',
    tooltip=['Actual', 'Predicted']
).interactive()

st.altair_chart(scatter_chart, use_container_width=True)

# ---------------------------------
# Residual analysis (Altair bar)
# ---------------------------------
st.subheader("Residual Analysis")

residuals = y_test.values - y_pred.values
residuals_df = pd.DataFrame({"Residuals": residuals})

residual_chart = alt.Chart(residuals_df).mark_bar().encode(
    x=alt.X('Residuals', bin=alt.Bin(maxbins=25)),
    y='count()'
)

st.altair_chart(residual_chart, use_container_width=True)

st.success("Regression analysis completed with Statsmodels and Altair!")
