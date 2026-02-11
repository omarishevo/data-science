import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

st.set_page_config(page_title="Minimal Data Science Dashboard", layout="wide")

# -----------------------------
# Sample Data Generator
# -----------------------------
def load_sample_data(name):
    np.random.seed(42)

    if name == "Student Performance":
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

        return df

# -----------------------------
# Manual Linear Regression
# -----------------------------
def linear_regression(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta

# -----------------------------
# Manual t-test (one sample)
# -----------------------------
def one_sample_ttest(sample, pop_mean):
    n = len(sample)
    mean = np.mean(sample)
    std = np.std(sample, ddof=1)
    t_stat = (mean - pop_mean) / (std / np.sqrt(n))
    return t_stat

# -----------------------------
# App UI
# -----------------------------
st.title("📊 Minimal Data Science Dashboard")

st.sidebar.header("Data Source")

source = st.sidebar.radio("Choose data source:", ["Upload CSV", "Sample Data"])

df = None

if source == "Upload CSV":
    file = st.sidebar.file_uploader("Upload CSV", type="csv")
    if file:
        df = pd.read_csv(file)
else:
    df = load_sample_data("Student Performance")

if df is None:
    st.info("Upload data or select sample dataset.")
    st.stop()

# -----------------------------
# Overview
# -----------------------------
st.header("📋 Data Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.dataframe(df.head(), use_container_width=True)
st.dataframe(df.describe(), use_container_width=True)

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

# -----------------------------
# Exploratory Analysis
# -----------------------------
st.header("📈 Exploratory Analysis")

if len(numeric_cols) > 0:
    selected = st.selectbox("Select column", numeric_cols)

    hist = alt.Chart(df).mark_bar().encode(
        x=alt.X(selected, bin=True),
        y='count()'
    )

    st.altair_chart(hist, use_container_width=True)

# -----------------------------
# Correlation Heatmap
# -----------------------------
if len(numeric_cols) > 1:
    st.header("🔗 Correlation Matrix")

    corr = df[numeric_cols].corr().reset_index().melt(id_vars='index')
    corr.columns = ['Var1', 'Var2', 'Correlation']

    heatmap = alt.Chart(corr).mark_rect().encode(
        x='Var1:O',
        y='Var2:O',
        color='Correlation:Q'
    )

    st.altair_chart(heatmap, use_container_width=True)

# -----------------------------
# Statistical Test
# -----------------------------
st.header("🔍 One-Sample T-Test")

if len(numeric_cols) > 0:
    test_col = st.selectbox("Select column for test", numeric_cols)
    pop_mean = st.number_input("Population mean", value=float(df[test_col].mean()))

    if st.button("Run Test"):
        sample = df[test_col].dropna().values
        t_stat = one_sample_ttest(sample, pop_mean)

        st.write(f"T-statistic: {t_stat:.4f}")
        st.write("Note: p-value not computed (requires SciPy).")

# -----------------------------
# Machine Learning (Manual Regression)
# -----------------------------
st.header("🤖 Linear Regression (Manual)")

if len(numeric_cols) >= 2:
    target = st.selectbox("Target variable", numeric_cols)
    features = st.multiselect(
        "Feature variables",
        [c for c in numeric_cols if c != target],
        default=[c for c in numeric_cols if c != target][:2]
    )

    if len(features) > 0 and st.button("Train Model"):

        X = df[features].values
        y = df[target].values.reshape(-1, 1)

        theta = linear_regression(X, y)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y_pred = X_b @ theta

        mse = np.mean((y - y_pred) ** 2)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_res = np.sum((y - y_pred) ** 2)
        r2 = 1 - ss_res / ss_total

        col1, col2 = st.columns(2)
        col1.metric("MSE", f"{mse:.3f}")
        col2.metric("R²", f"{r2:.3f}")

        pred_df = pd.DataFrame({
            "Actual": y.flatten(),
            "Predicted": y_pred.flatten()
        })

        scatter = alt.Chart(pred_df).mark_circle(size=60).encode(
            x="Actual",
            y="Predicted"
        ).interactive()

        st.altair_chart(scatter, use_container_width=True)

        coef_df = pd.DataFrame({
            "Feature": ["Intercept"] + features,
            "Coefficient": theta.flatten()
        })

        st.subheader("Model Coefficients")
        st.dataframe(coef_df, use_container_width=True)
