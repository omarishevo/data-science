import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ==============================
# CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Modular Data Analytics Platform",
    layout="wide"
)

# ==============================
# DATA LAYER
# ==============================
class DataLoader:

    @staticmethod
    def load_sample():
        np.random.seed(42)
        n = 400

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


# ==============================
# STATISTICS LAYER
# ==============================
class Statistics:

    @staticmethod
    def summary(df):
        return df.describe()

    @staticmethod
    def correlation(df):
        return df.corr()

    @staticmethod
    def one_sample_ttest(sample, pop_mean):
        n = len(sample)
        mean = np.mean(sample)
        std = np.std(sample, ddof=1)
        t_stat = (mean - pop_mean) / (std / np.sqrt(n))
        return t_stat


# ==============================
# MACHINE LEARNING LAYER
# ==============================
class LinearRegressionModel:

    def __init__(self):
        self.theta = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

    def evaluate(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        r2 = 1 - ss_res / ss_total
        return mse, r2


# ==============================
# VISUALIZATION LAYER
# ==============================
class Visualizer:

    @staticmethod
    def histogram(df, column):
        return alt.Chart(df).mark_bar().encode(
            x=alt.X(column, bin=True),
            y="count()"
        )

    @staticmethod
    def correlation_heatmap(corr_df):
        corr = corr_df.reset_index().melt(id_vars='index')
        corr.columns = ['Var1', 'Var2', 'Correlation']

        return alt.Chart(corr).mark_rect().encode(
            x='Var1:O',
            y='Var2:O',
            color='Correlation:Q'
        )

    @staticmethod
    def regression_plot(actual, predicted):
        df_plot = pd.DataFrame({
            "Actual": actual.flatten(),
            "Predicted": predicted.flatten()
        })

        return alt.Chart(df_plot).mark_circle(size=60).encode(
            x="Actual",
            y="Predicted"
        ).interactive()


# ==============================
# UI LAYER
# ==============================
def main():

    st.title("📊 Modular Data Analytics Platform")

    # Sidebar
    st.sidebar.header("Data Source")
    source = st.sidebar.radio("Select source", ["Sample Data", "Upload CSV"])

    if source == "Upload CSV":
        file = st.sidebar.file_uploader("Upload CSV", type="csv")
        if file:
            df = pd.read_csv(file)
        else:
            st.stop()
    else:
        df = DataLoader.load_sample()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Navigation
    section = st.sidebar.selectbox(
        "Analysis Section",
        ["Overview", "Exploration", "Statistics", "Regression"]
    )

    # ---------------- OVERVIEW ----------------
    if section == "Overview":
        st.header("Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing Values", df.isnull().sum().sum())

        st.dataframe(df.head(), use_container_width=True)
        st.dataframe(Statistics.summary(df), use_container_width=True)

    # ---------------- EXPLORATION ----------------
    elif section == "Exploration":
        st.header("Exploratory Analysis")

        if numeric_cols:
            column = st.selectbox("Select column", numeric_cols)
            chart = Visualizer.histogram(df, column)
            st.altair_chart(chart, use_container_width=True)

            if len(numeric_cols) > 1:
                st.subheader("Correlation Heatmap")
                corr = Statistics.correlation(df[numeric_cols])
                heatmap = Visualizer.correlation_heatmap(corr)
                st.altair_chart(heatmap, use_container_width=True)

    # ---------------- STATISTICS ----------------
    elif section == "Statistics":
        st.header("Statistical Testing")

        if numeric_cols:
            column = st.selectbox("Select column", numeric_cols)
            pop_mean = st.number_input("Population Mean", value=float(df[column].mean()))

            if st.button("Run One-Sample T-Test"):
                sample = df[column].dropna().values
                t_stat = Statistics.one_sample_ttest(sample, pop_mean)
                st.write(f"T-statistic: {t_stat:.4f}")
                st.write("P-value not computed (requires SciPy).")

    # ---------------- REGRESSION ----------------
    elif section == "Regression":

        if len(numeric_cols) >= 2:

            target = st.selectbox("Target Variable", numeric_cols)
            features = st.multiselect(
                "Feature Variables",
                [c for c in numeric_cols if c != target],
                default=[c for c in numeric_cols if c != target][:2]
            )

            if features and st.button("Train Linear Regression"):

                X = df[features].values
                y = df[target].values.reshape(-1, 1)

                model = LinearRegressionModel()
                model.fit(X, y)
                y_pred = model.predict(X)

                mse, r2 = model.evaluate(y, y_pred)

                col1, col2 = st.columns(2)
                col1.metric("MSE", f"{mse:.3f}")
                col2.metric("R²", f"{r2:.3f}")

                chart = Visualizer.regression_plot(y, y_pred)
                st.altair_chart(chart, use_container_width=True)

                coef_df = pd.DataFrame({
                    "Feature": ["Intercept"] + features,
                    "Coefficient": model.theta.flatten()
                })

                st.subheader("Model Coefficients")
                st.dataframe(coef_df, use_container_width=True)


if __name__ == "__main__":
    main()
