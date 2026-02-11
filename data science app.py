"""
Interactive Data Science Dashboard
A comprehensive Streamlit app for data analysis and visualization
Author: AI Assistant
Version: 5.0 (Minimal - Only Streamlit, Pandas, NumPy)
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Science Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    h2 {
        color: #ff7f0e;
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Helper Functions
def load_sample_data(dataset_name):
    """Load sample datasets for demonstration"""
    if dataset_name == "Iris":
        # Generate Iris-like dataset
        np.random.seed(42)
        n_samples = 150
        
        # Generate features for three species
        setosa = pd.DataFrame({
            'sepal length (cm)': np.random.normal(5.0, 0.35, 50),
            'sepal width (cm)': np.random.normal(3.4, 0.38, 50),
            'petal length (cm)': np.random.normal(1.5, 0.17, 50),
            'petal width (cm)': np.random.normal(0.2, 0.10, 50),
            'species': 'setosa',
            'target': 0
        })
        
        versicolor = pd.DataFrame({
            'sepal length (cm)': np.random.normal(5.9, 0.52, 50),
            'sepal width (cm)': np.random.normal(2.8, 0.31, 50),
            'petal length (cm)': np.random.normal(4.3, 0.47, 50),
            'petal width (cm)': np.random.normal(1.3, 0.20, 50),
            'species': 'versicolor',
            'target': 1
        })
        
        virginica = pd.DataFrame({
            'sepal length (cm)': np.random.normal(6.6, 0.64, 50),
            'sepal width (cm)': np.random.normal(3.0, 0.32, 50),
            'petal length (cm)': np.random.normal(5.6, 0.55, 50),
            'petal width (cm)': np.random.normal(2.0, 0.27, 50),
            'species': 'virginica',
            'target': 2
        })
        
        df = pd.concat([setosa, versicolor, virginica], ignore_index=True)
        return df
    
    elif dataset_name == "Wine Quality":
        # Generate synthetic wine quality data
        np.random.seed(42)
        n_samples = 178
        
        df = pd.DataFrame({
            'alcohol': np.random.uniform(11, 15, n_samples),
            'malic_acid': np.random.uniform(0.7, 5.8, n_samples),
            'ash': np.random.uniform(1.4, 3.2, n_samples),
            'alcalinity_of_ash': np.random.uniform(10, 30, n_samples),
            'magnesium': np.random.uniform(70, 162, n_samples),
            'total_phenols': np.random.uniform(0.98, 3.88, n_samples),
            'flavanoids': np.random.uniform(0.34, 5.08, n_samples),
            'nonflavanoid_phenols': np.random.uniform(0.13, 0.66, n_samples),
            'proanthocyanins': np.random.uniform(0.41, 3.58, n_samples),
            'color_intensity': np.random.uniform(1.3, 13, n_samples),
            'hue': np.random.uniform(0.48, 1.71, n_samples),
            'od280/od315_of_diluted_wines': np.random.uniform(1.27, 4, n_samples),
            'proline': np.random.uniform(278, 1680, n_samples),
            'target': np.random.choice([0, 1, 2], n_samples)
        })
        return df
    
    elif dataset_name == "Boston Housing":
        # Generate synthetic housing data
        np.random.seed(42)
        n_samples = 506
        df = pd.DataFrame({
            'CRIM': np.random.exponential(3.6, n_samples),
            'RM': np.random.normal(6.3, 0.7, n_samples),
            'AGE': np.random.uniform(0, 100, n_samples),
            'DIS': np.random.exponential(3.8, n_samples),
            'LSTAT': np.random.exponential(12.6, n_samples),
            'PRICE': np.random.normal(22.5, 9, n_samples)
        })
        return df
    
    elif dataset_name == "Sales Data":
        # Generate synthetic sales data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Sales': np.random.poisson(100, 365) + np.sin(np.arange(365) * 2 * np.pi / 365) * 20,
            'Marketing_Spend': np.random.uniform(500, 2000, 365),
            'Temperature': np.random.normal(20, 10, 365),
            'Region': np.random.choice(['North', 'South', 'East', 'West'], 365)
        })
        return df

def generate_statistical_summary(df):
    """Generate comprehensive statistical summary"""
    summary = {
        'Shape': df.shape,
        'Columns': df.columns.tolist(),
        'Data Types': df.dtypes.to_dict(),
        'Missing Values': df.isnull().sum().to_dict(),
        'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    return summary

# Main App
def main():
    st.title("📊 Interactive Data Science Dashboard")
    st.markdown("**Explore, Analyze, and Visualize Your Data**")
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Data Source Selection
    st.sidebar.header("1️⃣ Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV", "Sample Datasets"]
    )
    
    df = None
    
    if data_source == "Upload CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=['csv']
        )
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    else:
        dataset_choice = st.sidebar.selectbox(
            "Select a sample dataset:",
            ["Iris", "Wine Quality", "Boston Housing", "Sales Data"]
        )
        df = load_sample_data(dataset_choice)
        st.sidebar.success(f"✅ Loaded {dataset_choice} dataset")
    
    if df is not None:
        # Analysis Mode Selection
        st.sidebar.header("2️⃣ Analysis Mode")
        analysis_mode = st.sidebar.selectbox(
            "Choose analysis type:",
            [
                "📋 Data Overview",
                "📈 Exploratory Analysis"
            ]
        )
        
        # Main Content Area
        if analysis_mode == "📋 Data Overview":
            show_data_overview(df)
        
        elif analysis_mode == "📈 Exploratory Analysis":
            show_exploratory_analysis(df)
    
    else:
        st.info("👈 Please upload a CSV file or select a sample dataset from the sidebar to begin.")
        
        # Show welcome information
        st.markdown("---")
        st.header("Welcome to the Data Science Dashboard! 🎉")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 Features")
            st.markdown("""
            - Data profiling & overview
            - Exploratory data analysis
            - Built-in visualizations
            - Correlation analysis
            - Distribution analysis
            """)
        
        with col2:
            st.subheader("🔧 Technologies")
            st.markdown("""
            - Streamlit native charts
            - Pandas & NumPy
            - Pure Python
            - Minimal dependencies
            """)
        
        with col3:
            st.subheader("🚀 Get Started")
            st.markdown("""
            1. Upload your CSV file
            2. Or select a sample dataset
            3. Choose an analysis mode
            4. Explore your data!
            """)

def show_data_overview(df):
    """Display comprehensive data overview"""
    st.header("📋 Data Overview")
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        st.metric("Numeric Columns", df.select_dtypes(include=[np.number]).shape[1])
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Data preview
    st.subheader("Data Preview")
    n_rows = st.slider("Number of rows to display:", 5, min(100, len(df)), 10)
    st.dataframe(df.head(n_rows), use_container_width=True)
    
    # Column information
    st.subheader("Column Information")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Null Count': df.isnull().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info, use_container_width=True)
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Missing values analysis
    if df.isnull().sum().sum() > 0:
        st.subheader("Missing Values Analysis")
        
        missing_summary = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
        })
        missing_summary = missing_summary[missing_summary['Missing Count'] > 0]
        
        if not missing_summary.empty:
            st.dataframe(missing_summary, use_container_width=True)
            
            # Bar chart of missing values
            st.bar_chart(missing_summary.set_index('Column')['Missing Count'])

def show_exploratory_analysis(df):
    """Display exploratory data analysis"""
    st.header("📈 Exploratory Data Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for analysis.")
        return
    
    # Distribution Analysis
    st.subheader("Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_col = st.selectbox("Select column for distribution:", numeric_cols)
    
    with col2:
        plot_type = st.selectbox("Plot type:", ["Histogram", "Line Chart"])
    
    if plot_type == "Histogram":
        st.subheader(f"Distribution of {selected_col}")
        hist_data = df[selected_col].dropna()
        st.bar_chart(hist_data.value_counts().sort_index())
    
    elif plot_type == "Line Chart":
        st.subheader(f"Line Chart of {selected_col}")
        st.line_chart(df[selected_col])
    
    # Correlation Analysis
    if len(numeric_cols) > 1:
        st.subheader("Correlation Analysis")
        
        corr_matrix = df[numeric_cols].corr()
        
        # Display correlation matrix as dataframe with color styling
        st.dataframe(
            corr_matrix.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1),
            use_container_width=True
        )
        
        # Top correlations
        st.subheader("Top Correlations")
        corr_pairs = corr_matrix.unstack()
        corr_pairs = corr_pairs[corr_pairs != 1.0]
        top_corr = corr_pairs.abs().sort_values(ascending=False).head(10)
        
        for (var1, var2), corr_val in top_corr.items():
            actual_corr = corr_matrix.loc[var1, var2]
            st.write(f"**{var1}** ↔ **{var2}**: {actual_corr:.3f}")
    
    # Scatter plot
    if len(numeric_cols) >= 2:
        st.subheader("Scatter Plot Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols, index=0)
        with col2:
            y_col = st.selectbox("Y-axis:", numeric_cols, index=min(1, len(numeric_cols)-1))
        
        # Create scatter plot using Streamlit
        scatter_data = df[[x_col, y_col]].dropna()
        st.scatter_chart(scatter_data, x=x_col, y=y_col)
    
    # Descriptive Statistics
    st.subheader("Descriptive Statistics")
    
    desc_col = st.selectbox("Select column for detailed statistics:", numeric_cols, key='desc_stats')
    
    if desc_col:
        data = df[desc_col].dropna()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Count", f"{len(data)}")
            st.metric("Mean", f"{data.mean():.2f}")
            st.metric("Median", f"{data.median():.2f}")
        
        with col2:
            st.metric("Std Dev", f"{data.std():.2f}")
            st.metric("Min", f"{data.min():.2f}")
            st.metric("Max", f"{data.max():.2f}")
        
        with col3:
            st.metric("25th %ile", f"{data.quantile(0.25):.2f}")
            st.metric("75th %ile", f"{data.quantile(0.75):.2f}")
            st.metric("IQR", f"{data.quantile(0.75) - data.quantile(0.25):.2f}")

if __name__ == "__main__":
    main()
