"""
Interactive Data Science Dashboard
A comprehensive Streamlit app for data analysis, visualization, and machine learning
Author: AI Assistant
Version: 2.0 (Matplotlib-free, 100% Plotly)
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import probplot
import plotly.express as px
import plotly.graph_objects as go
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
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        return df
    
    elif dataset_name == "Wine Quality":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
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
    st.markdown("**Explore, Analyze, and Model Your Data**")
    
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
                "📈 Exploratory Analysis",
                "🔍 Statistical Analysis",
                "🤖 Machine Learning",
                "📊 Interactive Visualizations"
            ]
        )
        
        # Main Content Area
        if analysis_mode == "📋 Data Overview":
            show_data_overview(df)
        
        elif analysis_mode == "📈 Exploratory Analysis":
            show_exploratory_analysis(df)
        
        elif analysis_mode == "🔍 Statistical Analysis":
            show_statistical_analysis(df)
        
        elif analysis_mode == "🤖 Machine Learning":
            show_machine_learning(df)
        
        elif analysis_mode == "📊 Interactive Visualizations":
            show_interactive_visualizations(df)
    
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
            - Statistical analysis
            - Interactive visualizations
            - Machine learning models
            - Export capabilities
            """)
        
        with col2:
            st.subheader("🔧 Technologies")
            st.markdown("""
            - Pandas & NumPy
            - Plotly (interactive charts)
            - Scikit-learn (ML)
            - SciPy (statistics)
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
    
    # Missing values heatmap
    if df.isnull().sum().sum() > 0:
        st.subheader("Missing Values Heatmap")
        
        # Create heatmap data
        missing_data = df.isnull().astype(int)
        
        fig = px.imshow(missing_data.T, 
                       labels=dict(x="Row Index", y="Column", color="Missing"),
                       color_continuous_scale=['lightblue', 'darkblue'],
                       title="Missing Values Pattern")
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

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
        plot_type = st.selectbox("Plot type:", ["Histogram", "Box Plot", "Violin Plot"])
    
    if plot_type == "Histogram":
        fig = px.histogram(df, x=selected_col, nbins=30, 
                          title=f"Distribution of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Box Plot":
        fig = px.box(df, y=selected_col, 
                    title=f"Box Plot of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif plot_type == "Violin Plot":
        fig = px.violin(df, y=selected_col, box=True,
                       title=f"Violin Plot of {selected_col}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Correlation Analysis
    if len(numeric_cols) > 1:
        st.subheader("Correlation Analysis")
        
        corr_matrix = df[numeric_cols].corr()
        
        # Create interactive heatmap with Plotly
        fig = px.imshow(corr_matrix, 
                       text_auto='.2f',
                       color_continuous_scale='RdBu_r',
                       color_continuous_midpoint=0,
                       title="Correlation Heatmap",
                       labels=dict(color="Correlation"))
        fig.update_xaxes(side="bottom")
        st.plotly_chart(fig, use_container_width=True)
        
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
        
        # Create scatter plot with trendline
        fig = px.scatter(df, x=x_col, y=y_col, 
                        trendline="ols",
                        title=f"{y_col} vs {x_col}")
        st.plotly_chart(fig, use_container_width=True)

def show_statistical_analysis(df):
    """Display statistical analysis"""
    st.header("🔍 Statistical Analysis")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for analysis.")
        return
    
    # Hypothesis Testing
    st.subheader("Hypothesis Testing")
    
    test_type = st.selectbox(
        "Select test type:",
        ["T-Test (One Sample)", "T-Test (Two Independent)", "Chi-Square Test", "ANOVA"]
    )
    
    if test_type == "T-Test (One Sample)":
        col = st.selectbox("Select column:", numeric_cols)
        pop_mean = st.number_input("Population mean:", value=float(df[col].mean()))
        
        if st.button("Run Test"):
            t_stat, p_value = stats.ttest_1samp(df[col].dropna(), pop_mean)
            
            st.write(f"**T-statistic:** {t_stat:.4f}")
            st.write(f"**P-value:** {p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ Reject null hypothesis (p < 0.05)")
            else:
                st.info("❌ Fail to reject null hypothesis (p >= 0.05)")
    
    elif test_type == "T-Test (Two Independent)" and len(numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            group1 = st.selectbox("Group 1:", numeric_cols, index=0)
        with col2:
            group2 = st.selectbox("Group 2:", numeric_cols, index=1)
        
        if st.button("Run Test"):
            t_stat, p_value = stats.ttest_ind(df[group1].dropna(), df[group2].dropna())
            
            st.write(f"**T-statistic:** {t_stat:.4f}")
            st.write(f"**P-value:** {p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ Reject null hypothesis (p < 0.05)")
            else:
                st.info("❌ Fail to reject null hypothesis (p >= 0.05)")
    
    # Normality Test
    st.subheader("Normality Testing")
    norm_col = st.selectbox("Select column for normality test:", numeric_cols, key='norm')
    
    if st.button("Test Normality"):
        stat, p_value = stats.shapiro(df[norm_col].dropna().sample(min(5000, len(df[norm_col].dropna()))))
        
        st.write(f"**Shapiro-Wilk statistic:** {stat:.4f}")
        st.write(f"**P-value:** {p_value:.4f}")
        
        if p_value < 0.05:
            st.warning("⚠️ Data is NOT normally distributed (p < 0.05)")
        else:
            st.success("✅ Data appears normally distributed (p >= 0.05)")
        
        # Q-Q Plot
        qq_data = probplot(df[norm_col].dropna(), dist="norm")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], 
                                mode='markers', name='Data',
                                marker=dict(color='blue', size=5)))
        fig.add_trace(go.Scatter(x=qq_data[0][0], 
                                y=qq_data[1][0] * qq_data[0][0] + qq_data[1][1],
                                mode='lines', name='Theoretical',
                                line=dict(color='red', dash='dash')))
        fig.update_layout(title=f"Q-Q Plot for {norm_col}",
                         xaxis_title="Theoretical Quantiles",
                         yaxis_title="Sample Quantiles")
        st.plotly_chart(fig, use_container_width=True)
    
    # Descriptive Statistics
    st.subheader("Detailed Descriptive Statistics")
    desc_col = st.selectbox("Select column:", numeric_cols, key='desc')
    
    data = df[desc_col].dropna()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean", f"{data.mean():.2f}")
        st.metric("Median", f"{data.median():.2f}")
        st.metric("Mode", f"{data.mode().iloc[0]:.2f}" if len(data.mode()) > 0 else "N/A")
    
    with col2:
        st.metric("Std Dev", f"{data.std():.2f}")
        st.metric("Variance", f"{data.var():.2f}")
        st.metric("Range", f"{data.max() - data.min():.2f}")
    
    with col3:
        st.metric("Skewness", f"{data.skew():.2f}")
        st.metric("Kurtosis", f"{data.kurtosis():.2f}")
        st.metric("IQR", f"{data.quantile(0.75) - data.quantile(0.25):.2f}")

def show_machine_learning(df):
    """Display machine learning models"""
    st.header("🤖 Machine Learning")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for machine learning.")
        return
    
    # Model Selection
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_type = st.selectbox(
            "Select model type:",
            ["Classification", "Regression"]
        )
    
    with col2:
        target_col = st.selectbox("Select target variable:", numeric_cols)
    
    feature_cols = st.multiselect(
        "Select features:",
        [col for col in numeric_cols if col != target_col],
        default=[col for col in numeric_cols if col != target_col][:min(5, len(numeric_cols)-1)]
    )
    
    if len(feature_cols) == 0:
        st.warning("Please select at least one feature.")
        return
    
    test_size = st.slider("Test set size:", 0.1, 0.5, 0.2, 0.05)
    
    if st.button("Train Model"):
        # Prepare data
        X = df[feature_cols].dropna()
        y = df.loc[X.index, target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        with st.spinner("Training model..."):
            if model_type == "Classification":
                # Train Random Forest Classifier
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                
                # Results
                st.subheader("Classification Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Training Accuracy", f"{model.score(X_train_scaled, y_train):.3f}")
                    st.metric("Test Accuracy", f"{model.score(X_test_scaled, y_test):.3f}")
                
                # Confusion Matrix
                with col2:
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Create confusion matrix heatmap with Plotly
                    fig = px.imshow(cm, 
                                   text_auto=True,
                                   color_continuous_scale='Blues',
                                   labels=dict(x="Predicted Label", y="True Label", color="Count"),
                                   title="Confusion Matrix")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Classification Report
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred))
            
            else:  # Regression
                # Train Random Forest Regressor
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_test_scaled)
                
                # Results
                st.subheader("Regression Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
                
                with col2:
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
                
                with col3:
                    st.metric("MAE", f"{np.abs(y_test - y_pred).mean():.3f}")
                
                # Prediction vs Actual
                fig = px.scatter(x=y_test, y=y_pred, 
                               labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                               title="Predicted vs Actual Values")
                
                # Add diagonal line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                        mode='lines', name='Perfect Prediction',
                                        line=dict(color='red', dash='dash')))
                st.plotly_chart(fig, use_container_width=True)
                
                # Residual Plot
                residuals = y_test - y_pred
                fig = px.scatter(x=y_pred, y=residuals,
                               labels={'x': 'Predicted Values', 'y': 'Residuals'},
                               title="Residual Plot")
                fig.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature Importance
            st.subheader("Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=True)
            
            fig = px.bar(feature_importance, x='Importance', y='Feature', 
                        orientation='h',
                        title="Feature Importance")
            st.plotly_chart(fig, use_container_width=True)

def show_interactive_visualizations(df):
    """Display interactive Plotly visualizations"""
    st.header("📊 Interactive Visualizations")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        st.warning("No numeric columns found for visualization.")
        return
    
    # Visualization type
    viz_type = st.selectbox(
        "Select visualization type:",
        ["Scatter Plot", "Line Chart", "Bar Chart", "3D Scatter", "Box Plot", "Violin Plot"]
    )
    
    if viz_type == "Scatter Plot" and len(numeric_cols) >= 2:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols, index=0)
        with col2:
            y_col = st.selectbox("Y-axis:", numeric_cols, index=1)
        with col3:
            color_col = st.selectbox("Color by:", [None] + df.columns.tolist())
        
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} vs {x_col}",
                        hover_data=df.columns)
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Line Chart":
        y_cols = st.multiselect("Select columns to plot:", numeric_cols, default=numeric_cols[:3])
        
        if len(y_cols) > 0:
            fig = go.Figure()
            for col in y_cols:
                fig.add_trace(go.Scatter(y=df[col], name=col, mode='lines'))
            
            fig.update_layout(title="Line Chart", xaxis_title="Index", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Bar Chart":
        col = st.selectbox("Select column:", numeric_cols)
        
        fig = px.bar(df, y=col, title=f"Bar Chart of {col}")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "3D Scatter" and len(numeric_cols) >= 3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols, index=0, key='3d_x')
        with col2:
            y_col = st.selectbox("Y-axis:", numeric_cols, index=1, key='3d_y')
        with col3:
            z_col = st.selectbox("Z-axis:", numeric_cols, index=2, key='3d_z')
        
        color_col = st.selectbox("Color by:", [None] + df.columns.tolist(), key='3d_color')
        
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col,
                           title=f"3D Scatter Plot")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Box Plot":
        cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])
        
        if len(cols) > 0:
            fig = go.Figure()
            for col in cols:
                fig.add_trace(go.Box(y=df[col], name=col))
            
            fig.update_layout(title="Box Plot Comparison")
            st.plotly_chart(fig, use_container_width=True)
    
    elif viz_type == "Violin Plot":
        cols = st.multiselect("Select columns:", numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))], key='violin')
        
        if len(cols) > 0:
            fig = go.Figure()
            for col in cols:
                fig.add_trace(go.Violin(y=df[col], name=col, box_visible=True))
            
            fig.update_layout(title="Violin Plot Comparison")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
