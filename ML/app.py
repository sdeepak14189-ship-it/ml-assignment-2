import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(
    page_title="ML Classification Models Comparison",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        padding: 20px;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #424242;
        margin-top: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🤖 Machine Learning Classification Models Comparison</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.title("📊 Model Selection")
st.sidebar.markdown("Choose a model to view its performance metrics")

# Model selection
model_options = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors (kNN)": "model/knn.pkl",
    "Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl",
    "XGBoost": "model/xgboost.pkl"
}

selected_model_name = st.sidebar.selectbox(
    "Select Model:",
    list(model_options.keys())
)

# Load scaler
@st.cache_resource
def load_scaler():
    try:
        with open('model/scaler.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all required evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # Calculate AUC if probabilities available
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['AUC'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
        except:
            metrics['AUC'] = 'N/A'
    else:
        metrics['AUC'] = 'N/A'
    
    return metrics

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<p class="sub-header">📤 Upload Test Dataset</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your test data (CSV format)",
        type=['csv'],
        help="Upload a CSV file with the same features as the training data"
    )

# Process uploaded file
if uploaded_file is not None:
    try:
        # Load data
        test_data = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Data loaded successfully! Shape: {test_data.shape}")
        
        # Show data preview
        with st.expander("👀 Preview Test Data"):
            st.dataframe(test_data.head(10))
        
        # Separate features and target
        if 'Actual' in test_data.columns:
            X_test = test_data.drop('Actual', axis=1)
            y_test = test_data['Actual']
        else:
            st.warning("⚠️ No 'Actual' column found. Using all columns as features.")
            X_test = test_data
            y_test = None
        
        # Load selected model
        model = load_model(model_options[selected_model_name])
        
        if model is not None:
            # Determine if scaling is needed
            needs_scaling = selected_model_name in ["Logistic Regression", "K-Nearest Neighbors (kNN)", "Naive Bayes"]
            
            if needs_scaling:
                scaler = load_scaler()
                if scaler is not None:
                    X_test_processed = scaler.transform(X_test)
                else:
                    X_test_processed = X_test
            else:
                X_test_processed = X_test
            
            # Make predictions
            y_pred = model.predict(X_test_processed)
            
            # Get prediction probabilities if available
            try:
                y_pred_proba = model.predict_proba(X_test_processed)
            except:
                y_pred_proba = None
            
            # Display results
            st.markdown("---")
            st.markdown(f'<p class="sub-header">📈 Results for {selected_model_name}</p>', unsafe_allow_html=True)
            
            # Show metrics if actual labels available
            if y_test is not None:
                metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Display metrics in columns
                st.markdown("### 📊 Performance Metrics")
                metric_cols = st.columns(3)
                
                metric_items = list(metrics.items())
                for idx, (metric_name, metric_value) in enumerate(metric_items):
                    with metric_cols[idx % 3]:
                        if metric_value != 'N/A':
                            st.metric(label=metric_name, value=f"{metric_value:.4f}")
                        else:
                            st.metric(label=metric_name, value=metric_value)
                
                # Confusion Matrix
                st.markdown("---")
                st.markdown("### 🔢 Confusion Matrix")
                
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
                ax.set_title(f'Confusion Matrix - {selected_model_name}', fontsize=14, fontweight='bold')
                ax.set_ylabel('Actual', fontweight='bold')
                ax.set_xlabel('Predicted', fontweight='bold')
                st.pyplot(fig)
                
                # Classification Report
                st.markdown("---")
                st.markdown("### 📋 Classification Report")
                
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.background_gradient(cmap='RdYlGn', subset=['precision', 'recall', 'f1-score']))
                
            else:
                # Just show predictions
                st.markdown("### 🎯 Predictions")
                results_df = X_test.copy()
                results_df['Predicted'] = y_pred
                
                if y_pred_proba is not None:
                    for i in range(y_pred_proba.shape[1]):
                        results_df[f'Probability_Class_{i}'] = y_pred_proba[:, i]
                
                st.dataframe(results_df.head(20))
                
                # Download predictions
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name=f'predictions_{selected_model_name.replace(" ", "_")}.csv',
                    mime='text/csv'
                )
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.info("Please ensure your CSV file has the correct format and features.")

else:
    # Show instructions when no file uploaded
    st.info("👆 Please upload a test dataset (CSV) to see model predictions and metrics.")
    
    st.markdown("""
    ### 📝 Instructions:
    1. **Upload your test data** in CSV format
    2. **Select a model** from the sidebar
    3. **View performance metrics** including Accuracy, AUC, Precision, Recall, F1, and MCC
    4. **Analyze the confusion matrix** and classification report
    
    ### 📊 Available Models:
    - **Logistic Regression**: Linear model for classification
    - **Decision Tree**: Tree-based model with interpretable rules
    - **K-Nearest Neighbors**: Instance-based learning algorithm
    - **Naive Bayes**: Probabilistic classifier based on Bayes theorem
    - **Random Forest**: Ensemble of decision trees
    - **XGBoost**: Gradient boosting ensemble method
    
    ### 💡 Sample Data Format:
    Your CSV should contain the same features used during training, with an optional 'Actual' column for true labels.
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Machine Learning Assignment 2 | M.Tech (AIML/DSE) | BITS Pilani</p>
    </div>
    """, unsafe_allow_html=True)
