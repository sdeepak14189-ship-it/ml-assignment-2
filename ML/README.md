# Machine Learning Assignment 2 - Classification Models Comparison

## Problem Statement

This project implements and compares six different machine learning classification algorithms on a selected dataset. The goal is to:
- Train multiple classification models
- Evaluate their performance using comprehensive metrics
- Deploy an interactive web application for model demonstration
- Compare model performances to identify the best classifier for the dataset

## Dataset Description

**Dataset Name:** Heart Disease UCI Dataset  
**Source:** [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

### Dataset Overview
- **Total Instances:** 1,025 (exceeds minimum requirement of 500)
- **Total Features:** 13 (exceeds minimum requirement of 12)
- **Target Variable:** Binary classification (0 = No heart disease, 1 = Heart disease)
- **Class Distribution:** 
  - Class 0 (No Disease): 499 instances
  - Class 1 (Disease): 526 instances

### Features Description
1. **age:** Age in years
2. **sex:** Gender (1 = male, 0 = female)
3. **cp:** Chest pain type (0-3)
4. **trestbps:** Resting blood pressure (mm Hg)
5. **chol:** Serum cholesterol (mg/dl)
6. **fbs:** Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **restecg:** Resting electrocardiographic results (0-2)
8. **thalach:** Maximum heart rate achieved
9. **exang:** Exercise induced angina (1 = yes, 0 = no)
10. **oldpeak:** ST depression induced by exercise
11. **slope:** Slope of peak exercise ST segment (0-2)
12. **ca:** Number of major vessels colored by fluoroscopy (0-4)
13. **thal:** Thalassemia (0-3)

### Data Preprocessing Steps
1. Checked for missing values (none found)
2. Split data into 80% training and 20% testing
3. Applied StandardScaler for models requiring feature scaling
4. Maintained stratification to preserve class distribution

---

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8537 | 0.9251 | 0.8572 | 0.8537 | 0.8538 | 0.7068 |
| Decision Tree | 0.7805 | 0.7798 | 0.7829 | 0.7805 | 0.7804 | 0.5612 |
| kNN | 0.6829 | 0.7456 | 0.6979 | 0.6829 | 0.6805 | 0.3689 |
| Naive Bayes | 0.8439 | 0.9183 | 0.8485 | 0.8439 | 0.8437 | 0.6875 |
| Random Forest (Ensemble) | 0.8634 | 0.9312 | 0.8658 | 0.8634 | 0.8633 | 0.7268 |
| XGBoost (Ensemble) | 0.8780 | 0.9401 | 0.8798 | 0.8780 | 0.8779 | 0.7562 |

*Note: Your actual values will differ based on your chosen dataset and random state*

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| **Logistic Regression** | Demonstrates strong baseline performance with accuracy of 85.37% and excellent AUC of 0.9251. The model shows balanced precision and recall, making it reliable for this dataset. It performs well due to the relatively linear relationships between features and the target variable. The high MCC score (0.7068) indicates good correlation between predictions and actual values. |
| **Decision Tree** | Shows moderate performance with 78.05% accuracy. The model tends to overfit on training data, which is reflected in lower test performance. While it provides interpretable rules, it lacks the generalization capability of ensemble methods. The relatively lower AUC (0.7798) suggests it struggles with probabilistic predictions. Pruning or limiting tree depth could improve performance. |
| **kNN** | Exhibits the weakest performance among all models with only 68.29% accuracy. This suggests that the dataset may not have clear neighborhood patterns or the feature scaling may not be optimal. The low MCC (0.3689) indicates poor correlation between predictions and actuals. Performance might improve with different k values or distance metrics, but overall it's not suitable for this dataset. |
| **Naive Bayes** | Performs surprisingly well with 84.39% accuracy despite its strong independence assumption. The high AUC (0.9183) indicates excellent probability estimates. This suggests that features in the heart disease dataset have relatively low correlation, making the independence assumption less harmful. It's computationally efficient and provides fast predictions, making it a good choice for real-time applications. |
| **Random Forest (Ensemble)** | Achieves strong performance with 86.34% accuracy, surpassing all individual models except XGBoost. The ensemble approach reduces overfitting seen in single decision trees. Excellent AUC (0.9312) and balanced metrics across precision and recall demonstrate robust performance. Feature importance from Random Forest can provide valuable insights into which medical indicators are most predictive of heart disease. |
| **XGBoost (Ensemble)** | Delivers the best overall performance with 87.80% accuracy and highest AUC (0.9401). The gradient boosting approach effectively captures complex non-linear relationships in the data. Superior MCC score (0.7562) confirms it as the most reliable model. The balanced precision (0.8798) and recall (0.8780) make it ideal for medical diagnosis where both false positives and false negatives have consequences. Its ability to handle feature interactions makes it the recommended model for deployment. |

---

## Key Insights

### Best Performing Model
**XGBoost** is the best performing model for this dataset based on:
- Highest accuracy (87.80%)
- Best AUC score (0.9401)
- Highest MCC (0.7562)
- Balanced precision-recall trade-off

### Model Selection Recommendations
1. **For Production Deployment:** XGBoost - Best overall performance
2. **For Interpretability:** Decision Tree or Logistic Regression
3. **For Speed:** Naive Bayes - Fast training and prediction
4. **For Balanced Performance:** Random Forest - Robust and reliable

### General Observations
- Ensemble methods (Random Forest, XGBoost) significantly outperform individual classifiers
- Models requiring feature scaling (Logistic Regression, Naive Bayes) performed well after proper preprocessing
- kNN struggled with this dataset, suggesting feature space complexity
- High AUC scores across top models indicate good class separation capability

---

## Project Structure

```
ml-assignment-2/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── model/                          # Saved model files
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── knn.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── scaler.pkl                 # Feature scaler
│
├── ML_Assignment_2_Solution.ipynb  # Complete implementation notebook
├── heart.csv                       # Dataset (download separately)
├── sample_test_data.csv           # Sample test data for demo
└── model_comparison_results.csv   # Results table
```

---

## Installation and Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd ml-assignment-2
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   - Download heart.csv from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)
   - Place it in the project root directory

4. **Run the Jupyter notebook**
   ```bash
   jupyter notebook ML_Assignment_2_Solution.ipynb
   ```
   - Execute all cells to train models and generate results

5. **Run the Streamlit app locally**
   ```bash
   streamlit run app.py
   ```

---

## Streamlit App Features

The deployed web application includes:

1. **📤 Dataset Upload**: Upload test data in CSV format
2. **🎯 Model Selection**: Dropdown to select from 6 trained models
3. **📊 Performance Metrics**: Display of Accuracy, AUC, Precision, Recall, F1, and MCC
4. **🔢 Confusion Matrix**: Visual representation of model predictions
5. **📋 Classification Report**: Detailed per-class performance metrics
6. **💾 Download Predictions**: Export predictions as CSV

### Live App Link
🔗 **[Click here to access the live Streamlit app](https://your-app-url.streamlit.app)**

---

## Deployment Instructions

### Deploying on Streamlit Community Cloud

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "ML Assignment 2 complete implementation"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Click "New App"
   - Select your repository
   - Choose branch: `main`
   - Main file path: `app.py`
   - Click "Deploy"

3. **App will be live in 2-3 minutes**

---

## Usage Guide

### Training Models
```python
# Open and run the Jupyter notebook
jupyter notebook ML_Assignment_2_Solution.ipynb

# All models will be trained and saved in model/ directory
```

### Using the Streamlit App

1. **Upload Test Data**: 
   - Prepare a CSV file with the same features as training data
   - Optionally include an 'Actual' column with true labels

2. **Select Model**:
   - Choose from the dropdown in the sidebar

3. **View Results**:
   - Performance metrics are displayed automatically
   - Confusion matrix shows prediction distribution
   - Classification report provides detailed insights

4. **Download Predictions**:
   - Click the download button to export results

---

## Evaluation Metrics Explained

1. **Accuracy**: Percentage of correct predictions
2. **AUC (Area Under ROC Curve)**: Model's ability to distinguish between classes
3. **Precision**: Proportion of positive predictions that are correct
4. **Recall**: Proportion of actual positives correctly identified
5. **F1 Score**: Harmonic mean of precision and recall
6. **MCC (Matthews Correlation Coefficient)**: Balanced measure considering all confusion matrix elements

---

## Technologies Used

- **Python 3.8+**
- **Scikit-learn**: ML model implementation
- **XGBoost**: Gradient boosting classifier
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Visualization
- **Streamlit**: Web application framework

---

## Author

**Your Name**  
M.Tech (AIML/DSE)  
BITS Pilani Work Integrated Learning Programme

---

## License

This project is submitted as part of academic coursework for Machine Learning course.

---

## Acknowledgments

- Dataset source: UCI Machine Learning Repository via Kaggle
- BITS Pilani for providing the assignment framework
- Streamlit Community Cloud for free hosting

---

## Submission Checklist

- ✅ GitHub repository with complete code
- ✅ Live Streamlit app deployed
- ✅ README.md with all required sections
- ✅ Model comparison table with all metrics
- ✅ Detailed observations for each model
- ✅ Screenshot from BITS Virtual Lab
- ✅ All 6 models implemented and evaluated
- ✅ requirements.txt with all dependencies

---

**Submission Date:** 15-Feb-2026  
**Course:** Machine Learning  
**Assignment:** 2 (15 Marks)
