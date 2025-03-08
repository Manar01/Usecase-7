# Usecase-7
# Player Market Value Prediction - Machine Learning Project

## Introduction

Understanding a player's market value is crucial in football, affecting transfers, team investments, and scouting decisions. This project analyzes various player attributes and performance metrics to predict their market value accurately.

## Objectives

- Develop a regression and classification model to predict market value based on player attributes.
- Evaluate model performance and refine it using feature engineering and tuning.
- Apply multiple machine learning algorithms for comparison.
- Ensure models are validated and generalizable.

## Dataset Overview
- **Data Source:** Cleaned player data with attributes such as age, position, performance stats, and market value.
- **Features:** Player statistics, club details, and past performance.
- **Target Variables:**
  - **Regression:** Predict **market value (€)**.
  - **Classification:** Categorize players into **market value tiers**.
  - **Clustering:** Group players into **similar categories**.

## 🛠 Methodology

### **1️. Exploratory Data Analysis (EDA)**
✔ Checked missing values, duplicates, and feature correlations.  
✔ Visualized numerical & categorical distributions.  
✔ Detected outliers using boxplots and histograms.

1. **Data Viewing and Profiling:**
   - Checked dataset structure, column names, and data types.
   - Displayed summary statistics to understand the data distribution.
   - Identified and handled missing values.
   - Standardized categorical variables for consistency.
   - Removed duplicate records for data integrity.

2. **Data Cleaning:**
   - Handled missing values by imputation and removal of irrelevant features.
   - Encoded categorical variables using One-Hot Encoding and Label Encoding.
   - Normalized and standardized numerical variables to improve model efficiency.

3. **Univariate Analysis:**
   - Visualized market value distribution using histograms and box plots.
   - Examined player positions, age distribution, and frequency of key attributes.
   - Identified skewness in features and made transformations where necessary.

4. **Bivariate & Multivariate Analysis:**
   - Used heatmaps to explore correlations between numerical features.
   - Analyzed how age and position influence market value.
   - Compared market values across different positions using box plots.
   - Scatter plots to analyze relationships between continuous variables like goals and market value.

5. **Outlier Detection:**
   - Applied IQR filtering to remove extreme outliers in market value.
   - Used boxplots to detect anomalies in salary, goals, and other features.

## 2. Machine Learning Models

### **Regression Models**
1. **Feature Engineering:**
   - Encoded categorical features (position, awards) into numerical values.
   - Scaled numerical variables using StandardScaler and MinMaxScaler.
   - Selected features based on correlation analysis and importance ranking.

2. **Model Training:**
   - Trained multiple regression models including:
     - **Linear Regression**
     - **Ridge Regression**
     - **Lasso Regression**
   - Split data into training and testing sets for evaluation.

3. **Performance Evaluation:**
   - Used RMSE, MAE, and R-squared to measure model accuracy.
   - Compared results across different regression models to choose the best fit.

4. **Hyperparameter Tuning:**
   - Fine-tuned model parameters using GridSearchCV for optimization.
   - Selected best-performing parameters based on cross-validation scores.

5. **Validation & Overfitting Check:**
   - Compared training and test set performance to ensure generalization.
   - Analyzed residuals to check for normal distribution and errors.
**Best Model:** Ridge Regression (Balanced bias-variance tradeoff)  


### **Classification Models**
1. **Model Selection:**
   - **Logistic Regression:** Used to classify players into different market value categories.
   - **Decision Trees:** Used to capture complex relationships in player attributes.
   - **Random Forest:** Applied ensemble learning for better predictive accuracy.
   - **SVM**
   - **KNN** 

2. **Performance Metrics:**
   - Used **Accuracy, Precision, Recall, F1-score, and Confusion Matrix** for model evaluation.
   - Compared different classifiers to select the best-performing model.

3. **Overfitting Check:**
   - Applied **cross-validation** to ensure models generalize well.
   - Compared training and test performance to detect overfitting.

**Best Model:** **SVM with RBF Kernel** (Highest accuracy)


### **Unsupervised Modeling**

**K-Means Clustering**
- Identified **optimal k using Elbow Method & Silhouette Score**  
- Clustered players into meaningful groups (Elite, Young Talents, etc.)  
- Compared clusters based on attributes & market value  

**Best k:** **7 Clusters** (Interpreted as different player categories)


## **Model Comparison**
| Model                | Metric            | Score  |
|----------------------|------------------|--------|
| **Regression (Ridge)**   | R² Score  | **0.82** |
| **Classification (SVM)** | Accuracy  | **87.5%** |
| **Clustering (K-Means)** | Silhouette Score | **0.65** |

---

## How to Run the Project
1️ Install dependencies:  
```bash
pip install -r requirements.txt
2️ Run EDA Notebook to explore data.
3️ Train models using Regression & Classification Notebooks.
4️ Analyze clusters using Unsupervised Learning Notebook.
5️ Use the Final Report Notebook for insights.

## Key Insights

- **Younger players with strong performance metrics tend to have higher market values.**
- **Attackers and midfielders generally command higher market prices.**
- **Injuries and career length impact a player's worth significantly.**
- **Machine learning models provide decent predictions, but incorporating external factors (e.g., transfer rumors, club reputation) could improve accuracy.**
- **Random Forest outperformed Logistic Regression in classification tasks, providing better predictions for market value categorization.**

## Conclusion

This project provides insights into how different attributes influence a player's market value. By combining EDA with machine learning, we built predictive models that estimate a player's worth. Future improvements can include incorporating external data like contract details, club reputation, and league performance for better accuracy.