📌 Project Overview

Employee attrition is a critical challenge for organizations, leading to increased hiring costs, loss of expertise, and reduced productivity. This project aims to predict the likelihood of an employee leaving the organization using Machine Learning, based on historical HR data.

The system provides data-driven insights that can help HR teams take proactive retention measures.

🎯 Objectives

Predict whether an employee is likely to leave the organization

Identify key factors influencing employee attrition

Build an interpretable and deployable ML model

Provide a user-friendly web interface for real-time predictions

📊 Dataset

Source: IBM HR Analytics Employee Attrition Dataset (Kaggle)

Records: ~1,470 employees

Target Variable: Attrition (Yes / No)

Features Include:

Demographics (Age, Gender, Marital Status)

Job-related factors (Department, Job Role, Job Level)

Compensation (Monthly Income, Salary Hike, Stock Options)

Experience (Years at Company, Total Working Years)

Satisfaction metrics (Job Satisfaction, Work-Life Balance)

Work conditions (OverTime, Business Travel)

🧠 Methodology

Data Preprocessing

Handling categorical and numerical features

One-Hot Encoding for categorical variables

Train-test split with class stratification

Model Development

Algorithm used: Random Forest Classifier

Class imbalance handled using class_weight='balanced'

Model trained using a Scikit-learn pipeline

Evaluation Metrics

Accuracy

Precision

Recall

F1-score

ROC–AUC

Confusion Matrix

Threshold Optimization

Probability-based prediction

Custom decision threshold to improve recall for attrition cases

📈 Model Performance (Sample Results)

Accuracy: ~84%

ROC–AUC: ~0.79

Attrition Recall (improved using threshold tuning): Significantly higher than default prediction

The model prioritizes identifying at-risk employees, which is crucial in HR analytics.

🌐 Web Application

The project includes a Streamlit-based web application that allows users to:

Input employee details

Predict attrition risk in real time

View probability-based predictions

Understand actionable HR insights

🚀 Deployment

Deployed using Streamlit Community Cloud

Free-tier deployment

Publicly accessible web interface

No server management required

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

Streamlit

Joblib

GitHub
