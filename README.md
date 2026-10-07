# Customer Churn Prediction

## Project Overview
This project builds a machine learning system to predict customer churn for a telecommunications company. It uses real-world data to identify customers who are likely to cancel their subscription, enabling businesses to take proactive retention action.

## Problem Statement
Customer churn is costly for businesses. Acquiring a new customer costs 5 to 7 times more than retaining an existing one. This project predicts which customers are at risk of churning so that targeted retention strategies can be applied before they leave.

## Dataset
- Source: IBM Telco Customer Churn Dataset via Kaggle
- Records: 7,043 customers
- Features: 21 columns including demographics, services and billing
- Target: Churn Yes or No
- Class distribution: 73.5% No Churn and 26.5% Churn

## Technologies Used
- Python
- pandas and NumPy for data manipulation
- matplotlib and seaborn for visualisation
- scikit-learn for machine learning
- XGBoost for the final model
- Streamlit for the interactive web application
- pickle for model saving

## Project Pipeline
1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature encoding
5. Model training — Logistic Regression, Random Forest and XGBoost
6. Model evaluation using accuracy, precision, recall and F1 score
7. Feature importance analysis
8. Streamlit web app deployment

## Key Findings
- Month-to-month contract customers churn the most
- Higher monthly charges correlate with higher churn
- Customers with shorter tenure are more likely to churn
- Fibre optic internet service customers churn more than DSL customers

## How to Run
1. Clone this repository
2. Install dependencies: pip install -r requirements.txt
3. Run the Streamlit app: streamlit run app.py

## Author
Harika Bandari
- LinkedIn: https://www.linkedin.com/in/harikabandari1
- GitHub: github.com/HarikaBandari27
