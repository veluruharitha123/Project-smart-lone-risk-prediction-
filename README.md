# Project-smart-lone-risk-prediction-
learning-based system that analyzes applicant income, credit score, employment status, and loan amount to predict loan approval or rejection. It helps financial institutions reduce risk, automate decision-making, and improve efficiency using data-driven insights.
# Smart Loan Risk Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create Dataset
data = {
    'income': [50000, 30000, 80000, 25000, 60000, 45000, 70000, 20000],
    'credit_score': [700, 600, 750, 580, 720, 650, 710, 500],
    'loan_amount': [20000, 15000, 30000, 10000, 25000, 18000, 27000, 9000],
    'employment_years': [5, 2, 7, 1, 6, 3, 8, 1],
    'approved': [1, 0, 1, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['income', 'credit_score', 'loan_amount', 'employment_years']]
y = df['approved']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test Accuracy
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Predict New Applicant
income = float(input("Enter Income: "))
credit_score = int(input("Enter Credit Score: "))
loan_amount = float(input("Enter Loan Amount: "))
employment_years = int(input("Enter Employment Years: "))

new_data = [[income, credit_score, loan_amount, employment_years]]
result = model.predict(new_data)

if result[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")
