 #HR Employee Attrition Analysis & Prediction

An end-to-end data analysis and machine learning and data analytics project that helps HR departments identify employees at risk of leaving and take action before it's too late.
Live Demo: 
-------------------------------------------------------------------------------------------------------
1-Problem Statement
Employee attrition costs companies 50–200% of an employee's annual salary in recruiting and onboarding. Most HR teams only discover the problem after an employee resigns. This project builds a system that predicts attrition risk in real time, giving HR teams a chance to intervene early.
-------------------------------------------------------------------------------------------------------
2- Key Findings (from EDA)

-Employees who work overtime leave at 3x the rate of those who don't
-Sales department has the highest attrition rate among all departments
-Employees who left earned on average significantly less than those who stayed
-------------------------------------------------------------------------------------------------------

3- Model Performance
The Random Forest classifier was trained on the IBM HR Analytics dataset (1,470 employees, 35 features).
Accuracy: 88%

              precision    recall  f1-score
No Attrition     0.88      1.00      0.94
Attrition        0.83      0.13      0.22
--------------------------------------------------------------------------------------------------------
4- Run Locally
1. Clone the repository
bashgit https://github.com/farah569/HR_ANALYTICS.git
cd HR_ANALYTICS
2. Install dependencies
install -r requirements.txt
3. Train the model
run src/train_model.py
4. Run the dashboard
python -m streamlit run app.py
---------------------------------------------------------------------------------------------------------
5- Dashboard Features

1.Overview Page:
Total employee count, attrition rate, average salary KPIs

2.Analysis Page:
Salary distribution by attrition status
Overtime vs attrition breakdown
Filterable by department

3.Predict Page:
HR enters employee details (age, salary, overtime, satisfaction, etc.)
Model returns: risk level + probability + recommendation
---------------------------------------------------------------------------------------------------------
6-Dataset
IBM HR Analytics Employee Attrition & Performance dataset from Kaggle.

1,470 employee records
35 features including demographics, job role, satisfaction scores, and compensation
---------------------------------------------------------------------------------------------------------

7- Future Improvements

1.Apply SMOTE to handle class imbalance and improve recall on attrition class
2.Connect to a live HR database instead of static CSV
3.Add email alert system when high-risk employees are detected
---------------------------------------------------------------------------------------------------------
