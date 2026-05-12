import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

st.set_page_config(page_title="HR Attrition Dashboard", layout="wide")

df = pd.read_csv("data/HR_Employee_Attrition.csv")
model = joblib.load("models/model.pkl")

df_model = df.copy()
df_model['Attrition']  = df_model['Attrition'].map({'Yes': 1, 'No': 0})
df_model['OverTime']   = df_model['OverTime'].map({'Yes': 1, 'No': 0})
df_model['Gender']     = df_model['Gender'].map({'Male': 1, 'Female': 0})

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df_model.select_dtypes(include='object').columns:
    df_model[col] = le.fit_transform(df_model[col])

for col in ['EmployeeCount','EmployeeNumber','Over18','StandardHours']:
    if col in df_model.columns:
        df_model.drop(col, axis=1, inplace=True)

feature_cols = [c for c in df_model.columns if c != 'Attrition']

st.title("HR Employee Attrition Dashboard")

# KPI Overview
st.header("Company Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Employees", len(df))
col2.metric("Attrition Rate",
            f"{df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%")
col3.metric("Avg Monthly Salary", f"${df['MonthlyIncome'].mean():,.0f}")

# Visual Analysis
st.header("Attrition Analysis")

dept = st.selectbox("Select Department", ['All'] + list(df['Department'].unique()))
filtered = df if dept == 'All' else df[df['Department'] == dept]

# Chart 1 :Salary distribution
fig1 = px.histogram(filtered, x='MonthlyIncome', color='Attrition',
                    barmode='overlay',
                    title='Salary Distribution by Attrition',
                    color_discrete_map={'Yes': 'red', 'No': 'steelblue'})
st.plotly_chart(fig1, use_container_width=True)

# Chart 2:Overtime vs Attrition
overtime_data = (filtered.groupby(['OverTime', 'Attrition'])
                         .size()
                         .reset_index(name='Count'))
fig2 = px.bar(overtime_data, x='OverTime', y='Count', color='Attrition',
              title='Overtime vs Attrition',
              color_discrete_map={'Yes': 'red', 'No': 'steelblue'})
st.plotly_chart(fig2, use_container_width=True)

st.header("Predict Employee Attrition Risk")
st.write("Enter employee details to predict if they might leave:")

col1, col2 = st.columns(2)

with col1:
    age          = st.slider("Age", 18, 60, 30)
    income       = st.number_input("Monthly Income", 1000, 20000, 5000, step=500)
    overtime     = st.selectbox("Works Overtime?", ['No', 'Yes'])
    satisfaction = st.slider("Job Satisfaction (1=Low, 4=High)", 1, 4, 3)
    years        = st.slider("Years at Company", 0, 40, 5)

with col2:
    distance     = st.slider("Distance from Home (km)", 1, 30, 10)
    work_life    = st.slider("Work-Life Balance (1=Low, 4=High)", 1, 4, 3)
    environment  = st.slider("Environment Satisfaction (1=Low, 4=High)", 1, 4, 3)
    job_level    = st.slider("Job Level (1-5)", 1, 5, 2)
    num_companies= st.slider("Number of Companies Worked", 0, 9, 2)

if st.button("Predict"):

    input_df = pd.DataFrame([df_model[feature_cols].median()], columns=feature_cols)

    input_df['Age']                    = age
    input_df['MonthlyIncome']          = income
    input_df['OverTime']               = 1 if overtime == 'Yes' else 0
    input_df['JobSatisfaction']        = satisfaction
    input_df['YearsAtCompany']         = years
    input_df['DistanceFromHome']       = distance
    input_df['WorkLifeBalance']        = work_life
    input_df['EnvironmentSatisfaction']= environment
    input_df['JobLevel']               = job_level
    input_df['NumCompaniesWorked']     = num_companies

    # Predict
    prediction  = model.predict(input_df)[0]          # 0 = stays, 1 = leaves
    probability = model.predict_proba(input_df)[0][1] # probability of leaving

    st.markdown("---")

    if prediction == 1:
        st.error(f" High Attrition Risk — {probability*100:.0f}% chance of leaving")
        st.write("**Recommendation:** Consider a salary review, reduce overtime load, or schedule a 1-on-1 check-in.")
    else:
        st.success(f"Low Attrition Risk — {probability*100:.0f}% chance of leaving")
        st.write("**Status:** Employee appears stable. Continue monitoring satisfaction scores.")