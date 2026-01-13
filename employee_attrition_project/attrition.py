# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 13:58:50 2026

@author: aksha
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import json


# Open and read the JSON file
with open('medians.json', 'r') as file:
    medians = json.load(file)
    
# Open and read the JSON file
with open('modes.json', 'r') as file:
    modes = json.load(file)


@st.cache_data
def load_data():
    features_rf = pd.read_csv('C:/Users/aksha/employee_attrition_project/data/feature_importances_rf.csv')
    features_lg = pd.read_csv('C:/Users/aksha/employee_attrition_project/data/feature_importances_lg.csv')
    features_dt = pd.read_csv('C:/Users/aksha/employee_attrition_project/data/feature_importances_dt.csv')
    features_cat = pd.read_csv('C:/Users/aksha/employee_attrition_project/data/feature_importances_cat.csv')
    
    return features_lg, features_rf, features_dt, features_cat

@st.cache_resource
def load_models():
    model_lg = joblib.load('C:/Users/aksha/employee_attrition_project/models/lg_attrition.joblib')
    model_rf = joblib.load('C:/Users/aksha/employee_attrition_project/models/rf_attrition.joblib')
    model_dt = joblib.load('C:/Users/aksha/employee_attrition_project/models/dt_attrition.joblib')
    model_cat = joblib.load('C:/Users/aksha/employee_attrition_project/models/cat_attrition.joblib')
    
    return model_lg, model_rf, model_dt, model_cat


lg_df, rf_df, dt_df, cat_df = load_data()
model_lg, model_rf, model_dt, model_cat = load_models()

common_features = ['Work_life_balance', 'Years_since_last_promotion', 
                   'Distance_From_Home', 'Job_Involvement',
                   'Environment_Satisfaction', 'Job_Satisfaction',
                   'Job_Level', 'Number_of_Companies_Worked_previously',
                   'Business_Travel', 'Department', 'Job_Role', 'Overtime']

st.set_page_config(page_title='Employee Attrition Prediction System', layout='wide')

st.title('Employee Attrition Prediction System')

st.sidebar.markdown(
    """
    <img src="https://whatfix.com/blog/wp-content/uploads/2022/09/employee-churn.png" 
         style="width:100%; margin-left:0;">
    """,
    unsafe_allow_html=True
)


wlb = st.sidebar.number_input('Work-Life balance', min_value=1, max_value=5, step=1)
yslp = st.sidebar.number_input('Years since last promotion', min_value=0, max_value=10, step=1)
dist_f_home = st.sidebar.number_input('Distance from Home location', min_value=0, max_value=50, step=1)
job_inv = st.sidebar.number_input("Job involvement", min_value=1, max_value=5, step=1)
env_sats = st.sidebar.number_input("Environment Satisfaction", min_value=1, max_value=5, step=1)
job_sats = st.sidebar.number_input("Job Satisfaction", min_value=1, max_value=5, step=1)
job_level = st.sidebar.number_input("Job level", min_value=1, max_value=8, step=1)
nocwp = st.sidebar.number_input("No. of Companies worked previously", min_value=0, max_value=8, step=1)
bt = st.sidebar.radio('Business Travel', options=['Travel Rarely', 'No Travel', 'Travel Frequently'])
dept = st.sidebar.selectbox('Department', ['Software Development', 'Cyber Security', 'Data Science',
                                           'Network Administration', 'IT Services'])
job_role = st.sidebar.selectbox('Job Role', ['Developer', 'Software Engineer', 'IT', 'Technician', 
                                             'Support', 'Consultant', 'Director', 'HR', 'Help Desk', 
                                             'QA Analyst', 'Manager', 'Business Analyst'])

overtime = st.sidebar.radio('Overtime', ['Yes', 'No'])

inputs = {'Work_life_balance': wlb, 
          'Years_since_last_promotion': yslp, 
          'Distance_From_Home': dist_f_home, 
          'Job_Involvement': job_inv,
          'Environment_Satisfaction': env_sats, 
          'Job_Satisfaction': job_sats,
          'Job_Level': job_level, 
          'Number_of_Companies_Worked_previously': nocwp,
          'Business_Travel': bt, 
          'Department': dept, 
          'Job_Role': job_role, 
          'Overtime': overtime
          }


with st.sidebar:
    st.image("C:/Users/aksha/employee_attrition_project/employee-attrition-rate.jpg", use_container_width=True)


baseline = {}
for key, values in medians.items():
    medians[key] = int(values)
    
for i, j in modes.items():
    modes[i] = str(modes[i])
    
baseline.update(medians)
baseline.update(modes)

final_inputs = baseline.copy()
final_inputs.update(inputs)

model_box = st.selectbox('Choose a model', ['Logistic Regression', 'Catboost', 'Random Forest', 'Decision Tree'])

col1, col2 = st.columns(2)

with col1:
    if model_box == "Logistic Regression":
        # ---------------- Visualization ----------------
        fig = px.bar(
            lg_df.head(10).sort_values(by='importance', ascending=False),
            x="importance",
            y="feature",
            title=f"Feature Importance / F-score ({model_box})",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if model_box == 'Decision Tree':
        # ---------------- Visualization ----------------
        fig = px.bar(
            dt_df.head(10).sort_values(by='importance', ascending=False),
            x="importance",
            y="feature",
            title=f"Feature Importance / F-score ({model_box})",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
    if model_box == 'Random Forest':
        # ---------------- Visualization ----------------
        fig = px.bar(
            rf_df.head(10).sort_values(by='importance', ascending=False),
            x="importance",
            y="feature",
            title=f"Feature Importance / F-score ({model_box})",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
    if model_box == 'Catboost':
        # ---------------- Visualization ----------------
        fig = px.bar(
            cat_df.head(10).sort_values(by='importance', ascending=False),
            x="importance",
            y="feature",
            title=f"Feature Importance / F-score ({model_box})",
            text_auto=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
with col2:
    if st.button('Predict'):
        df = pd.DataFrame([final_inputs])
        if model_box == 'Logistic Regression':
            predict = model_lg.predict(df)
            predict_proba = model_lg.predict_proba(df)[0, 1]
            
            if predict_proba >= 0.45:
                st.warning(f'Employee can leave the organization with attrition rate of {predict_proba:.2%}')
            else:
                st.success(f'Employee can continue in the organization with attrition rate of {predict_proba:.2%}')
        
        if model_box == 'Decision Tree':
            predict = model_dt.predict(df)
            predict_proba = model_dt.predict_proba(df)[0, 1]
            
            if predict_proba >= 0.45:
                st.warning(f'Employee can leave the organization with attrition rate of {predict_proba:.2%}')
            else:
                st.success(f'Employee can continue in the organization with attrition rate of {predict_proba:.2%}')
        
        if model_box == 'Random Forest':
            predict = model_rf.predict(df)
            predict_proba = model_rf.predict_proba(df)[0, 1]
            
            if predict_proba >= 0.45:
                st.warning(f'Employee can leave the organization with attrition rate of {predict_proba:.2%}')
            else:
                st.success(f'Employee can continue in the organization with attrition rate of {predict_proba:.2%}')
            
        if model_box == 'Catboost':
            df = df[['Age', 'Business_Travel', 'Department', 'Distance_From_Home',
                      'Education', 'Environment_Satisfaction', 'Gender', 'Salary',
                      'Job_Involvement', 'Job_Level', 'Job_Role', 'Job_Satisfaction',
                      'Marital_Status', 'Number_of_Companies_Worked_previously', 
                      'Overtime', 'Salary_Hike_in_percent', 'Total_working_years_experience',
                      'Work_life_balance', 'No_of_years_worked_at_current_company',
                      'No_of_years_in_current_role', 'Years_since_last_promotion']]
            predict = model_cat.predict(df)
            predict_proba = model_cat.predict_proba(df)[0, 1]
            
            if predict >= 0.45:
                st.warning(f'Employee can leave the organization with attrition rate of {predict_proba:.2%}')
            else:
                st.success(f'Employee can continue in the organization with attrition rate of {predict_proba:.2%}')
        
        st.write(final_inputs)
        
        
        
        