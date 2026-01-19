# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 13:58:50 2026

@author: aksha
"""

# Importing required modules
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import json
import shap

# Open and read the JSON file
with open('employee_attrition_project/data/medians.json', 'r') as file:
    medians = json.load(file)
    
# Open and read the JSON file
with open('employee_attrition_project/data/modes.json', 'r') as file:
    modes = json.load(file)


# Load csv data
@st.cache_data
def load_data():
    features_rf = pd.read_csv('employee_attrition_project/data/feature_importances_rf.csv')
    features_lg = pd.read_csv('employee_attrition_project/data/feature_importances_lg.csv')
    features_dt = pd.read_csv('employee_attrition_project/data/feature_importances_dt.csv')
    features_cat = pd.read_csv('employee_attrition_project/data/feature_importances_cat.csv')
    
    return features_lg, features_rf, features_dt, features_cat


# Load trained pipeline
@st.cache_resource
def load_models():
    model_lg = joblib.load('employee_attrition_project/models/lg_attrition.joblib')
    model_rf = joblib.load('employee_attrition_project/models/rf_attrition.joblib')
    model_dt = joblib.load('employee_attrition_project/models/dt_attrition.joblib')
    model_cat = joblib.load('employee_attrition_project/models/cat_attrition.joblib')
    
    return model_lg, model_rf, model_dt, model_cat


lg_df, rf_df, dt_df, cat_df = load_data()
model_lg, model_rf, model_dt, model_cat = load_models()

# common selected features 
common_features = ['Environment_Satisfaction', 'Salary_Hike_in_percent', 
                   'Salary', 'Job_Involvement', 'Years_since_last_promotion',
                   'Age', 'Overtime', 'Job_Satisfaction', 'Business_Travel',
                   'Distance_From_Home', 'Work_life_balance', 'Department', 'Job_Role']

st.set_page_config(page_title='Employee Attrition Prediction System', layout='wide')

st.title('Employee Attrition Prediction System')

st.sidebar.markdown(
    """
    <img src="https://whatfix.com/blog/wp-content/uploads/2022/09/employee-churn.png" 
         style="width:100%; margin-left:0;">
    """,
    unsafe_allow_html=True
)

# Employee ID
employee_ID = st.sidebar.text_input('Please enter Employee ID')

# Age
age = st.sidebar.number_input("Age", min_value=18, max_value=63)

# Salary
salary = st.sidebar.slider("Salary", min_value=30000, max_value=200000)

# Salary Hike in percent
ship = st.sidebar.slider("Salary Hike(%)", 0, 100)

# Work-Life balance
wlb = st.sidebar.number_input('Work-Life balance', min_value=1, max_value=5, step=1)

# Years since last promotion
yslp = st.sidebar.number_input('Years since last promotion', min_value=0, max_value=10, step=1)

# Distance from home
dist_f_home = st.sidebar.number_input('Distance from Home location', min_value=0, max_value=50, step=1)

# Job involvement
job_inv = st.sidebar.number_input("Job involvement", min_value=1, max_value=5, step=1)

# Environment Satisfaction
env_sats = st.sidebar.number_input("Environment Satisfaction", min_value=1, max_value=5, step=1)

# Job satisfaction
job_sats = st.sidebar.number_input("Job Satisfaction", min_value=1, max_value=5, step=1)

# Business Travel
bt = st.sidebar.radio('Business Travel', options=['Travel Rarely', 'No Travel', 'Travel Frequently'])

# Department 
dept = st.sidebar.selectbox('Department', ['Software Development', 'Cyber Security', 'Data Science',
                                           'Network Administration', 'IT Services'])

# Job role
job_role = st.sidebar.selectbox('Job Role', ['Developer', 'Software Engineer', 'IT', 'Technician', 
                                             'Support', 'Consultant', 'Director', 'HR', 'Help Desk', 
                                             'QA Analyst', 'Manager', 'Business Analyst'])

# Overtime
overtime = st.sidebar.radio('Overtime', ['Yes', 'No'])

# User inputs
inputs = {'Age': age,
          'Salary': salary,
          'Salary_Hike_in_percent': ship,
          'Work_life_balance': wlb, 
          'Years_since_last_promotion': yslp, 
          'Distance_From_Home': dist_f_home, 
          'Job_Involvement': job_inv,
          'Environment_Satisfaction': env_sats, 
          'Job_Satisfaction': job_sats,
          'Business_Travel': bt, 
          'Department': dept, 
          'Job_Role': job_role, 
          'Overtime': overtime
          }


with st.sidebar:
    st.image("employee_attrition_project/employee-attrition-rate.jpg", use_container_width=True)

# baseline input features on which models are trained
baseline = {}
for key, values in medians.items():
    medians[key] = int(values)
    
for i, j in modes.items():
    modes[i] = str(modes[i])
    
baseline.update(medians)
baseline.update(modes)

# Updating user inputs in the baseline inputs
final_inputs = baseline.copy()
final_inputs.update(inputs)

#model_box = st.selectbox('Choose a model', ['Logistic Regression', 'Catboost', 'Random Forest', 'Decision Tree'])
if st.button('Predict'):
    df = pd.DataFrame([final_inputs])
    predict = model_rf.predict(df)
    predict_proba = model_rf.predict_proba(df)[0, 1]
    if predict_proba < 0.35:
        st.success(f'✅ Employee is likely to stay with a low attrition risk score of {predict_proba:.2%}')
        st.write(f'Attrition rate: {predict_proba:.2%}')
    elif predict_proba >= 0.35 and predict_proba < 0.65:
        st.warning(f'⚠️ Employee has a moderate risk of leaving with a score of {predict_proba:.2%}')
        st.write(f'Attrition rate: {predict_proba:.2%}')
    else:
        st.error(f'❌ Employee is at high risk of leaving with a probability of {predict_proba:.2%}')
        st.write(f'Attrition rate: {predict_proba:.2%}')

col1, col2 = st.columns(2)

with col1:
    # ---------------- Visualization ----------------
    # feature importance scores
    fig = px.bar(
            rf_df.head(10).sort_values(by='importance', ascending=False),
            x="importance",
            y="feature",
            title=f"Feature Importance / F-score (Random Forest)",
            text_auto=True
    )
    st.plotly_chart(fig, use_container_width=True)

    """
        - Stay (Safe Zone) → <35%
        - Can Leave (Borderline Zone) → 35%-65%
        - Must Leave (Risk Zone) → >65%
    """

# estimating the probability of employee attrition rate with threshold settings
with col2:
    st.subheader("SHAP explanations", )
    st.text('Features contributions which decides the final outcome is shown using bar plot')
    preprocessor = model_rf.named_steps["preprocessing"]
    rf_model = model_rf.named_steps["rf_bal"]
    df_pre = preprocessor.transform(df)
    new_df = pd.DataFrame(df_pre, columns=preprocessor.get_feature_names_out())
    exp = shap.TreeExplainer(rf_model, feature_perturbation="tree_path_dependent")
    shap_values = exp(new_df)
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0, :, 1], max_display=10)
    st.pyplot(fig, use_container_width=True)
        
        
        

        























