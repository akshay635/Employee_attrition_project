# -*- coding: utf-8 -*-

import streamlit as st
import config
def load_user_inputs():
    
    # Employee ID
    employee_ID = st.sidebar.text_input('Please enter Employee ID')
    
    # Company name
    comp_name = st.sidebar.text_input('Company name')

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
    bt = st.sidebar.radio('Business Travel', options=config.BT_FEATURES)

    # Department 
    dept = st.sidebar.selectbox('Department', config.SOFTWARE_FIELDS)

    # Job role
    job_role = st.sidebar.selectbox('Job Role', config.SOFTWARE_ROLES)

    # Overtime
    overtime = st.sidebar.radio('Overtime', config.OVERTIME)

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
    
    return inputs