# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib

#Load csv data
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

