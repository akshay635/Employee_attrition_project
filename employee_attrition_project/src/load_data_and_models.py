# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

#Load csv data
@st.cache_data
def load_data():
    features_rf = pd.read_csv('employee_attrition_project/data/forest_importances.csv')
    
    return features_rf

# Load trained pipeline
@st.cache_resource
def load_models():
    model_rf = joblib.load('employee_attrition_project/models/current_model.joblib')
    
    return model_rf



