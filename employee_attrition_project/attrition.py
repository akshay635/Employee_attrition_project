# Importing required modules
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
import json
import importlib
import src.config as config
importlib.reload(config)
from src.load_data_and_models import load_data, load_models
from src.user_inputs import load_user_inputs
from src.final_user_data import final_inputs
from src.model_explanation import Feature_Importance, SHAP_explanations
from src.insights import generate_feature_insight 
from src.model_explanation import collapse_shap_values
from src.shap_exp import ShapCollapser

# Open and read the JSON file
with open(config.MEDIANS_PATH, 'r') as file:
    medians = json.load(file)
    
# Open and read the JSON file
with open(config.MODES_PATH, 'r') as file:
    modes = json.load(file)

# imported all the models and features importances, but Random Forest(rf) will be used as the best case
lg_df, rf_df, dt_df, cat_df = load_data()
model_lg, model_rf, model_dt, model_cat = load_models()

# common selected features 
common_features = config.COMMON_FEATURES

st.set_page_config(page_title='Employee Attrition Prediction System', layout='wide')

st.title('Employee Attrition Prediction System')

st.sidebar.markdown(
    """
    <img src="https://whatfix.com/blog/wp-content/uploads/2022/09/employee-churn.png" 
         style="width:100%; margin-left:0;">
    """,
    unsafe_allow_html=True
)

# User inputs
inputs = load_user_inputs()

with st.sidebar:
    st.image("employee_attrition_project/employee-attrition-rate.jpg", use_container_width=True)

final_inputs = final_inputs(inputs, medians, modes)

# model_box = st.selectbox('Choose a model', ['Logistic Regression', 'Catboost', 'Random Forest', 'Decision Tree'])
if st.button('Predict'):
    df = pd.DataFrame([final_inputs])
    predict = model_rf.predict(df)
    predict_proba = model_rf.predict_proba(df)[0, 1]
    if predict_proba < 0.35:
        st.success(f'✅ Employee is likely to stay with a low attrition risk score of {predict_proba:.2%}')
        st.write(f'Attrition rate: {predict_proba:.2%}')
    elif predict_proba >= 0.35 and predict_proba < 0.65:
        st.warning(f'⚠️ Employee has a moderate risk of leaving the organization with a score of {predict_proba:.2%}')
        st.write(f'Attrition rate: {predict_proba:.2%}')
    else:
        st.error(f'❌ Employee is at a high risk of leaving the organization with a probability of {predict_proba:.2%}')
        st.write(f'Attrition rate: {predict_proba:.2%}')

    col1, col2 = st.columns(2)

    with col1:
        # ---------------- Visualization ----------------
        # feature importance scores
        Feature_Importance(rf_df)
        with st.expander('Features in global feature importances'):
            st.markdown("### 🧠 Top 5 Feature Insights\n")
            st.dataframe(generate_feature_insight(rf_df))

    # estimating the probability of employee attrition rate with threshold settings
    with col2:
        shap_values, encoded_features = SHAP_explanations(model_rf, df)
        collapser = ShapCollapser(encoded_features, class_index=1)

        # Plot signed contributions for class 1
        fig, ax = plt.subplots()
        collapser.plot_signed_bar(shap_values, class_index=1, top_n=10)
        st.pyplot(fig, use_container_width=True)
        st.markdown("""Features shown in red increase the employee’s likelihood of leaving the organization, 
                       while features shown in blue increase the likelihood of the employee staying.""")
        
        with st.expander('Feature explanations using SHAP'):
            st.subheader("Top 5 features insights:")
            # Generate recruiter-friendly narrative
            st.markdown(collapser.explain(shap_values, top_n=3))
            #st.dataframe(shap_df_collapsed.head())
            #st.dataframe(collapse_shap_values(shap_values, encoded_features, config.COMMON_FEATURES))

    







        

















































