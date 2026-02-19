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
target = config.TARGET

st.set_page_config(page_title='Employee Attrition Prediction System', layout='wide')

st.title('Employee Attrition Rate Estimator')

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
tab1, tab2 = st.tabs(['Single employee Prediction', 'Batch Prediction'])
with tab1:
    if st.button('Predict'):
    df = pd.DataFrame([final_inputs])
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
        collapser.plot_signed_bar(shap_values, class_index=1)
        st.pyplot(fig, use_container_width=False)
        st.markdown("""Features shown in red increase the employee’s likelihood of leaving the organization, 
                       while features shown in blue increase the likelihood of the employee staying.""")
        
        with st.expander('Feature explanations using SHAP'):
            st.subheader("Top 5 features insights:")
            # Generate recruiter-friendly narrative
            st.markdown(collapser.explain(shap_values, top_n=3))

with tab2:
    st.title("📊 Portfolio Attrition Evaluation – Batch Processing")
    st.markdown("""
    Upload employees dataset collected from forms, surveys to perform portfolio-level attrition risk scoring.
    The model applies cost-sensitive learning and threshold-based decision logic.
    """)

    uploaded_file = st.file_uploader(
    "Upload CSV file containing employees data (must include target column)",
    type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"File uploaded successfully. Records detected: {len(df)}")
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head(2))

        required_features = common_features + target
        missing_features = [col for col in required_features if col not in df.columns]

        if missing_cols:
            st.error(f"Missing required columns: {missing_cols}")
            st.stop()
  
        st.header("⚙️ Decision Configuration")

        threshold = st.slider(
              "Decision Threshold",
              min_value=0.0,
              max_value=1.0,
              value=0.5,
              step=0.01)

        y_true = df[target]
        X_batch = df[common_features]

        y_proba = model_rf.predict_proba(X_batch)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
      
        df["Probability"] = y_proba
        df["Prediction"] = y_pred
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        recall = tp / (tp + fn)
        miss_rate = fn / (tp + fn)
        precision = tp / (tp + fp)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        flagged_rate = y_pred.mean()

        st.header("📌 Portfolio Summary")

        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric("Total Records", len(df))
        col2.metric("Flagged High Risk", f"{flagged_rate*100:.2f}%")
        col3.metric("Recall (Catch Rate)", f"{recall*100:.2f}%")
        col4.metric("Miss Rate", f"{miss_rate*100:.2f}%")

        st.subheader("🔎 Confusion Matrix")

        st.write(f"""
        - True Positives: {tp}
        - False Positives: {fp}
        - True Negatives: {tn}
        - False Negatives: {fn}
        """)

        df["Risk Bucket"] = pd.cut(y_proba, bins=[0, 0.3, 0.6, 1],
                                   labels=["Low Risk", "Medium Risk", "High Risk"])

        st.subheader("📊 Risk Segmentation Distribution")
        
        st.bar_chart(df["Risk Bucket"].value_counts())

        st.subheader("⬇️ Export Scored Portfolio")

        st.download_button(label="Download Scored Dataset", data=df.to_csv(index=False), 
                           file_name="scored_portfolio.csv", mime="text/csv")

        st.info(f"""At threshold {threshold}, the model detects {recall*100:.1f}% of defaulters 
        while missing {miss_rate*100:.1f}%. Approximately {flagged_rate*100:.1f}% 
        of the portfolio is flagged for review.
        """)

    







        
























































