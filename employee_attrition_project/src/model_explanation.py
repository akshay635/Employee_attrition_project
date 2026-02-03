# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap

def Feature_Importance(df):
    # ---------------- Visualization ----------------
    # feature importance scores
    fig = px.bar(
        df.head(10).sort_values(by='importance', ascending=False),
        x="importance",
        y="feature",
        title="Feature Importance / F-score (Random Forest)",
        text_auto=True
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    """
    - Stay (Safe Zone) → <35%
    - Can Leave (Borderline Zone) → 35%-65%
    - Must Leave (Risk Zone) → >65%
    """
    
def SHAP_explanations(model, df):
    st.subheader("SHAP explanations")
    st.text("Features contributions which decides the final outcome.")
    preprocessor = model.named_steps["preprocessing"]
    rf_model = model.named_steps["rf_bal"]
    df_pre = preprocessor.transform(df)
    new_df = pd.DataFrame(df_pre, columns=preprocessor.get_feature_names_out())
    exp = shap.TreeExplainer(rf_model, feature_perturbation="tree_path_dependent")
    shap_values = exp(new_df)
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0, :, 1], max_display=10)
    st.pyplot(fig, use_container_width=True)
    st.markdown("""Features shown in red increase the employee’s likelihood of leaving the organization, 
                   while features shown in blue increase the likelihood of the employee staying.""")
    