# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import shap
import importlib
import src.config as config
importlib.reload(config)

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
    
def collapse_shap_values(shap_values, encoded_feature_names, original_features):
    """
    Collapse one-hot encoded SHAP values back to original categorical features.

    Parameters:
    - shap_values: SHAP values object (shap.Explanation or array)
    - encoded_feature_names: list of encoded feature names (from encoder.get_feature_names_out)
    - encoder: fitted OneHotEncoder (or similar)
    - original_features: list of original categorical feature names

    Returns:
    - shap_df_collapsed: DataFrame with SHAP values grouped by original features
    """
    shap_df = pd.DataFrame(shap_values.values[:, :, 1], columns=encoded_feature_names)

    # Collapse each categorical feature
    for i, feature in enumerate(original_features):
        # Get encoded columns for this feature
        encoded_cols = [col for col in encoded_feature_names if col.startswith('cat' + "__" + feature + '_')]
        if encoded_cols:
            shap_df[feature] = shap_df[encoded_cols].sum(axis=1)
            shap_df.drop(columns=encoded_cols, inplace=True)

    return shap_df
    
def SHAP_explanations(model, df):
    st.subheader("SHAP explanations")
    st.text("Features contributions which decides the final outcome.")
    preprocessor = model.named_steps["preprocessing"]
    rf_model = model.named_steps["rf_bal"]
    encoded_features = preprocessor.get_feature_names_out()
    df_pre = preprocessor.transform(df)
    new_df = pd.DataFrame(df_pre, columns=preprocessor.get_feature_names_out())
    exp = shap.TreeExplainer(rf_model, feature_perturbation="tree_path_dependent")
    shap_values = exp(new_df)
    fig, ax = plt.subplots()
    shap.plots.bar(shap_values[0, :, 1], max_display=10)
    st.pyplot(fig, use_container_width=True)
    st.markdown("""Features shown in red increase the employee’s likelihood of leaving the organization, 
                   while features shown in blue increase the likelihood of the employee staying.""")
    
    return shap_values, encoded_features

    
















