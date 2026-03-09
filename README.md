# Employee_attrition_project
# 📌 Project Objective and Approach

Developed an Employee Attrition Risk Estimation application using Python, machine learning, and Streamlit. The application predicts the likelihood of employee attrition based on workplace survey data and categorizes the risk into low, medium, or high levels using decision thresholds.

The dataset was collected from Kaggle, containing approximately 10–15k employee records with a mildly imbalanced target variable. The data was processed through a complete machine learning workflow, including data validation, feature engineering, and feature transformation.

Multiple classification models were implemented using machine learning pipelines, and model performance was evaluated using Stratified K-Fold cross-validation along with hyperparameter tuning to identify the best-performing model. The trained model was then tested on user-based inference data to estimate attrition probabilities.

Finally, the application was deployed as an interactive web app using Streamlit, with hosting on Render for public access.

#🔹 Key Results 

~Built an end-to-end employee attrition prediction pipeline using feature engineering, preprocessing, and cross-validated model comparison.

~Achieved ROC-AUC of 0.75 and PR-AUC of 0.49, demonstrating strong discrimination on imbalanced data.

~Optimized classification threshold (0.50 → 0.20) using Precision-Recall analysis, improving recall from 29% to 74% while maintaining 35% precision.

~Enabled proactive retention strategy by prioritizing high-risk employees based on cost-sensitive decision tuning.

Skills: Python (Programming Language) · Plotly · Scikit-Learn · Catboost · Streamlit · Pandas · NumPy · Joblib · Seaborn · Matplotlib
