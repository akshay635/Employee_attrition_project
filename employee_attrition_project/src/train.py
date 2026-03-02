# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore')
import json
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_auc_score, average_precision_score)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import loguniform

np.random.seed(42)
pd.set_option('display.max_columns', 500)

data = pd.read_csv('employee_attrition_project/data/HR_attrition_dataset.csv')

print(data.shape)
print(data.head(1))
print(data.info())
print(data.columns)
print(data.isna().sum())
print(data.isnull().sum())

data['Attrition'] = data['Attrition'].map({'Yes': 1, 'No': 0})

print(data['Attrition'].dtype)

#print(data.head())

X = data.drop(columns=['Employee_ID', 'Attrition'])
y = data[['Attrition']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

num_cols = X_train.select_dtypes(include=['int', 'float']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

plt.figure(figsize=(14, 8))
sns.heatmap(X_train[num_cols].corr(), annot=True)
plt.show()

# Assuming 'data' contains only features
# 1. Compute Correlation Matrix
corr_matrix = data[num_cols + ['Attrition']].corr()

# 2. Visualize with Heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

num_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('encode', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
], remainder='passthrough')

pos_count = np.sum(y['Attrition']==1)
neg_count = np.sum(y['Attrition']==0)

scale_pos_weight = neg_count / pos_count

scale_pos_weight = round(scale_pos_weight, 2)

models = {
    'Log_Reg': LogisticRegression(random_state=42, max_iter=2000, class_weight={0:1.0, 1:scale_pos_weight}),
    'Ridge': RidgeClassifier(max_iter=2000, random_state=42, class_weight={0:1.0, 1:scale_pos_weight}),
    'DT': DecisionTreeClassifier(random_state=42, max_depth=8, class_weight={0:1.0, 1:scale_pos_weight}),
    'RF': RandomForestClassifier(n_estimators=2000, max_depth=8, random_state=42, class_weight={0:1.0, 1:scale_pos_weight}),
    'XGB': XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=8, scale_pos_weight=scale_pos_weight),
    'LGBM': LGBMClassifier(max_depth=8, n_estimators=2000, learning_rate=0.05, class_weight={0:1.0, 1:scale_pos_weight})
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {'Accuracy': 'accuracy',
           'Precision': 'precision',
           'Recall': 'recall',
           'F1_score': 'f1',
           'pr-auc': 'average_precision',
           'roc-auc': 'roc_auc'}

row = []
for name, model in models.items():
    pipe = Pipeline(steps=[
       ('preprocessing', preprocessor),
       ('ml_model', model)
    ])
    
    cv_scores = cross_validate(pipe, X_train, y_train.values.ravel(), cv=skf, scoring=scoring, n_jobs=-1)
    row.append({
        'Model': name,
        'cv_accuracy': cv_scores['test_Accuracy'].mean(),
        'cv_precision': cv_scores['test_Precision'].mean(),
        'cv_recall': cv_scores['test_Recall'].mean(),
        'cv_f1_score': cv_scores['test_F1_score'].mean(),
        'cv_pr-auc': cv_scores['test_pr-auc'].mean(),
        'cv_roc-auc': cv_scores['test_roc-auc'].mean()       
    })
    
cv_scores_df = pd.DataFrame(row).sort_values(by='cv_recall', ascending=False)
print(cv_scores_df)

best_model_cv = cv_scores_df['cv_recall'].iloc[0]
print(best_model_cv)

params = {
    "ridge__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
    "ridge__solver": ["auto", "saga", "lsqr", "lbfgs"],
    "ridge__tol": loguniform(1e-5, 1e-2),
    "ridge__positive": [True, False]
}


ridge_pipe = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('ridge', RidgeClassifier(random_state=42, class_weight={0:1, 1:scale_pos_weight}, max_iter=2000))
])

random = RandomizedSearchCV(ridge_pipe, params, n_iter=10, cv=skf, scoring='recall', n_jobs=-1)

random.fit(X_train, y_train.values.ravel())

best_score = random.best_score_
best_params = random.best_params_

print(best_score, best_params)

# Remove pipeline prefixes
clean_params = {k.split("__", 1)[-1]: v for k, v in best_params.items()}

# storing the best params in json
with open("employee_attrition_project/data/best_params.json", "w") as f:
    json.dump(clean_params, f)

# loading the best params from json
with open("employee_attrition_project/data/best_params.json", "r") as f:
    best_params = json.load(f)
    
from sklearn.calibration import CalibratedClassifierCV

ridge = RidgeClassifier(random_state=42, 
                        class_weight={0:1.0, 1:scale_pos_weight},
                        **best_params)

calibrated_ridge = CalibratedClassifierCV(
    ridge,
    method='sigmoid',  # Platt scaling
    cv=skf
)

final_ridge_pipe = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('ridge', calibrated_ridge)
])

final_ridge_pipe.fit(X_train, y_train.values.ravel())
proba = final_ridge_pipe.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test, proba)
pr_auc = average_precision_score(y_test, proba)

threshold = 0.20
pred = (proba >= threshold).astype(int)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1_score = f1_score(y_test, pred)

print('roc_auc score for tuned model', roc_auc)
print('pr_auc score for tuned model', pr_auc)
print('accuracy score for tuned model', accuracy)
print('precision score for tuned model', precision)
print('recall score for tuned model', recall)
print('f1_score for tuned model', f1_score)

schema = {
    "features" : X_train.columns.tolist(),
    'dtypes' : X_train.dtypes.astype(str).to_dict()
}

# storing the best params in json
with open("employee_attrition_project/data/schema.json", "w") as f:
    json.dump(schema, f, indent=4)
    
joblib.dump(final_ridge_pipe,'employee_attrition_project/models/current_model.joblib')
    
cm = confusion_matrix(y_test, pred)

# confusion matrix plot
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("employee_attrition_project/confusion_matrix.png")

plt.close()


