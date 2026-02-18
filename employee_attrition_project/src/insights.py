import pandas as pd
import numpy as np
import shap

def generate_feature_insight(importances, top_n = 5):
    top_features = importances.sort_values(by='Importances', ascending=False).head(top_n)
    return top_features

    

    
