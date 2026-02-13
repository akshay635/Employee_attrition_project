import pandas as pd
from collections import defaultdict
import importlib
import src.config as config
importlib.reload(config)

class ShapCollapser:
    def __init__(self, encoded_feature_names, class_index=1):
        """
        Parameters:
        - encoded_feature_names: list of encoded feature names (from encoder.get_feature_names_out or ColumnTransformer)
        - class_index: which class to explain if multi-class (default=1 for positive class)
        """
        self.encoded_feature_names = encoded_feature_names
        self.class_index = class_index
        self.groups = self._group_encoded_features(encoded_feature_names, config.COMMON_FEATURES)

    def _group_encoded_features(self, encoded_feature_names, original_features):
        """
        Group encoded feature names by their original base feature.
        Handles ColumnTransformer prefixes like 'cat__'.
        """
        groups = defaultdict(list)
        for col in encoded_feature_names:
            # Step 1: remove transformer prefix if present
            after_prefix = col.split("__", 1)[1] if "__" in col else col
    
            # Step 2: if original_features provided, check direct match
            if original_features and after_prefix in original_features:
                base = after_prefix
            else:
                # Step 3: collapse one-hot by taking everything before last underscore
                base = after_prefix.rsplit("_", 1)[0] if "_" in after_prefix else after_prefix
    
            groups[base].append(col)
        return groups

    def collapse(self, shap_values):
        """
        Collapse SHAP values back to original features.
        """
        values = shap_values.values
        if values.ndim == 3:  # (n_samples, n_features, n_classes)
            zeros = values[:, :, 0]
            ones = values[:, :, 1]

        shap_diff = ones - zeros
        shap_df = pd.DataFrame(shap_diff, columns=self.encoded_feature_names)

        # Collapse each group
        for feature, cols in self.groups.items():
            if len(cols) > 1:  # one-hot encoded → sum contributions
                shap_df[feature] = shap_df[cols].sum(axis=1)
                shap_df.drop(columns=cols, inplace=True)
            else:  # single column → just rename
                shap_df.rename(columns={cols[0]: feature}, inplace=True)

        return shap_df

    def explain(self, shap_values, top_n=5):
        """
        Return a recruiter-friendly narrative of top contributing features.
        """
        shap_df = self.collapse(shap_values)
        mean_abs = shap_df.abs().mean().sort_values(ascending=False)
        top_features = mean_abs.head(top_n)

        narrative = "Top drivers of risk:\n"
        for feat, val in top_features.items():
            narrative += f"- {feat}: contribution {val:.3f}\n"
        return narrative
