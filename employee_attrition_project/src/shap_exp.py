import pandas as pd
from collections import defaultdict

class ShapCollapser:
    def __init__(self, encoded_feature_names, class_index=1):
        """
        Parameters:
        - encoded_feature_names: list of encoded feature names (from encoder.get_feature_names_out or ColumnTransformer)
        - class_index: which class to explain if multi-class (default=1 for positive class)
        """
        self.encoded_feature_names = encoded_feature_names
        self.class_index = class_index
        self.groups = self._group_encoded_features(encoded_feature_names)

    def _group_encoded_features(self, encoded_feature_names):
        """
        Group encoded feature names by their original base feature.
        Handles ColumnTransformer prefixes like 'cat__'.
        """
        groups = defaultdict(list)
        for col in encoded_feature_names:
            # Remove transformer prefix if present
            after_prefix = col.split("__", 1)[1] if "__" in col else col
            # Extract base feature before first underscore
            base = after_prefix.rsplit("_", 1)[1]
            groups[base].append(col)
        return groups

    def collapse(self, shap_values):
        """
        Collapse SHAP values back to original features.
        """
        values = shap_values.values
        if values.ndim == 3:  # (n_samples, n_features, n_classes)
            values = values[:, :, self.class_index]

        shap_df = pd.DataFrame(values, columns=self.encoded_feature_names)

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
