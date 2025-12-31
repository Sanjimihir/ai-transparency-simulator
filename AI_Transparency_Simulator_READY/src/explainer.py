# src/explainer.py
import shap
import numpy as np
import pandas as pd

class ExplainerWrapper:
    def __init__(self, pipeline, X_train):
        self.pipeline = pipeline
        self.preproc = pipeline.named_steps['preproc']
        self.model = pipeline.named_steps['clf']

        # Build a preprocessed sample and initialize explainer
        try:
            X_pre = self.preproc.transform(X_train)
        except Exception:
            X_pre = self.preproc.transform(X_train.iloc[:min(50, len(X_train))])

        try:
            self.explainer = shap.TreeExplainer(self.model, data=X_pre)
            self.use_tree = True
        except Exception:
            bg = shap.kmeans(X_pre, 50)
            self.explainer = shap.KernelExplainer(self.model.predict_proba, bg)
            self.use_tree = False

        self.feature_names = self._get_feature_names(X_train)

    def _get_feature_names(self, X_sample):
        names = []
        try:
            if hasattr(self.preproc, 'transformers_'):
                for name, transformer, cols in self.preproc.transformers_:
                    if transformer == 'drop' or transformer is None:
                        continue
                    if hasattr(transformer, 'named_steps') and 'scaler' in transformer.named_steps:
                        names.extend(list(cols))
                    elif hasattr(transformer, 'named_steps') and 'ohe' in transformer.named_steps:
                        ohe = transformer.named_steps['ohe']
                        try:
                            ohe_names = list(ohe.get_feature_names_out(cols))
                            names.extend(ohe_names)
                        except Exception:
                            for c in cols:
                                names.append(f"{c}_ohe")
                    else:
                        names.extend(list(cols))
            if len(names) == 0:
                X_pre = self.preproc.transform(X_sample.iloc[:min(50,len(X_sample))])
                n = X_pre.shape[1]
                names = [f"f{i}" for i in range(n)]
        except Exception:
            try:
                X_pre = self.preproc.transform(X_sample.iloc[:min(50,len(X_sample))])
                n = X_pre.shape[1]
                names = [f"f{i}" for i in range(n)]
            except Exception:
                names = [f"f{i}" for i in range(100)]
        return names

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)[:,1]

    def basic_top_features(self, X_row, k=3):
        clf = self.model
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            idx = np.argsort(importances)[-k:][::-1]
            return [(self.feature_names[i], float(importances[i])) for i in idx]
        elif hasattr(clf, 'coef_'):
            coefs = np.abs(clf.coef_).ravel()
            idx = np.argsort(coefs)[-k:][::-1]
            return [(self.feature_names[i], float(coefs[i])) for i in idx]
        else:
            feat_names, shap_vals = self.shap_values(X_row)
            absvals = np.abs(shap_vals).ravel()
            idx = np.argsort(absvals)[-k:][::-1]
            return [(feat_names[i], float(shap_vals.ravel()[i])) for i in idx]

    def shap_values(self, X_row):
        X_pre = self.preproc.transform(X_row)
        sv = self.explainer.shap_values(X_pre)
        if isinstance(sv, (list, tuple)) and len(sv) >= 2:
            vals = np.array(sv[1][0]).ravel()
        else:
            vals = np.array(sv[0]).ravel() if isinstance(sv, (list, tuple)) else np.array(sv).ravel()
        return self.feature_names, vals
