# src/counterfactuals.py
import pandas as pd
import numpy as np

def find_greedy_counterfactual(pipeline, X_row, feature_bounds, target_prob=0.5, max_candidates=5):
    orig = X_row.copy()
    try:
        orig_prob = float(pipeline.predict_proba(pd.DataFrame([orig]))[0, 1])
    except Exception:
        orig_prob = 0.0
    if orig_prob >= target_prob:
        return []

    candidates = []
    for feat, (minv, maxv, step) in feature_bounds.items():
        try:
            val0 = float(orig.get(feat, None))
        except Exception:
            continue
        val = val0
        while val <= maxv:
            val = val + step
            test = orig.copy()
            test[feat] = val
            prob = float(pipeline.predict_proba(pd.DataFrame([test]))[0, 1])
            if prob >= target_prob:
                candidates.append({'feature': feat, 'new_value': val, 'prob': prob, 'delta': val - val0})
                break
        val = val0
        while val >= minv:
            val = val - step
            test = orig.copy()
            test[feat] = val
            prob = float(pipeline.predict_proba(pd.DataFrame([test]))[0, 1])
            if prob >= target_prob:
                candidates.append({'feature': feat, 'new_value': val, 'prob': prob, 'delta': val - val0})
                break
    candidates = sorted(candidates, key=lambda x: (abs(x['delta']), -x['prob']))
    return candidates[:max_candidates]
