# app.py - Robust AI Transparency Simulator
import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_utils import load_data, prepare_data
from src.model import train_model, save_model, load_model
from src.explainer import ExplainerWrapper
from src.counterfactuals import find_greedy_counterfactual
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

st.set_page_config(layout="wide")
st.title("AI Transparency Simulator — Loan Decision Explainability (Ready)")

DATA_PATH = "data/german_credit.csv"
MODEL_PATH = "model.joblib"

@st.cache_data(show_spinner=False)
def load_and_prepare():
    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_data(df)
    return df, X_train, X_test, y_train, y_test

# Load data
try:
    df, X_train, X_test, y_train, y_test = load_and_prepare()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Train or load model
if os.path.exists(MODEL_PATH):
    pipeline = load_model(MODEL_PATH)
else:
    with st.spinner("Training model..."):
        pipeline = train_model(X_train, y_train, model_type='rf')
        save_model(pipeline, MODEL_PATH)

# Initialize explainer, but guard against very small X_train
try:
    explainer = ExplainerWrapper(pipeline, X_train if len(X_train)>0 else df.drop(columns=['target']))
except Exception as e:
    st.warning(f"Explainer init warning: {e}")
    explainer = None

# Sidebar controls
st.sidebar.header("Controls")
mode = st.sidebar.radio("Explanation mode", ["None", "Basic", "Detailed (SHAP)", "Counterfactuals"])
use_sample = st.sidebar.checkbox("Use sample applicant from test set", value=True)

X_row = None
# Sample selection with safety checks
if use_sample and len(X_test) > 0:
    max_index = max(0, len(X_test)-1)
    if max_index == 0:
        st.sidebar.info("Only one test sample available; using index 0.")
        idx = 0
    else:
        idx = st.sidebar.slider("Sample index", 0, max_index, 0)
    X_row = X_test.iloc[idx]
elif use_sample and len(X_test) == 0:
    st.sidebar.warning("No test samples available; switch to manual entry.")
    use_sample = False

# Manual input fallback
if X_row is None:
    st.sidebar.write("Enter applicant features manually:")
    inputs = {}
    for col in (X_test.columns if len(X_test)>0 else df.drop(columns=['target']).columns):
        if col == 'target':
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            default = float(df[col].median())
            inputs[col] = st.sidebar.number_input(col, value=default)
        else:
            inputs[col] = st.sidebar.text_input(col, value=str(df[col].mode().iloc[0]) if not df[col].mode().empty else "")
    X_row = pd.Series(inputs)

# Ensure X_row is a Series with correct columns
X_df = pd.DataFrame([X_row])[ (df.drop(columns=['target']).columns.tolist()) ]

# Prediction
try:
    prob = float(pipeline.predict_proba(X_df)[0, 1])
except Exception as e:
    st.error(f"Prediction error: {e}")
    st.stop()

decision = "APPROVE" if prob >= 0.5 else "REJECT"
st.metric("Model decision", decision, delta=f"Probability: {prob:.3f}")

st.markdown("---")
col1, col2 = st.columns([2,1])

with col1:
    if mode == "None":
        st.write("No explanation provided (control).")
        trust = st.slider("How much do you trust this decision?", 0.0, 1.0, 0.5, step=0.01, key='trust_none')
    elif mode == "Basic":
        st.write("Basic explanation — top features by importance")
        if explainer:
            top = explainer.basic_top_features(X_df, k=6)
            st.table(pd.DataFrame(top, columns=['feature','importance']))
            fig, ax = plt.subplots(figsize=(6,3))
            df_top = pd.DataFrame(top, columns=['feature','importance']).sort_values('importance')
            ax.barh(df_top['feature'], df_top['importance'])
            st.pyplot(fig)
        else:
            st.write("Explainer not available.")
        trust = st.slider("How much do you trust this decision?", 0.0, 1.0, 0.5, step=0.01, key='trust_basic')
    elif mode == "Detailed (SHAP)":
        st.write("Detailed explanation — SHAP values (top contributors)")
        if explainer:
            feat_names, shap_vals = explainer.shap_values(X_df)
            contrib = pd.DataFrame({'feature': feat_names, 'shap_value': shap_vals}).assign(abs_shap=lambda d: d['shap_value'].abs()).sort_values('abs_shap', ascending=False)
            st.dataframe(contrib.head(20).drop(columns='abs_shap'))
            fig, ax = plt.subplots(figsize=(7,3.5))
            topN = contrib.head(10).sort_values('shap_value')
            ax.barh(topN['feature'], topN['shap_value'])
            st.pyplot(fig)
        else:
            st.write("Explainer not available.")
        trust = st.slider("How much do you trust this decision?", 0.0, 1.0, 0.5, step=0.01, key='trust_shap')
    elif mode == "Counterfactuals":
        st.write("Counterfactual suggestions (simple greedy)")
        # build bounds
        feature_bounds = {}
        for col in X_df.columns:
            try:
                col_min = float(df[col].min())
                col_max = float(df[col].max())
                step = max(float(df[col].std()/4 if pd.api.types.is_numeric_dtype(df[col]) and not pd.isna(df[col].std()) else 1.0), 0.01)
                feature_bounds[col] = (col_min, col_max, step)
            except Exception:
                continue
        try:
            cands = find_greedy_counterfactual(pipeline, X_row, feature_bounds)
            if not cands:
                st.write("No simple single-feature counterfactual found within bounds.")
            else:
                st.table(pd.DataFrame(cands))
        except Exception as e:
            st.write("Counterfactual generation error:", e)
        trust = st.slider("How much do you trust this decision?", 0.0, 1.0, 0.5, step=0.01, key='trust_cf')

with col2:
    st.write("Applicant summary")
    st.dataframe(X_df.transpose())
    st.write("Model test AUC:")
    try:
        test_probs = pipeline.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, test_probs)
        st.write(f"{auc:.3f}")
    except Exception:
        st.write("N/A (not enough test samples or prediction error)")

st.markdown("---")
if st.button("Save interaction"):
    row = {'mode': mode, 'decision': decision, 'prob': prob, 'trust': trust}
    for c in X_df.columns:
        row[f'feat__{c}'] = X_df.iloc[0][c]
    out = "interactions.csv"
    pd.DataFrame([row]).to_csv(out, mode='a', index=False, header=not os.path.exists(out))
    st.success("Saved interaction to interactions.csv")
