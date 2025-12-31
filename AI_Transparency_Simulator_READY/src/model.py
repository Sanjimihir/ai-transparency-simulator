# src/model.py
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd

def build_preprocessor(X):
    numeric = X.select_dtypes(include=['int64','float64','int','float']).columns.tolist()
    categorical = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]) if len(numeric) > 0 else 'drop'

    # Use sparse_output for scikit-learn >=1.2/1.3 compatibility
    try:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]) if len(categorical) > 0 else 'drop'
    except TypeError:
        # Fallback for older scikit-learn that expects 'sparse' argument
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ]) if len(categorical) > 0 else 'drop'

    transformers = []
    if num_pipe != 'drop':
        transformers.append(('num', num_pipe, numeric))
    if cat_pipe != 'drop':
        transformers.append(('cat', cat_pipe, categorical))

    preproc = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)
    return preproc

def train_model(X_train, y_train, model_type='rf'):
    preproc = build_preprocessor(X_train)
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    pipeline = Pipeline([('preproc', preproc), ('clf', clf)])
    pipeline.fit(X_train, y_train)
    return pipeline

def save_model(pipeline, path='model.joblib'):
    joblib.dump(pipeline, path)

def load_model(path='model.joblib'):
    return joblib.load(path)
