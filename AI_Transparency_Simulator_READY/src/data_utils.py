# src/data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

def load_data(path="data/german_credit.csv"):
    df = pd.read_csv(path)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    # Try to detect target column if not present
    if 'target' not in df.columns:
        # common alternatives
        for alt in ['label','default','y','outcome','target_label']:
            if alt in df.columns:
                df = df.rename(columns={alt:'target'})
                break
    if 'target' not in df.columns:
        raise ValueError("Dataset must contain a target column (try naming it 'target' or 'default' etc).")
    # Map common string labels to 0/1
    if df['target'].dtype == object:
        df['target'] = df['target'].str.lower().map({'yes':1,'y':1,'true':1,'t':1,'approve':1,'approved':1,'good':1,'1':1,
                                                    'no':0,'n':0,'false':0,'f':0,'reject':0,'rejected':0,'bad':0,'0':0}).fillna(df['target'])
    # Attempt to coerce to numeric
    try:
        df['target'] = pd.to_numeric(df['target']).astype(int)
    except Exception:
        pass
    return df

def prepare_data(df, target_col='target', test_size=0.2, random_state=42):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # If y has non-binary values, attempt to binarize (anything >0 -> 1)
    unique_vals = pd.Series(y.unique()).dropna().tolist()
    if len(unique_vals) > 2:
        # try threshold at median
        try:
            y = (pd.to_numeric(y) > pd.to_numeric(y).median()).astype(int)
        except Exception:
            # fallback: keep as-is
            pass

    counts = Counter(y)
    min_count = min(counts.values()) if len(counts)>0 else 0
    if min_count < 2:
        # fallback: non-stratified split
        print(f"Warning: class counts = {dict(counts)}; falling back to non-stratified split.")
        return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=True)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
