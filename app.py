import streamlit as st
import pandas as pd
import numpy as np
import gdown
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Test Probability", layout="wide")

PARQUET_FILE_ID = "1UIAgg0cHBW5TMekpoohpiP23Fd6aeqg8"   # ← replace

@st.cache_data
def load_data():
    gdown.download(f"https://drive.google.com/uc?id={PARQUET_FILE_ID}", "data.parquet", quiet=True)
    return pd.read_parquet("data.parquet")

data = load_data()

# Use only historical fights for training
hist = data[data['Win?'].isin(['Yes','No'])]
# Select three numeric columns (first three available)
num_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c]) and c not in ['FightNumber','Opponent_FightNumber']]
x, y, z = num_cols[0], num_cols[1], num_cols[2]

train = hist[[x, y, z, 'Win?']].dropna()
if len(train) >= 10:
    X = train[[x, y, z]].values
    y_train = (train['Win?'] == 'Yes').astype(int).values
    lr = LogisticRegression(max_iter=1000).fit(X, y_train)
    st.write("Model trained.")

    # Get first upcoming fight
    up = data[data['Win?'].isna() | (data['Win?'] == '')]
    if not up.empty:
        row = up.iloc[0]
        st.write(f"Upcoming fight: {row['Fighter']} vs {row['Opponent']}")
        # Extract features
        try:
            v1 = float(row[x]) if pd.notna(row[x]) else train[x].mean()
            v2 = float(row[y]) if pd.notna(row[y]) else train[y].mean()
            v3 = float(row[z]) if pd.notna(row[z]) else train[z].mean()
        except:
            v1 = train[x].mean()
            v2 = train[y].mean()
            v3 = train[z].mean()
        prob = lr.predict_proba(np.array([[v1, v2, v3]]))[0, 1]
        st.success(f"**LR win probability:** {prob:.1%}")
    else:
        st.warning("No upcoming fights.")
else:
    st.warning("Not enough training data.")
