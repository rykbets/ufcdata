import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import itertools
import gdown
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cdist

st.set_page_config(page_title="UFC Pre‑Fight Dashboard", layout="wide")

# ============================================================
# Load data from Google Drive using file ID
# ============================================================
PARQUET_FILE_ID = "1uIpfbGFmDolA8P2vc15VvA1qbNzWetxf"   # <--- REPLACE THIS

@st.cache_data
def load_data():
    gdown.download(f"https://drive.google.com/uc?id={PARQUET_FILE_ID}", "data.parquet", quiet=True)
    return pd.read_parquet("data.parquet")

data = load_data()
original_data = data.copy()

# ---------- Helper functions ----------
def normalize_title_col(series):
    if series is None:
        return pd.Series('', index=series.index)
    return series.astype(str).str.strip().str.lower()

def get_diff_range(df, col_name):
    if col_name not in df.columns:
        return -1.0, 1.0
    vals = df[col_name].dropna()
    if len(vals) == 0:
        return -1.0, 1.0
    return float(vals.min()), float(vals.max())

def get_first_col(df, col_name):
    if col_name not in df.columns:
        return np.full(len(df), np.nan)
    sub = df[col_name]
    if isinstance(sub, pd.DataFrame):
        return sub.iloc[:, 0].to_numpy(dtype=np.float64, na_value=np.nan)
    return pd.to_numeric(sub, errors='coerce').to_numpy(dtype=np.float64)

# ---------- Define features ----------
adjperf_diff_cols = [c for c in data.columns if c.endswith('_diff') and c.startswith('adjperf_')]
base_cols = ['Age', 'AgeDiff', 'HeightDiff', 'ReachDiff',
             'DaysSincePrev', 'DaysSincePrev_diff', 'Avg3DaysGap_diff',
             'FightNumber', 'FightNumber_diff',
             'FighterOddsNum', 'PrevFighterOddsNum',
             'CareerWinPct_diff', 'Prev7WinPct',
             'FighterColleyDecay', 'OpponentColleyDecay', 'ColleyDecayDiff',
             'FighterMasseyDecay', 'OpponentMasseyDecay', 'MasseyDecayDiff',
             'FighterWeightedMasseyDecay', 'OpponentWeightedMasseyDecay', 'WeightedMasseyDecayDiff']

new_features = []
for col in base_cols:
    if col in data.columns:
        new_features.append(col)
for col in adjperf_diff_cols:
    if col in data.columns:
        new_features.append(col)
# Remove duplicates
seen = set()
new_features_unique = []
for col in new_features:
    if col not in seen:
        seen.add(col)
        new_features_unique.append(col)
new_features = new_features_unique

three_d_features = [c for c in new_features if data[c].nunique(dropna=True) >= 2 and np.issubdtype(data[c].dtype, np.number)]

# For combo builders, exclude rating/win‑pct columns to avoid leakage
exclude_combo = [
    'CareerWinPct_diff', 'Prev7WinPct',
    'FighterColleyDecay', 'OpponentColleyDecay', 'ColleyDecayDiff',
    'FighterMasseyDecay', 'OpponentMasseyDecay', 'MasseyDecayDiff',
    'FighterWeightedMasseyDecay', 'OpponentWeightedMasseyDecay', 'WeightedMasseyDecayDiff'
]
combo_candidates = [c for c in new_features if c not in exclude_combo]

# =========================================================================
# INITIALIZE SESSION STATE
# =========================================================================
if 'lr_model' not in st.session_state:
    st.session_state.lr_model = None
if 'calibrated_knn' not in st.session_state:
    st.session_state.calibrated_knn = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train_knn' not in st.session_state:
    st.session_state.y_train_knn = None
if 'overall_wr' not in st.session_state:
    st.session_state.overall_wr = 0.0
if 'recent_wr' not in st.session_state:
    st.session_state.recent_wr = 0.0
if 'recent_count' not in st.session_state:
    st.session_state.recent_count = 0
if 'lr_train_status' not in st.session_state:
    st.session_state.lr_train_status = "Not trained"
if 'knn_train_status' not in st.session_state:
    st.session_state.knn_train_status = "Not trained"
if 'selected_fight_row' not in st.session_state:
    st.session_state.selected_fight_row = None
if 'lr_feature_names' not in st.session_state:
    st.session_state.lr_feature_names = []
if 'knn_feature_names' not in st.session_state:
    st.session_state.knn_feature_names = []
if 'x_lr' not in st.session_state:
    st.session_state.x_lr = None
if 'y_lr' not in st.session_state:
    st.session_state.y_lr = None
if 'z_lr' not in st.session_state:
    st.session_state.z_lr = None
if 'x_knn' not in st.session_state:
    st.session_state.x_knn = None
if 'y_knn' not in st.session_state:
    st.session_state.y_knn = None
if 'z_knn' not in st.session_state:
    st.session_state.z_knn = None
if 'knn_model_k' not in st.session_state:
    st.session_state.knn_model_k = 5
if 'lr_combo_results' not in st.session_state:
    st.session_state.lr_combo_results = None
if 'knn_combo_results' not in st.session_state:
    st.session_state.knn_combo_results = None

# Set default features if three_d_features exists
if len(three_d_features) >= 3:
    if st.session_state.x_lr is None:
        st.session_state.x_lr = three_d_features[0]
    if st.session_state.y_lr is None:
        st.session_state.y_lr = three_d_features[1]
    if st.session_state.z_lr is None:
        st.session_state.z_lr = three_d_features[2]
    if st.session_state.x_knn is None:
        st.session_state.x_knn = three_d_features[0]
    if st.session_state.y_knn is None:
        st.session_state.y_knn = three_d_features[1]
    if st.session_state.z_knn is None:
        st.session_state.z_knn = three_d_features[2]

# =========================================================================
# Sidebar Filters
# =========================================================================
st.sidebar.title("Filters")

# ---- General ----
with st.sidebar.expander("General", expanded=True):
    wc = st.multiselect("Weight Class", sorted(data['WC'].dropna().unique()), key="filter_wc") if 'WC' in data.columns else []
    stance = st.multiselect("Stance", sorted(data['Stance'].dropna().unique()), key="filter_stance") if 'Stance' in data.columns else []
    country = st.multiselect("Country", sorted(data['Country'].dropna().unique()), key="filter_country") if 'Country' in data.columns else []
    sched_rounds = st.multiselect("Scheduled Rounds", sorted(data['ScheduledRounds'].dropna().unique()), key="filter_sched") if 'ScheduledRounds' in data.columns else []
    title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"], key="filter_titlefight") if 'Title' in data.columns else "All"
    hometown = st.selectbox("Hometown", ["All", "Yes", "No"], key="filter_hometown") if 'HometownFighter' in data.columns else "All"
    opp_hometown = st.selectbox("Opp Hometown", ["All", "Yes", "No"], key="filter_opphometown") if 'Opponent_Hometown' in data.columns else "All"
    event_country = st.multiselect("Event Country", sorted(data['EventCountry'].dropna().unique()), key="filter_event") if 'EventCountry' in data.columns else []

# ---- Fight Numbers ----
with st.sidebar.expander("Fight Numbers", expanded=False):
    fn_min = st.number_input("Min Fight #", value=1, min_value=1, max_value=int(data['FightNumber'].max()), key="filter_fn_min") if 'FightNumber' in data.columns else 1
    fn_max = st.number_input("Max Fight #", value=int(data['FightNumber'].max()), key="filter_fn_max") if 'FightNumber' in data.columns else 1000
    ofn_min = st.number_input("Opp Min Fight #", value=1, key="filter_ofn_min") if 'Opponent_FightNumber' in data.columns else 1
    ofn_max = st.number_input("Opp Max Fight #", value=int(data['Opponent_FightNumber'].max()), key="filter_ofn_max") if 'Opponent_FightNumber' in data.columns else 1000

# ---- Career Win % Diff ----
with st.sidebar.expander("Career Win % Diff", expanded=False):
    if 'CareerWinPct_diff' in data.columns:
        cwp_min = st.slider("Min Career Win % Diff", -100, 100, -100, step=5, key="filter_cwp_min")
        cwp_max = st.slider("Max Career Win % Diff", -100, 100, 100, step=5, key="filter_cwp_max")
    else:
        cwp_min, cwp_max = -100, 100

# ---- Physical Attributes ----
with st.sidebar.expander("Physical Attributes", expanded=False):
    age = st.slider("Age", int(data['Age'].min()), int(data['Age'].max()), (int(data['Age'].min()), int(data['Age'].max())), key="filter_age") if 'Age' in data.columns else (0,100)
    age_diff = st.slider("Age Diff", int(data['AgeDiff'].min()), int(data['AgeDiff'].max()), (int(data['AgeDiff'].min()), int(data['AgeDiff'].max())), key="filter_age_diff") if 'AgeDiff' in data.columns else (-100,100)
    height_diff = st.slider("Height Diff (in)", int(data['HeightDiff'].min()), int(data['HeightDiff'].max()), (int(data['HeightDiff'].min()), int(data['HeightDiff'].max())), key="filter_height_diff") if 'HeightDiff' in data.columns else (-50,50)
    reach_diff = st.slider("Reach Diff (in)", int(data['ReachDiff'].min()), int(data['ReachDiff'].max()), (int(data['ReachDiff'].min()), int(data['ReachDiff'].max())), key="filter_reach_diff") if 'ReachDiff' in data.columns else (-50,50)

# ---- Days & Gaps ----
with st.sidebar.expander("Days & Gaps", expanded=False):
    days = st.slider("Days Since Prev", int(data['DaysSincePrev'].min()), int(data['DaysSincePrev'].max()), (int(data['DaysSincePrev'].min()), int(data['DaysSincePrev'].max())), key="filter_days") if 'DaysSincePrev' in data.columns else (0,1000)
    days_diff = st.slider("Days Since Prev Diff", int(data['DaysSincePrev_diff'].min()), int(data['DaysSincePrev_diff'].max()), (int(data['DaysSincePrev_diff'].min()), int(data['DaysSincePrev_diff'].max())), key="filter_days_diff") if 'DaysSincePrev_diff' in data.columns else (-1000,1000)
    avg3_diff = st.slider("Avg3DaysGap Diff", int(data['Avg3DaysGap_diff'].min()), int(data['Avg3DaysGap_diff'].max()), (int(data['Avg3DaysGap_diff'].min()), int(data['Avg3DaysGap_diff'].max())), key="filter_avg3_diff") if 'Avg3DaysGap_diff' in data.columns else (-1000,1000)

# ---- Odds ----
with st.sidebar.expander("Odds", expanded=False):
    cur_odds = st.slider("Fighter Odds", int(data['FighterOddsNum'].min()), int(data['FighterOddsNum'].max()), (int(data['FighterOddsNum'].min()), int(data['FighterOddsNum'].max())), step=10, key="filter_cur_odds") if 'FighterOddsNum' in data.columns else (-1000,1000)
    prev_odds = st.slider("Prev Fighter Odds", int(data['PrevFighterOddsNum'].min()), int(data['PrevFighterOddsNum'].max()), (int(data['PrevFighterOddsNum'].min()), int(data['PrevFighterOddsNum'].max())), step=10, key="filter_prev_odds") if 'PrevFighterOddsNum' in data.columns else (-1000,1000)

# ---- Previous Outcomes ----
skip_nc = st.sidebar.checkbox("Skip NC outcomes", key="filter_skip_nc")
if skip_nc:
    prev1_col = 'Prev1_Outcome_skipNC'
    prev2_col = 'Prev2_Outcome_skipNC'
    prev3_col = 'Prev3_Outcome_skipNC'
    career1_col = 'Career1_Outcome_skipNC'
    career2_col = 'Career2_Outcome_skipNC'
    career3_col = 'Career3_Outcome_skipNC'
    opp_career1_col = 'Opponent_Career1_Outcome_skipNC'
    opp_career2_col = 'Opponent_Career2_Outcome_skipNC'
    opp_career3_col = 'Opponent_Career3_Outcome_skipNC'
else:
    prev1_col = 'Prev1_Outcome_raw'
    prev2_col = 'Prev2_Outcome_raw'
    prev3_col = 'Prev3_Outcome_raw'
    career1_col = 'Career1_Outcome_raw'
    career2_col = 'Career2_Outcome_raw'
    career3_col = 'Career3_Outcome_raw'
    opp_career1_col = 'Opponent_Career1_Outcome_raw'
    opp_career2_col = 'Opponent_Career2_Outcome_raw'
    opp_career3_col = 'Opponent_Career3_Outcome_raw'

all_outcomes_raw = sorted(data[prev1_col].dropna().unique()) if prev1_col in data.columns else []
all_outcomes_career = sorted(data[career1_col].dropna().unique()) if career1_col in data.columns else []

with st.sidebar.expander("Previous Outcomes", expanded=False):
    prev1 = st.multiselect("Prev Fight 1", all_outcomes_raw, key="filter_prev1")
    prev2 = st.multiselect("Prev Fight 2", all_outcomes_raw, key="filter_prev2")
    prev3 = st.multiselect("Prev Fight 3", all_outcomes_raw, key="filter_prev3")
    opp_prev1 = st.multiselect("Opp Prev 1", all_outcomes_raw, key="filter_opp_prev1")
    opp_prev2 = st.multiselect("Opp Prev 2", all_outcomes_raw, key="filter_opp_prev2")
    opp_prev3 = st.multiselect("Opp Prev 3", all_outcomes_raw, key="filter_opp_prev3")
    career1 = st.multiselect("Career F1", all_outcomes_career, key="filter_career1")
    career2 = st.multiselect("Career F2", all_outcomes_career, key="filter_career2")
    career3 = st.multiselect("Career F3", all_outcomes_career, key="filter_career3")
    opp_career1 = st.multiselect("Opp Career F1", all_outcomes_career, key="filter_opp_career1")
    opp_career2 = st.multiselect("Opp Career F2", all_outcomes_career, key="filter_opp_career2")
    opp_career3 = st.multiselect("Opp Career F3", all_outcomes_career, key="filter_opp_career3")

# ---- Ratings ----
with st.sidebar.expander("Ratings", expanded=False):
    use_colley = st.checkbox("Filter ColleyDecayDiff", value=False, key="filter_use_colley")
    if use_colley:
        min_cd, max_cd = get_diff_range(data, 'ColleyDecayDiff')
        colley_range = st.slider("ColleyDecayDiff range", min_cd, max_cd, (min_cd, max_cd), step=0.01, key="filter_colley")
    use_massey = st.checkbox("Filter MasseyDecayDiff", value=False, key="filter_use_massey")
    if use_massey:
        min_md, max_md = get_diff_range(data, 'MasseyDecayDiff')
        massey_range = st.slider("MasseyDecayDiff range", min_md, max_md, (min_md, max_md), step=0.01, key="filter_massey")
    use_wmd = st.checkbox("Filter WeightedMasseyDecayDiff", value=False, key="filter_use_wmd")
    if use_wmd:
        min_wmd, max_wmd = get_diff_range(data, 'WeightedMasseyDecayDiff')
        wmd_range = st.slider("WeightedMasseyDecayDiff range", min_wmd, max_wmd, (min_wmd, max_wmd), step=0.01, key="filter_wmd")

prev_title = st.sidebar.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"], key="filter_prev_title")
opp_prev_title = st.sidebar.selectbox("Opp Prev Fight Was Title?", ["All", "Yes", "No"], key="filter_opp_prev_title")
new_wc = st.sidebar.checkbox("New Weight Class", key="filter_new_wc") if 'IsNewWeightClass' in data.columns else False

# =========================================================================
# APPLY FILTERS (with robust checks – ALL filters shown in debug)
# =========================================================================
filtered = data.copy()
debug_info = []  # to collect counts

def filter_step(df, label, condition):
    if condition is not None:
        before = len(df)
        df = df[condition]
        debug_info.append(f"{label}: {before} → {len(df)}")
    else:
        # Show that the filter was skipped (optional)
        debug_info.append(f"{label}: SKIPPED (condition missing)")
    return df

# 1. General – pull from session state to ensure values are captured
wc = st.session_state.get('filter_wc', [])
stance = st.session_state.get('filter_stance', [])
country = st.session_state.get('filter_country', [])
sched_rounds = st.session_state.get('filter_sched', [])
title_fight = st.session_state.get('filter_titlefight', "All")
hometown = st.session_state.get('filter_hometown', "All")
opp_hometown = st.session_state.get('filter_opphometown', "All")
event_country = st.session_state.get('filter_event', [])
new_wc = st.session_state.get('filter_new_wc', False)

# Now apply each filter and log it
if wc and 'WC' in filtered.columns:
    filtered = filter_step(filtered, "Weight Class", filtered['WC'].isin(wc))
else:
    debug_info.append("Weight Class: SKIPPED (no selection)")
if stance and 'Stance' in filtered.columns:
    filtered = filter_step(filtered, "Stance", filtered['Stance'].isin(stance))
else:
    debug_info.append("Stance: SKIPPED (no selection)")
if country and 'Country' in filtered.columns:
    filtered = filter_step(filtered, "Country", filtered['Country'].isin(country))
else:
    debug_info.append("Country: SKIPPED (no selection)")
if sched_rounds and 'ScheduledRounds' in filtered.columns:
    filtered = filter_step(filtered, "Sched Rounds", filtered['ScheduledRounds'].isin(sched_rounds))
else:
    debug_info.append("Sched Rounds: SKIPPED (no selection)")
if title_fight != "All" and 'Title' in filtered.columns:
    filtered = filter_step(filtered, "Title Fight", filtered['Title'] == title_fight)
else:
    debug_info.append("Title Fight: SKIPPED (set to All or column missing)")
if hometown != "All" and 'HometownFighter' in filtered.columns:
    filtered = filter_step(filtered, "Hometown", filtered['HometownFighter'] == hometown)
else:
    debug_info.append("Hometown: SKIPPED (set to All or column missing)")
if opp_hometown != "All" and 'Opponent_Hometown' in filtered.columns:
    filtered = filter_step(filtered, "Opp Hometown", filtered['Opponent_Hometown'] == opp_hometown)
else:
    debug_info.append("Opp Hometown: SKIPPED (set to All or column missing)")
if event_country and 'EventCountry' in filtered.columns:
    filtered = filter_step(filtered, "Event Country", filtered['EventCountry'].isin(event_country))
else:
    debug_info.append("Event Country: SKIPPED (no selection)")
if new_wc and 'IsNewWeightClass' in filtered.columns:
    filtered = filter_step(filtered, "New Weight Class", filtered['IsNewWeightClass'] == True)
else:
    debug_info.append("New Weight Class: SKIPPED (not checked or column missing)")

# 2. Title filters (Prev Title)
if 'Prev1_Title' in filtered.columns:
    filtered['Prev1_Title_clean'] = normalize_title_col(filtered['Prev1_Title'])
    prev_title = st.session_state.get('filter_prev_title', "All")
    if prev_title != "All":
        filtered = filter_step(filtered, "Prev Title", filtered['Prev1_Title_clean'] == prev_title.lower())
    else:
        debug_info.append("Prev Title: SKIPPED (set to All)")
if 'Opponent_Prev1_Title' in filtered.columns:
    filtered['Opp_Prev1_Title_clean'] = normalize_title_col(filtered['Opponent_Prev1_Title'])
    opp_prev_title = st.session_state.get('filter_opp_prev_title', "All")
    if opp_prev_title != "All":
        filtered = filter_step(filtered, "Opp Prev Title", filtered['Opp_Prev1_Title_clean'] == opp_prev_title.lower())
    else:
        debug_info.append("Opp Prev Title: SKIPPED (set to All)")

# 3. Fight numbers
fn_min = st.session_state.get('filter_fn_min', 1)
fn_max = st.session_state.get('filter_fn_max', 1000)
ofn_min = st.session_state.get('filter_ofn_min', 1)
ofn_max = st.session_state.get('filter_ofn_max', 1000)

if 'FightNumber' in filtered.columns:
    filtered = filter_step(filtered, "Fight #", (filtered['FightNumber'] >= fn_min) & (filtered['FightNumber'] <= fn_max))
if 'Opponent_FightNumber' in filtered.columns:
    filtered = filter_step(filtered, "Opp Fight #", (filtered['Opponent_FightNumber'] >= ofn_min) & (filtered['Opponent_FightNumber'] <= ofn_max))

# 4. Physical, days, odds (pull from session state)
age = st.session_state.get('filter_age', (0, 100))
age_diff = st.session_state.get('filter_age_diff', (-100, 100))
height_diff = st.session_state.get('filter_height_diff', (-50, 50))
reach_diff = st.session_state.get('filter_reach_diff', (-50, 50))
days = st.session_state.get('filter_days', (0, 1000))
days_diff = st.session_state.get('filter_days_diff', (-1000, 1000))
avg3_diff = st.session_state.get('filter_avg3_diff', (-1000, 1000))
cwp_min = st.session_state.get('filter_cwp_min', -100)
cwp_max = st.session_state.get('filter_cwp_max', 100)
cur_odds = st.session_state.get('filter_cur_odds', (-1000, 1000))
prev_odds = st.session_state.get('filter_prev_odds', (-1000, 1000))

if 'Age' in filtered.columns:
    filtered = filter_step(filtered, "Age", (filtered['Age'] >= age[0]) & (filtered['Age'] <= age[1]))
if 'AgeDiff' in filtered.columns:
    filtered = filter_step(filtered, "Age Diff", (filtered['AgeDiff'] >= age_diff[0]) & (filtered['AgeDiff'] <= age_diff[1]))
if 'HeightDiff' in filtered.columns:
    filtered = filter_step(filtered, "Height Diff", (filtered['HeightDiff'] >= height_diff[0]) & (filtered['HeightDiff'] <= height_diff[1]))
if 'ReachDiff' in filtered.columns:
    filtered = filter_step(filtered, "Reach Diff", (filtered['ReachDiff'] >= reach_diff[0]) & (filtered['ReachDiff'] <= reach_diff[1]))
if 'DaysSincePrev' in filtered.columns:
    filtered = filter_step(filtered, "Days Since Prev", (filtered['DaysSincePrev'] >= days[0]) & (filtered['DaysSincePrev'] <= days[1]))
if 'DaysSincePrev_diff' in filtered.columns:
    filtered = filter_step(filtered, "Days Since Prev Diff", (filtered['DaysSincePrev_diff'] >= days_diff[0]) & (filtered['DaysSincePrev_diff'] <= days_diff[1]))
if 'Avg3DaysGap_diff' in filtered.columns:
    filtered = filter_step(filtered, "Avg3 Gap Diff", (filtered['Avg3DaysGap_diff'] >= avg3_diff[0]) & (filtered['Avg3DaysGap_diff'] <= avg3_diff[1]))
if 'CareerWinPct_diff' in filtered.columns and filtered['CareerWinPct_diff'].notna().any():
    filtered = filter_step(filtered, "Career Win% Diff", (filtered['CareerWinPct_diff'] >= cwp_min) & (filtered['CareerWinPct_diff'] <= cwp_max))
if 'FighterOddsNum' in filtered.columns and filtered['FighterOddsNum'].notna().any():
    filtered = filter_step(filtered, "Fighter Odds", (filtered['FighterOddsNum'] >= cur_odds[0]) & (filtered['FighterOddsNum'] <= cur_odds[1]))
if 'PrevFighterOddsNum' in filtered.columns and filtered['PrevFighterOddsNum'].notna().any():
    filtered = filter_step(filtered, "Prev Fighter Odds", (filtered['PrevFighterOddsNum'] >= prev_odds[0]) & (filtered['PrevFighterOddsNum'] <= prev_odds[1]))

# 5. Previous outcomes (fighter) – pull from session state
prev1 = st.session_state.get('filter_prev1', [])
prev2 = st.session_state.get('filter_prev2', [])
prev3 = st.session_state.get('filter_prev3', [])
career1 = st.session_state.get('filter_career1', [])
career2 = st.session_state.get('filter_career2', [])
career3 = st.session_state.get('filter_career3', [])

if prev1 and prev1_col in filtered.columns:
    filtered = filter_step(filtered, "Prev1 Outcome", filtered[prev1_col].isin(prev1))
if prev2 and prev2_col in filtered.columns:
    filtered = filter_step(filtered, "Prev2 Outcome", filtered[prev2_col].isin(prev2))
if prev3 and prev3_col in filtered.columns:
    filtered = filter_step(filtered, "Prev3 Outcome", filtered[prev3_col].isin(prev3))
if career1 and career1_col in filtered.columns:
    filtered = filter_step(filtered, "Career1 Outcome", filtered[career1_col].isin(career1))
if career2 and career2_col in filtered.columns:
    filtered = filter_step(filtered, "Career2 Outcome", filtered[career2_col].isin(career2))
if career3 and career3_col in filtered.columns:
    filtered = filter_step(filtered, "Career3 Outcome", filtered[career3_col].isin(career3))

# 6. Opponent previous outcomes (shifted)
opp_prev1 = st.session_state.get('filter_opp_prev1', [])
opp_prev2 = st.session_state.get('filter_opp_prev2', [])
opp_prev3 = st.session_state.get('filter_opp_prev3', [])

for shift, wlist in [(1, opp_prev1), (2, opp_prev2), (3, opp_prev3)]:
    col = f'Opponent_Prev{shift}_Outcome_raw'
    if col in filtered.columns and wlist:
        if skip_nc:
            col_use = f'Opponent_Prev{shift}_Outcome_skipNC'
            if col_use in filtered.columns:
                filtered = filter_step(filtered, f"Opp Prev{shift} (skipNC)", filtered[col_use].isin(wlist))
        else:
            filtered = filter_step(filtered, f"Opp Prev{shift} (raw)", filtered[col].isin(wlist))

# 7. Opponent career outcomes
opp_career1 = st.session_state.get('filter_opp_career1', [])
opp_career2 = st.session_state.get('filter_opp_career2', [])
opp_career3 = st.session_state.get('filter_opp_career3', [])

if opp_career1 and opp_career1_col in filtered.columns:
    filtered = filter_step(filtered, "Opp Career1", filtered[opp_career1_col].isin(opp_career1))
if opp_career2 and opp_career2_col in filtered.columns:
    filtered = filter_step(filtered, "Opp Career2", filtered[opp_career2_col].isin(opp_career2))
if opp_career3 and opp_career3_col in filtered.columns:
    filtered = filter_step(filtered, "Opp Career3", filtered[opp_career3_col].isin(opp_career3))

# 8. Ratings filters
use_colley = st.session_state.get('filter_use_colley', False)
use_massey = st.session_state.get('filter_use_massey', False)
use_wmd = st.session_state.get('filter_use_wmd', False)

if use_colley and 'ColleyDecayDiff' in filtered.columns and filtered['ColleyDecayDiff'].notna().any():
    colley_range = st.session_state.get('filter_colley', (0, 1))
    filtered = filter_step(filtered, "Colley Decay Diff", (filtered['ColleyDecayDiff'] >= colley_range[0]) & (filtered['ColleyDecayDiff'] <= colley_range[1]))
if use_massey and 'MasseyDecayDiff' in filtered.columns and filtered['MasseyDecayDiff'].notna().any():
    massey_range = st.session_state.get('filter_massey', (0, 1))
    filtered = filter_step(filtered, "Massey Decay Diff", (filtered['MasseyDecayDiff'] >= massey_range[0]) & (filtered['MasseyDecayDiff'] <= massey_range[1]))
if use_wmd and 'WeightedMasseyDecayDiff' in filtered.columns and filtered['WeightedMasseyDecayDiff'].notna().any():
    wmd_range = st.session_state.get('filter_wmd', (0, 1))
    filtered = filter_step(filtered, "Weighted Massey Decay Diff", (filtered['WeightedMasseyDecayDiff'] >= wmd_range[0]) & (filtered['WeightedMasseyDecayDiff'] <= wmd_range[1]))

# Show debug info in a collapsible expander
with st.sidebar.expander("🔍 Filter Debug (collapsible)", expanded=False):
    st.write("Initial rows:", len(original_data))
    st.write("Rows after all filters:", len(filtered))
    for line in debug_info:
        st.write(line)
    if len(debug_info) == 0:
        st.write("No filters applied.")

# Overwrite data
data = filtered

# 2. Title filters (Prev Title)
if 'Prev1_Title' in filtered.columns:
    filtered['Prev1_Title_clean'] = normalize_title_col(filtered['Prev1_Title'])
    if prev_title != "All":
        filtered = filter_step(filtered, "Prev Title", filtered['Prev1_Title_clean'] == prev_title.lower())
if 'Opponent_Prev1_Title' in filtered.columns:
    filtered['Opp_Prev1_Title_clean'] = normalize_title_col(filtered['Opponent_Prev1_Title'])
    if opp_prev_title != "All":
        filtered = filter_step(filtered, "Opp Prev Title", filtered['Opp_Prev1_Title_clean'] == opp_prev_title.lower())

# 3. Fight numbers
if 'FightNumber' in filtered.columns:
    filtered = filter_step(filtered, "Fight #", (filtered['FightNumber'] >= fn_min) & (filtered['FightNumber'] <= fn_max))
if 'Opponent_FightNumber' in filtered.columns:
    filtered = filter_step(filtered, "Opp Fight #", (filtered['Opponent_FightNumber'] >= ofn_min) & (filtered['Opponent_FightNumber'] <= ofn_max))

# 4. Physical, days, odds
if 'Age' in filtered.columns:
    filtered = filter_step(filtered, "Age", (filtered['Age'] >= age[0]) & (filtered['Age'] <= age[1]))
if 'AgeDiff' in filtered.columns:
    filtered = filter_step(filtered, "Age Diff", (filtered['AgeDiff'] >= age_diff[0]) & (filtered['AgeDiff'] <= age_diff[1]))
if 'HeightDiff' in filtered.columns:
    filtered = filter_step(filtered, "Height Diff", (filtered['HeightDiff'] >= height_diff[0]) & (filtered['HeightDiff'] <= height_diff[1]))
if 'ReachDiff' in filtered.columns:
    filtered = filter_step(filtered, "Reach Diff", (filtered['ReachDiff'] >= reach_diff[0]) & (filtered['ReachDiff'] <= reach_diff[1]))
if 'DaysSincePrev' in filtered.columns:
    filtered = filter_step(filtered, "Days Since Prev", (filtered['DaysSincePrev'] >= days[0]) & (filtered['DaysSincePrev'] <= days[1]))
if 'DaysSincePrev_diff' in filtered.columns:
    filtered = filter_step(filtered, "Days Since Prev Diff", (filtered['DaysSincePrev_diff'] >= days_diff[0]) & (filtered['DaysSincePrev_diff'] <= days_diff[1]))
if 'Avg3DaysGap_diff' in filtered.columns:
    filtered = filter_step(filtered, "Avg3 Gap Diff", (filtered['Avg3DaysGap_diff'] >= avg3_diff[0]) & (filtered['Avg3DaysGap_diff'] <= avg3_diff[1]))
if 'CareerWinPct_diff' in filtered.columns:
    if filtered['CareerWinPct_diff'].notna().any():
        filtered = filter_step(filtered, "Career Win% Diff", (filtered['CareerWinPct_diff'] >= cwp_min) & (filtered['CareerWinPct_diff'] <= cwp_max))
if 'FighterOddsNum' in filtered.columns:
    if filtered['FighterOddsNum'].notna().any():
        filtered = filter_step(filtered, "Fighter Odds", (filtered['FighterOddsNum'] >= cur_odds[0]) & (filtered['FighterOddsNum'] <= cur_odds[1]))
if 'PrevFighterOddsNum' in filtered.columns:
    if filtered['PrevFighterOddsNum'].notna().any():
        filtered = filter_step(filtered, "Prev Fighter Odds", (filtered['PrevFighterOddsNum'] >= prev_odds[0]) & (filtered['PrevFighterOddsNum'] <= prev_odds[1]))

# 5. Previous outcomes (fighter)
if prev1 and prev1_col in filtered.columns:
    filtered = filter_step(filtered, "Prev1 Outcome", filtered[prev1_col].isin(prev1))
if prev2 and prev2_col in filtered.columns:
    filtered = filter_step(filtered, "Prev2 Outcome", filtered[prev2_col].isin(prev2))
if prev3 and prev3_col in filtered.columns:
    filtered = filter_step(filtered, "Prev3 Outcome", filtered[prev3_col].isin(prev3))
if career1 and career1_col in filtered.columns:
    filtered = filter_step(filtered, "Career1 Outcome", filtered[career1_col].isin(career1))
if career2 and career2_col in filtered.columns:
    filtered = filter_step(filtered, "Career2 Outcome", filtered[career2_col].isin(career2))
if career3 and career3_col in filtered.columns:
    filtered = filter_step(filtered, "Career3 Outcome", filtered[career3_col].isin(career3))

# 6. Opponent previous outcomes (shifted)
for shift, wlist in [(1, opp_prev1), (2, opp_prev2), (3, opp_prev3)]:
    col = f'Opponent_Prev{shift}_Outcome_raw'
    if col in filtered.columns and wlist:
        if skip_nc:
            col_use = f'Opponent_Prev{shift}_Outcome_skipNC'
            if col_use in filtered.columns:
                filtered = filter_step(filtered, f"Opp Prev{shift} (skipNC)", filtered[col_use].isin(wlist))
        else:
            filtered = filter_step(filtered, f"Opp Prev{shift} (raw)", filtered[col].isin(wlist))

# 7. Opponent career outcomes
if opp_career1 and opp_career1_col in filtered.columns:
    filtered = filter_step(filtered, "Opp Career1", filtered[opp_career1_col].isin(opp_career1))
if opp_career2 and opp_career2_col in filtered.columns:
    filtered = filter_step(filtered, "Opp Career2", filtered[opp_career2_col].isin(opp_career2))
if opp_career3 and opp_career3_col in filtered.columns:
    filtered = filter_step(filtered, "Opp Career3", filtered[opp_career3_col].isin(opp_career3))

# 8. Ratings filters
if use_colley and 'ColleyDecayDiff' in filtered.columns and filtered['ColleyDecayDiff'].notna().any():
    filtered = filter_step(filtered, "Colley Decay Diff", (filtered['ColleyDecayDiff'] >= colley_range[0]) & (filtered['ColleyDecayDiff'] <= colley_range[1]))
if use_massey and 'MasseyDecayDiff' in filtered.columns and filtered['MasseyDecayDiff'].notna().any():
    filtered = filter_step(filtered, "Massey Decay Diff", (filtered['MasseyDecayDiff'] >= massey_range[0]) & (filtered['MasseyDecayDiff'] <= massey_range[1]))
if use_wmd and 'WeightedMasseyDecayDiff' in filtered.columns and filtered['WeightedMasseyDecayDiff'].notna().any():
    filtered = filter_step(filtered, "Weighted Massey Decay Diff", (filtered['WeightedMasseyDecayDiff'] >= wmd_range[0]) & (filtered['WeightedMasseyDecayDiff'] <= wmd_range[1]))

# Show debug info in a collapsible expander
with st.sidebar.expander("🔍 Filter Debug (collapsible)", expanded=False):
    st.write("Initial rows:", len(original_data))
    st.write("Rows after all filters:", len(filtered))
    for line in debug_info:
        st.write(line)
    if len(debug_info) == 0:
        st.write("No filters applied.")

# Overwrite data
data = filtered

# For matchup display, we need both rows from original_data for surviving FightIDs
surviving_fight_ids = data['FightID'].unique()
matchup_data = original_data[original_data['FightID'].isin(surviving_fight_ids)]

# =========================================================================
# COMMON DEFINITIONS
# =========================================================================
def detailed_result(row):
    win_raw = row.get('Win?')
    if win_raw is None or pd.isna(win_raw) or str(win_raw).strip().lower() in ('', 'none', 'nan'):
        return 'Upcoming'
    win_val = str(win_raw).strip()
    method = str(row.get('Method', '')).strip().lower()
    if 'dq' in method or 'disqualif' in method:
        return 'Win by DQ' if win_val == 'Yes' else 'Loss by DQ'
    if win_val in ('No Contest', 'NC'):
        return 'No Contest'
    if win_val == 'Draw':
        return 'Draw'
    if win_val == 'Yes':
        return 'Win'
    if win_val == 'No':
        return 'Loss'
    return 'Upcoming'

data['DetailedResult'] = data.apply(detailed_result, axis=1)
data['Fight'] = data['Fighter'].astype(str) + ' vs ' + data['Opponent'].astype(str)

color_map = {
    'Win': 'green',
    'Loss': 'red',
    'Win by DQ': 'limegreen',
    'Loss by DQ': 'darkred',
    'No Contest': 'purple',
    'Upcoming': 'blue',
    'Draw': 'gray'
}

prior_weight = st.sidebar.slider("Bayesian prior weight", 0.0, 20.0, 5.0, step=0.5, key="prior_weight_global")
recent_window = st.sidebar.slider("Recent fights window", 1, 100, 50, key="recent_win_global")

# =========================================================================
# TRAIN MODELS ON FILTERED DATA
# =========================================================================
def train_models_on_filtered():
    x_lr = st.session_state.x_lr
    y_lr = st.session_state.y_lr
    z_lr = st.session_state.z_lr
    x_knn = st.session_state.x_knn
    y_knn = st.session_state.y_knn
    z_knn = st.session_state.z_knn
    k_knn = st.session_state.knn_model_k

    # LR
    if x_lr is None or y_lr is None or z_lr is None:
        st.session_state.lr_model = None
        st.session_state.lr_train_status = "LR features not set."
        st.session_state.y_train_lr = None
        st.session_state.lr_feature_names = []
    else:
        hist = data[data['Win?'].isin(['Yes','No'])].copy()
        sub = hist[[x_lr, y_lr, z_lr, 'Win?']].dropna()
        if len(sub) < 10:
            st.session_state.lr_model = None
            st.session_state.lr_train_status = f"LR: only {len(sub)} rows (need ≥10)."
            st.session_state.y_train_lr = None
            st.session_state.lr_feature_names = []
        elif sub['Win?'].nunique() < 2:
            st.session_state.lr_model = None
            st.session_state.lr_train_status = "LR: need both Win and Loss."
            st.session_state.y_train_lr = None
            st.session_state.lr_feature_names = []
        else:
            try:
                sub['target'] = (sub['Win?'] == 'Yes').astype(int)
                X = sub[[x_lr, y_lr, z_lr]].values
                y = sub['target'].values
                lr = LogisticRegression(max_iter=1000)
                lr.fit(X, y)
                st.session_state.lr_model = lr
                st.session_state.lr_train_status = f"LR trained on {len(sub)} fights."
                st.session_state.y_train_lr = y
                st.session_state.X_train_lr = X
                st.session_state.lr_feature_names = [x_lr, y_lr, z_lr]
            except Exception as e:
                st.session_state.lr_model = None
                st.session_state.lr_train_status = f"LR error: {str(e)}"
                st.session_state.y_train_lr = None
                st.session_state.lr_feature_names = []

    # KNN
    if x_knn is None or y_knn is None or z_knn is None:
        st.session_state.calibrated_knn = None
        st.session_state.knn_train_status = "KNN features not set."
        st.session_state.y_train_knn = None
        st.session_state.knn_feature_names = []
    else:
        hist = data[data['Win?'].isin(['Yes','No'])].copy()
        c1 = get_first_col(hist, x_knn)
        c2 = get_first_col(hist, y_knn)
        c3 = get_first_col(hist, z_knn)
        win_col = hist['Win?']
        if isinstance(win_col, pd.DataFrame):
            win_vals = win_col.iloc[:, 0].values
        else:
            win_vals = win_col.values
        train_df = pd.DataFrame({'f1': c1, 'f2': c2, 'f3': c3, 'Win?': win_vals}).dropna()
        if len(train_df) < 10:
            st.session_state.calibrated_knn = None
            st.session_state.knn_train_status = f"KNN: only {len(train_df)} rows (need ≥10)."
            st.session_state.y_train_knn = None
            st.session_state.knn_feature_names = []
        elif train_df['Win?'].nunique() < 2:
            st.session_state.calibrated_knn = None
            st.session_state.knn_train_status = "KNN: need both Win and Loss."
            st.session_state.y_train_knn = None
            st.session_state.knn_feature_names = []
        else:
            try:
                X = train_df[['f1','f2','f3']].values.astype(np.float64)
                y = (train_df['Win?'] == 'Yes').astype(int).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                base_knn = KNeighborsClassifier(n_neighbors=k_knn, weights='distance')
                calibrated = CalibratedClassifierCV(base_knn, method='sigmoid', cv=5)
                calibrated.fit(X_scaled, y)
                st.session_state.calibrated_knn = calibrated
                st.session_state.scaler = scaler
                st.session_state.X_train = X
                st.session_state.y_train_knn = y
                st.session_state.knn_train_status = f"KNN trained on {len(train_df)} fights."
                st.session_state.knn_feature_names = [x_knn, y_knn, z_knn]
            except Exception as e:
                st.session_state.calibrated_knn = None
                st.session_state.knn_train_status = f"KNN error: {str(e)}"
                st.session_state.y_train_knn = None
                st.session_state.knn_feature_names = []

# Train models now
train_models_on_filtered()

# =========================================================================
# PERFORMANCE SUMMARY
# =========================================================================
st.title("UFC Pre‑Fight Performance Dashboard (Adjusted)")

if len(data) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

total = len(data)
wins = (data['Win?'] == 'Yes').sum()
win_rate = wins / total * 100

st.header("Performance Summary (2015+)")
col1, col2, col3 = st.columns(3)
col1.metric("Total Fights", total)
col2.metric("Wins", wins)
col3.metric("Win Rate", f"{win_rate:.1f}%")

cols_to_show = ['CareerWinPct_diff', 'AgeDiff', 'HeightDiff', 'ReachDiff', 'DaysSincePrev_diff']
cols_to_show = [c for c in cols_to_show if c in data.columns]
if cols_to_show:
    colA, colB = st.columns(2)
    for result, col in zip(['Yes', 'No'], [colA, colB]):
        subset = data[data['Win?'] == result]
        if len(subset) == 0: continue
        label = "Winners" if result == 'Yes' else "Losers"
        with col:
            st.subheader(label)
            for feat in cols_to_show:
                if feat in subset.columns:
                    st.write(f"**{feat}:** {subset[feat].mean():.2f}")

# =========================================================================
# UPCOMING FIGHT MATCHUP (with adjperf table)
# =========================================================================
st.header("Upcoming Fight Matchup")

if st.session_state.lr_train_status and "error" not in st.session_state.lr_train_status.lower():
    st.success(f"✅ LR: {st.session_state.lr_train_status}")
else:
    st.error(f"❌ LR: {st.session_state.lr_train_status}")
if st.session_state.knn_train_status and "error" not in st.session_state.knn_train_status.lower():
    st.success(f"✅ KNN: {st.session_state.knn_train_status}")
else:
    st.error(f"❌ KNN: {st.session_state.knn_train_status}")

upcoming_display = matchup_data[matchup_data['Win?'].isna() | (matchup_data['Win?'] == '')]
st.write(f"**Upcoming fights after filters:** {len(upcoming_display['FightID'].unique())}")

if not upcoming_display.empty:
    upcoming_fight_ids = sorted(upcoming_display['FightID'].unique())
    selected_fight = st.selectbox("Choose an upcoming fight", upcoming_fight_ids)
    if selected_fight:
        fight_rows = upcoming_display[upcoming_display['FightID'] == selected_fight]
        if len(fight_rows) == 2:
            f1_row = fight_rows.iloc[0]
            f2_row = fight_rows.iloc[1]
            st.session_state.selected_fight_row = f1_row
            st.write(f"### {f1_row['Fighter']} vs {f2_row['Fighter']}")

            def show_fighter_stats(row, label):
                st.subheader(label)
                basic_cols = ['Age', 'AgeDiff', 'HeightDiff', 'ReachDiff', 'DaysSincePrev',
                              'DaysSincePrev_diff', 'Avg3DaysGap_diff', 'CareerWinPct_diff',
                              'Prev7WinPct', 'FighterOddsNum', 'PrevFighterOddsNum',
                              'ColleyDecayDiff', 'MasseyDecayDiff', 'WeightedMasseyDecayDiff']
                for col in basic_cols:
                    if col in row:
                        st.write(f"**{col}:** {row[col]:.2f}" if isinstance(row[col], (int, float)) else f"**{col}:** {row[col]}")
                st.write("---")
                
                # Show adjperf stats (pre‑fight)
                st.write("**Adjusted Performance (pre‑fight, from past data)**")
                adjperf_cols = [c for c in row.index if c.startswith('adjperf_') and not c.endswith('_diff') and 'Opponent_' not in c]
                if adjperf_cols:
                    opp_adjperf = {}
                    for c in adjperf_cols:
                        opp_c = f'Opponent_{c}'
                        if opp_c in row.index:
                            opp_adjperf[c] = row[opp_c]
                    data_dict = {'Fighter': {}, 'Opponent': {}, 'Diff': {}}
                    for c in adjperf_cols:
                        data_dict['Fighter'][c] = row[c]
                        data_dict['Opponent'][c] = opp_adjperf.get(c, np.nan)
                        diff_c = f'{c}_diff'
                        if diff_c in row.index:
                            data_dict['Diff'][c] = row[diff_c]
                    df_adj = pd.DataFrame(data_dict).T
                    st.dataframe(df_adj, use_container_width=True)
                else:
                    st.write("No adjperf columns found.")
                st.write("---")

            colA, colB = st.columns(2)
            with colA:
                show_fighter_stats(f1_row, f1_row['Fighter'])
            with colB:
                show_fighter_stats(f2_row, f2_row['Fighter'])

            st.subheader(f"Model Win Probabilities for {f1_row['Fighter']}")

            # LR
            lr_model = st.session_state.lr_model
            lr_feats = st.session_state.lr_feature_names
            if lr_model is not None and len(lr_feats) == 3:
                def safe_val(row, col):
                    if col not in row.index:
                        return 0.0
                    try:
                        val = row[col]
                        return val if pd.notna(val) else 0.0
                    except:
                        return 0.0
                v1 = safe_val(f1_row, lr_feats[0])
                v2 = safe_val(f1_row, lr_feats[1])
                v3 = safe_val(f1_row, lr_feats[2])
                try:
                    prob_lr = lr_model.predict_proba(np.array([[v1, v2, v3]]))[0, 1]
                    overall_wr = st.session_state.overall_wr
                    recent_wr = st.session_state.recent_wr
                    recent_count = st.session_state.recent_count
                    if recent_count > 0:
                        shrunk_recent = (prior_weight * overall_wr + recent_count * recent_wr) / (prior_weight + recent_count)
                    else:
                        shrunk_recent = overall_wr
                    shrunk_lr = (prior_weight * (shrunk_recent / 100) + prob_lr) / (prior_weight + 1)
                    st.write(f"**LR win probability:** {prob_lr:.1%}  |  **shrunken:** {shrunk_lr:.1%}")
                except Exception as e:
                    st.error(f"LR probability error: {e}")
            else:
                st.info("LR model not trained or feature mismatch.")

            # KNN
            calibrated_knn = st.session_state.calibrated_knn
            scaler = st.session_state.scaler
            X_train = st.session_state.X_train
            knn_feats = st.session_state.knn_feature_names
            if calibrated_knn is not None and scaler is not None and len(knn_feats) == 3 and X_train is not None:
                means_knn = X_train.mean(axis=0) if X_train is not None else np.zeros(3)
                vals_knn = []
                for i, col_name in enumerate(knn_feats):
                    raw = f1_row[col_name] if col_name in f1_row else np.nan
                    try:
                        v = float(raw) if pd.notna(raw) else means_knn[i]
                    except:
                        v = means_knn[i]
                    vals_knn.append(v)
                try:
                    up_arr = np.array([vals_knn], dtype=np.float64)
                    up_scaled = scaler.transform(up_arr)
                    prob_knn = calibrated_knn.predict_proba(up_scaled)[0, 1]
                    prob_knn = np.clip(prob_knn, 0.1, 0.9)
                    overall_wr = st.session_state.overall_wr
                    recent_wr = st.session_state.recent_wr
                    recent_count = st.session_state.recent_count
                    if recent_count > 0:
                        shrunk_recent = (prior_weight * overall_wr + recent_count * recent_wr) / (prior_weight + recent_count)
                    else:
                        shrunk_recent = overall_wr
                    shrunk_knn = (prior_weight * (shrunk_recent / 100) + prob_knn) / (prior_weight + 1)
                    st.write(f"**KNN win probability:** {prob_knn:.1%}  |  **shrunken:** {shrunk_knn:.1%}")
                except Exception as e:
                    st.error(f"KNN probability error: {e}")
            else:
                st.info("KNN model not trained or feature mismatch.")
        else:
            st.warning(f"Expected 2 rows for this fight, but got {len(fight_rows)}. Check data.")
else:
    st.write("No upcoming fights match the current filters.")

# =========================================================================
# 3D LR SCATTER & COMBO BUILDER
# =========================================================================
st.header("3D LR Win/Loss Prediction & Best LR Combinations")

if len(three_d_features) >= 3:
    col1, col2, col3 = st.columns(3)
    with col1:
        x_lr = st.selectbox("X", three_d_features, index=three_d_features.index(st.session_state.x_lr) if st.session_state.x_lr in three_d_features else 0, key="lr_x")
    with col2:
        y_lr = st.selectbox("Y", three_d_features, index=three_d_features.index(st.session_state.y_lr) if st.session_state.y_lr in three_d_features else min(1, len(three_d_features)-1), key="lr_y")
    with col3:
        z_lr = st.selectbox("Z", three_d_features, index=three_d_features.index(st.session_state.z_lr) if st.session_state.z_lr in three_d_features else min(2, len(three_d_features)-1), key="lr_z")

    if (x_lr != st.session_state.x_lr or y_lr != st.session_state.y_lr or z_lr != st.session_state.z_lr):
        st.session_state.x_lr = x_lr
        st.session_state.y_lr = y_lr
        st.session_state.z_lr = z_lr
        train_models_on_filtered()
        st.rerun()

    if x_lr and y_lr and z_lr:
        plot_data = data[[x_lr, y_lr, z_lr, 'DetailedResult', 'Fight']].copy()
        plot_data = plot_data.loc[:, ~plot_data.columns.duplicated()].dropna()
        if len(plot_data) >= 10:
            fig = px.scatter_3d(
                plot_data,
                x=x_lr, y=y_lr, z=z_lr,
                color='DetailedResult',
                color_discrete_map=color_map,
                hover_data=['Fight'],
                title="3D Scatter – Logistic Regression"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough data for 3D LR plot.")

        # Display LR metrics
        lr_model = st.session_state.lr_model
        if lr_model is not None and hasattr(st.session_state, 'y_train_lr') and st.session_state.y_train_lr is not None:
            X_lr = st.session_state.X_train_lr
            y_lr_true = st.session_state.y_train_lr
            if X_lr is not None and len(X_lr) > 0:
                y_prob_lr = lr_model.predict_proba(X_lr)[:, 1]
                ll_lr = log_loss(y_lr_true, y_prob_lr)
                bs_lr = brier_score_loss(y_lr_true, y_prob_lr)
                overall_wr = st.session_state.overall_wr
                recent_wr = st.session_state.recent_wr
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("LR Log‑loss", f"{ll_lr:.3f}")
                with col_m2:
                    st.metric("LR Brier", f"{bs_lr:.3f}")
                with col_m3:
                    st.metric("Overall Win%", f"{overall_wr:.1f}%")
                    st.metric(f"Recent Win% (last {recent_window})", f"{recent_wr:.1f}%")

                # Predicted probability for selected fight
                if st.session_state.selected_fight_row is not None:
                    f1_row = st.session_state.selected_fight_row
                    st.subheader(f"Predicted LR Win Probability for {f1_row['Fighter']} (using current features)")
                    lr_feats = st.session_state.lr_feature_names
                    if len(lr_feats) == 3:
                        v1 = f1_row[lr_feats[0]] if lr_feats[0] in f1_row else 0.0
                        v2 = f1_row[lr_feats[1]] if lr_feats[1] in f1_row else 0.0
                        v3 = f1_row[lr_feats[2]] if lr_feats[2] in f1_row else 0.0
                        try:
                            prob_lr_fight = lr_model.predict_proba(np.array([[v1, v2, v3]]))[0, 1]
                            overall_wr = st.session_state.overall_wr
                            recent_wr = st.session_state.recent_wr
                            recent_count = st.session_state.recent_count
                            if recent_count > 0:
                                shrunk_recent = (prior_weight * overall_wr + recent_count * recent_wr) / (prior_weight + recent_count)
                            else:
                                shrunk_recent = overall_wr
                            shrunk_lr_fight = (prior_weight * (shrunk_recent / 100) + prob_lr_fight) / (prior_weight + 1)
                            st.write(f"**LR win probability:** {prob_lr_fight:.1%}  |  **shrunken:** {shrunk_lr_fight:.1%}")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
            else:
                st.info("No training data available for LR metrics.")
        else:
            st.info("LR model not trained.")

        # ---- LR combo builder ----
        st.subheader("LR 3‑Variable Combinations (Cross‑Validated Brier)")
        candidates = [c for c in combo_candidates if c in data.columns and c != 'FighterOddsNum']
        if len(candidates) < 3:
            st.warning("Not enough features to test (need at least 3).")
        else:
            @st.cache_data
            def numerical_importance(_data, features):
                hist = _data[_data['Win?'].isin(['Yes','No'])].copy()
                hist['Target'] = (hist['Win?'] == 'Yes').astype(int)
                features = list(dict.fromkeys(features))
                X = hist[features].dropna()
                y = hist.loc[X.index, 'Target']
                if len(X) > 10:
                    X_imp = SimpleImputer(strategy='median').fit_transform(X)
                    mi = mutual_info_classif(X_imp, y, discrete_features=False)
                    return pd.DataFrame({'Feature': features, 'Mutual Information': mi}).sort_values('Mutual Information', ascending=False).head(20)
                return pd.DataFrame()
            mi_df = numerical_importance(data, candidates)
            top_feats = mi_df['Feature'].tolist() if not mi_df.empty else candidates
            num_top = st.slider("Top features to test", 5, min(30, len(top_feats)), 10, key="lr_combo_top")
            candidates = top_feats[:num_top]
            candidates = [c for c in candidates if c != 'FighterOddsNum']

            data_fp = hash(str(data.shape))
            if "lr_combo_results" not in st.session_state:
                st.session_state.lr_combo_results = None
                st.session_state.lr_combo_hash = data_fp
            if st.session_state.lr_combo_hash != data_fp:
                st.session_state.lr_combo_results = None
                st.session_state.lr_combo_hash = data_fp

            if len(candidates) >= 3:
                if st.button("Compute LR 3‑Var Combos (Cross‑Validated)", key="lr_combo_btn"):
                    with st.spinner("Computing cross‑validated Brier (5‑fold)..."):
                        hist = data[data['Win?'].isin(['Yes','No'])].copy()
                        hist['WinNum'] = (hist['Win?'] == 'Yes').astype(int)
                        results = []
                        for combo in itertools.combinations(candidates, 3):
                            sub = hist[list(combo) + ['WinNum']].dropna()
                            if len(sub) < 10 or sub['WinNum'].nunique() < 2:
                                continue
                            X = sub[list(combo)].values
                            y = sub['WinNum'].values
                            try:
                                lr = LogisticRegression(max_iter=1000)
                                y_prob = cross_val_predict(lr, X, y, cv=5, method='predict_proba')[:, 1]
                                bs = brier_score_loss(y, y_prob)
                                results.append({'Variables': ', '.join(combo), 'CV Brier': bs})
                            except:
                                pass
                        if results:
                            st.session_state.lr_combo_results = pd.DataFrame(results).sort_values('CV Brier').head(20)
                        else:
                            st.warning("Could not evaluate any combination.")
                if st.session_state.lr_combo_results is not None:
                    st.write("**Top 20 3‑Variable Combinations (Cross‑Validated Brier)**")
                    st.dataframe(st.session_state.lr_combo_results, use_container_width=True)
            else:
                st.warning("Not enough features to test (need at least 3).")
else:
    st.warning("Not enough numerical features for a 3D LR plot (need at least 3).")

# =========================================================================
# 3D KNN SCATTER & COMBO BUILDER
# =========================================================================
st.header("3D Weighted KNN Win/Loss Prediction (Platt‑scaled) & Best KNN Combinations")

if len(three_d_features) >= 3:
    col1k, col2k, col3k = st.columns(3)
    with col1k:
        x_knn = st.selectbox("X", three_d_features, index=three_d_features.index(st.session_state.x_knn) if st.session_state.x_knn in three_d_features else 0, key="knn_x")
    with col2k:
        y_knn = st.selectbox("Y", three_d_features, index=three_d_features.index(st.session_state.y_knn) if st.session_state.y_knn in three_d_features else min(1, len(three_d_features)-1), key="knn_y")
    with col3k:
        z_knn = st.selectbox("Z", three_d_features, index=three_d_features.index(st.session_state.z_knn) if st.session_state.z_knn in three_d_features else min(2, len(three_d_features)-1), key="knn_z")

    if (x_knn != st.session_state.x_knn or y_knn != st.session_state.y_knn or z_knn != st.session_state.z_knn):
        st.session_state.x_knn = x_knn
        st.session_state.y_knn = y_knn
        st.session_state.z_knn = z_knn
        train_models_on_filtered()
        st.rerun()

    if x_knn and y_knn and z_knn:
        plot_data_knn = data[[x_knn, y_knn, z_knn, 'DetailedResult', 'Fight']].copy()
        plot_data_knn = plot_data_knn.loc[:, ~plot_data_knn.columns.duplicated()].dropna()
        if len(plot_data_knn) >= 10:
            fig_knn = px.scatter_3d(
                plot_data_knn,
                x=x_knn, y=y_knn, z=z_knn,
                color='DetailedResult',
                color_discrete_map=color_map,
                hover_data=['Fight'],
                title="3D Scatter – Weighted KNN"
            )
            st.plotly_chart(fig_knn, use_container_width=True)
        else:
            st.warning("Not enough data for 3D KNN plot.")

        # Display KNN metrics
        calibrated_knn = st.session_state.calibrated_knn
        if calibrated_knn is not None and hasattr(st.session_state, 'y_train_knn') and st.session_state.y_train_knn is not None:
            X_knn = st.session_state.X_train
            y_knn_true = st.session_state.y_train_knn
            if X_knn is not None and len(X_knn) > 0:
                scaler = st.session_state.scaler
                X_scaled = scaler.transform(X_knn)
                y_prob_knn = calibrated_knn.predict_proba(X_scaled)[:, 1]
                y_prob_knn = np.clip(y_prob_knn, 0.1, 0.9)
                ll_knn = log_loss(y_knn_true, y_prob_knn)
                bs_knn = brier_score_loss(y_knn_true, y_prob_knn)
                overall_wr = st.session_state.overall_wr
                recent_wr = st.session_state.recent_wr
                col_m1, col_m2, col_m3 = st.columns(3)
                with col_m1:
                    st.metric("KNN Log‑loss", f"{ll_knn:.3f}")
                with col_m2:
                    st.metric("KNN Brier", f"{bs_knn:.3f}")
                with col_m3:
                    st.metric("Overall Win%", f"{overall_wr:.1f}%")
                    st.metric(f"Recent Win% (last {recent_window})", f"{recent_wr:.1f}%")

                # Predicted probability for selected fight
                if st.session_state.selected_fight_row is not None:
                    f1_row = st.session_state.selected_fight_row
                    st.subheader(f"Predicted KNN Win Probability for {f1_row['Fighter']} (using current features)")
                    knn_feats = st.session_state.knn_feature_names
                    if len(knn_feats) == 3:
                        means_knn = X_knn.mean(axis=0)
                        vals_knn = []
                        for i, col_name in enumerate(knn_feats):
                            raw = f1_row[col_name] if col_name in f1_row else np.nan
                            try:
                                v = float(raw) if pd.notna(raw) else means_knn[i]
                            except:
                                v = means_knn[i]
                            vals_knn.append(v)
                        try:
                            up_arr = np.array([vals_knn], dtype=np.float64)
                            up_scaled = scaler.transform(up_arr)
                            prob_knn_fight = calibrated_knn.predict_proba(up_scaled)[0, 1]
                            prob_knn_fight = np.clip(prob_knn_fight, 0.1, 0.9)
                            overall_wr = st.session_state.overall_wr
                            recent_wr = st.session_state.recent_wr
                            recent_count = st.session_state.recent_count
                            if recent_count > 0:
                                shrunk_recent = (prior_weight * overall_wr + recent_count * recent_wr) / (prior_weight + recent_count)
                            else:
                                shrunk_recent = overall_wr
                            shrunk_knn_fight = (prior_weight * (shrunk_recent / 100) + prob_knn_fight) / (prior_weight + 1)
                            st.write(f"**KNN win probability:** {prob_knn_fight:.1%}  |  **shrunken:** {shrunk_knn_fight:.1%}")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                    else:
                        st.info("KNN features not set correctly.")
            else:
                st.info("No training data available for KNN metrics.")
        else:
            st.info("KNN model not trained.")

        # KNN slider
        k_knn = st.slider("KNN neighbors", 1, 20, 5, key="knn_model_k")
        if k_knn != st.session_state.knn_model_k:
            st.session_state.knn_model_k = k_knn
            train_models_on_filtered()
            st.rerun()

        if st.session_state.calibrated_knn is not None:
            st.write("KNN model trained with current settings.")
        else:
            st.info("KNN model not trained. Check status above.")

        # ---- KNN combo builder ----
        st.subheader("KNN 3‑Variable Combinations (Cross‑Validated Brier)")
        candidates_knn = [c for c in combo_candidates if c in data.columns and c != 'FighterOddsNum']
        if len(candidates_knn) < 3:
            st.warning("Not enough features to test (need at least 3).")
        else:
            # Reuse importance from LR or compute again
            if not mi_df.empty:
                top_features_knn = mi_df['Feature'].tolist()
            else:
                top_features_knn = candidates_knn
            num_top_knn = st.slider("Top features to test", 5, min(30, len(top_features_knn)), 10, key="knn_combo_top")
            candidates_knn = top_features_knn[:num_top_knn]
            candidates_knn = [c for c in candidates_knn if c != 'FighterOddsNum']
            k_combo = st.slider("KNN neighbors (combo builder)", 1, 20, 5, key="knn_combo_k")

            data_fp_knn = hash(str(data.shape))
            if "knn_combo_results" not in st.session_state:
                st.session_state.knn_combo_results = None
                st.session_state.knn_combo_hash = data_fp_knn
            if st.session_state.knn_combo_hash != data_fp_knn:
                st.session_state.knn_combo_results = None
                st.session_state.knn_combo_hash = data_fp_knn

            if len(candidates_knn) >= 3:
                if st.button("Compute KNN 3‑Var Combos (Cross‑Validated)", key="knn_combo_btn"):
                    with st.spinner("Computing cross‑validated Brier (5‑fold) for KNN..."):
                        hist_combo = data[data['Win?'].isin(['Yes','No'])].copy()
                        hist_combo = hist_combo.loc[:, ~hist_combo.columns.duplicated()]
                        hist_combo['WinNum'] = (hist_combo['Win?'] == 'Yes').astype(int)
                        results = []
                        for combo in itertools.combinations(candidates_knn, 3):
                            c1 = get_first_col(hist_combo, combo[0])
                            c2 = get_first_col(hist_combo, combo[1])
                            c3 = get_first_col(hist_combo, combo[2])
                            y = hist_combo['WinNum'].values
                            mask = ~(np.isnan(c1) | np.isnan(c2) | np.isnan(c3))
                            if mask.sum() < 10 or np.unique(y[mask]).size < 2:
                                continue
                            X = np.column_stack([c1[mask], c2[mask], c3[mask]])
                            y_clean = y[mask]
                            try:
                                pipeline = Pipeline([
                                    ('scaler', StandardScaler()),
                                    ('knn', KNeighborsClassifier(n_neighbors=k_combo, weights='distance'))
                                ])
                                y_prob = cross_val_predict(pipeline, X, y_clean, cv=5, method='predict_proba')[:, 1]
                                y_prob = np.clip(y_prob, 0.1, 0.9)
                                bs = brier_score_loss(y_clean, y_prob)
                                results.append({'Variables': ', '.join(combo), 'CV Brier': bs})
                            except:
                                pass
                        if results:
                            st.session_state.knn_combo_results = pd.DataFrame(results).sort_values('CV Brier').head(20)
                        else:
                            st.warning("Could not evaluate any combination.")
                if st.session_state.knn_combo_results is not None:
                    st.write("**Top 20 3‑Variable Combinations (Cross‑Validated Brier)**")
                    st.dataframe(st.session_state.knn_combo_results, use_container_width=True)
            else:
                st.warning("Not enough features to test (need at least 3).")
else:
    st.warning("Not enough numerical features for a 3D KNN plot (need at least 3).")

# =========================================================================
# LAST 20 FIGHTS
# =========================================================================
st.header("Last 20 Fights")
last20 = data.sort_values('FightDate', ascending=False).head(20)
display_cols = ['FightDate','Fighter','Opponent','Win?','Method']
for col in ['AgeDiff','HeightDiff','ReachDiff','CareerWinPct_diff','Prev7WinPct','ColleyDecayDiff','MasseyDecayDiff','WeightedMasseyDecayDiff']:
    if col in last20.columns:
        display_cols.append(col)
for ks in ['adjperf_KD', 'adjperf_SS', 'adjperf_TD']:
    if f'{ks}_diff' in last20.columns:
        display_cols.append(f'{ks}_diff')
display_cols = [c for c in display_cols if c in last20.columns]
st.dataframe(last20[display_cols])

# =========================================================================
# FEATURE IMPORTANCE (use all new_features)
# =========================================================================
st.header("Top 20 Feature Importance (Current Filter Set)")
hist_imp = data[data['Win?'].isin(['Yes', 'No'])].copy()
if len(hist_imp) < 10:
    st.warning("Too few historical fights after filtering to compute importance.")
else:
    hist_imp['Target'] = (hist_imp['Win?'] == 'Yes').astype(int)
    eligible = list(dict.fromkeys([c for c in three_d_features if c in hist_imp.columns]))
    if eligible:
        X_num = hist_imp[eligible].dropna()
        if len(X_num) > 10 and X_num.shape[1] > 0:
            st.caption(f"Computing importance on **{len(X_num)}** historical fights.")
            imputer = SimpleImputer(strategy='median')
            X_imp = imputer.fit_transform(X_num)
            y_num = hist_imp.loc[X_num.index, 'Target']
            mi = mutual_info_classif(X_imp, y_num, discrete_features=False, random_state=42)
            mi_df_num = pd.DataFrame({'Feature': eligible, 'Mutual Information': mi}).sort_values('Mutual Information', ascending=False).head(20)
            fig_num = px.bar(mi_df_num, x='Mutual Information', y='Feature', orientation='h',
                             title="Top 20 Features by Mutual Information with Win/Loss")
            st.plotly_chart(fig_num, use_container_width=True)
        else:
            st.warning("Not enough complete rows for numerical importance.")
    else:
        st.warning("No numerical features available after filtering.")

# =========================================================================
# SPIDER CHART (using all new_features, similarity only)
# =========================================================================
st.header("Fight Similarity & Comparison (Independent Filters)")
st.subheader("Spider Chart Filters (fighter data only)")

col_sp1, col_sp2 = st.columns(2)
with col_sp1:
    spider_wc = st.multiselect("Weight Class", sorted(original_data['WC'].dropna().unique()), key="spider_wc") if 'WC' in original_data.columns else []
    spider_stance = st.multiselect("Stance", sorted(original_data['Stance'].dropna().unique()), key="spider_stance") if 'Stance' in original_data.columns else []
    spider_country = st.multiselect("Country", sorted(original_data['Country'].dropna().unique()), key="spider_country") if 'Country' in original_data.columns else []
    spider_sched_rounds = st.multiselect("Scheduled Rounds", sorted(original_data['ScheduledRounds'].dropna().unique()), key="spider_sched") if 'ScheduledRounds' in original_data.columns else []
    spider_event_country = st.multiselect("Event Country", sorted(original_data['EventCountry'].dropna().unique()), key="spider_eventc") if 'EventCountry' in original_data.columns else []
with col_sp2:
    spider_title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"], key="spider_title") if 'Title' in original_data.columns else "All"
    spider_hometown = st.selectbox("Hometown", ["All", "Yes", "No"], key="spider_home") if 'HometownFighter' in original_data.columns else "All"
    spider_new_wc = st.checkbox("New Weight Class", key="spider_new_wc") if 'IsNewWeightClass' in original_data.columns else False
    spider_prev_title = st.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"], key="spider_prev_title")

# Previous outcome filters for spider (use raw)
spider_prev1_col = 'Prev1_Outcome_raw'
spider_prev2_col = 'Prev2_Outcome_raw'
spider_prev3_col = 'Prev3_Outcome_raw'
spider_career1_col = 'Career1_Outcome_raw'
spider_career2_col = 'Career2_Outcome_raw'
spider_career3_col = 'Career3_Outcome_raw'

all_outcomes_raw_spider = sorted(original_data[spider_prev1_col].dropna().unique()) if spider_prev1_col in original_data.columns else []
all_outcomes_career_spider = sorted(original_data[spider_career1_col].dropna().unique()) if spider_career1_col in original_data.columns else []

with st.expander("Previous Outcomes (Spider)"):
    spider_prev1 = st.multiselect("Prev Fight 1", all_outcomes_raw_spider, key="spider_prev1")
    spider_prev2 = st.multiselect("Prev Fight 2", all_outcomes_raw_spider, key="spider_prev2")
    spider_prev3 = st.multiselect("Prev Fight 3", all_outcomes_raw_spider, key="spider_prev3")
    spider_career1 = st.multiselect("Career F1", all_outcomes_career_spider, key="spider_career1")
    spider_career2 = st.multiselect("Career F2", all_outcomes_career_spider, key="spider_career2")
    spider_career3 = st.multiselect("Career F3", all_outcomes_career_spider, key="spider_career3")

# Start with original_data
spider_data = original_data.copy()
mask = pd.Series(True, index=spider_data.index)

if spider_wc and 'WC' in spider_data.columns: mask &= spider_data['WC'].isin(spider_wc)
if spider_stance and 'Stance' in spider_data.columns: mask &= spider_data['Stance'].isin(spider_stance)
if spider_country and 'Country' in spider_data.columns: mask &= spider_data['Country'].isin(spider_country)
if spider_sched_rounds and 'ScheduledRounds' in spider_data.columns: mask &= spider_data['ScheduledRounds'].isin(spider_sched_rounds)
if spider_title_fight != "All" and 'Title' in spider_data.columns: mask &= spider_data['Title'] == spider_title_fight
if spider_hometown != "All" and 'HometownFighter' in spider_data.columns: mask &= spider_data['HometownFighter'] == spider_hometown
if spider_event_country and 'EventCountry' in spider_data.columns: mask &= spider_data['EventCountry'].isin(spider_event_country)
if spider_new_wc and 'IsNewWeightClass' in spider_data.columns: mask &= spider_data['IsNewWeightClass'] == True

# Previous outcomes
if spider_prev1 and spider_prev1_col in spider_data.columns:
    mask &= spider_data[spider_prev1_col].isin(spider_prev1)
if spider_prev2 and spider_prev2_col in spider_data.columns:
    mask &= spider_data[spider_prev2_col].isin(spider_prev2)
if spider_prev3 and spider_prev3_col in spider_data.columns:
    mask &= spider_data[spider_prev3_col].isin(spider_prev3)
if spider_career1 and spider_career1_col in spider_data.columns:
    mask &= spider_data[spider_career1_col].isin(spider_career1)
if spider_career2 and spider_career2_col in spider_data.columns:
    mask &= spider_data[spider_career2_col].isin(spider_career2)
if spider_career3 and spider_career3_col in spider_data.columns:
    mask &= spider_data[spider_career3_col].isin(spider_career3)

# Title filter at fight level
if 'Prev1_Title' in spider_data.columns:
    spider_data['Prev1_Title_clean'] = normalize_title_col(spider_data['Prev1_Title'])
    if spider_prev_title != "All":
        fighter_mask = spider_data['Prev1_Title_clean'] == spider_prev_title.lower()
        matching_fight_ids = spider_data.loc[fighter_mask, 'FightID'].unique()
        mask &= spider_data['FightID'].isin(matching_fight_ids)

filtered_spider = spider_data[mask]
surviving_spider_fight_ids = filtered_spider['FightID'].unique()
spider_data = original_data[original_data['FightID'].isin(surviving_spider_fight_ids)]

spider_upcoming = spider_data[spider_data['Win?'].isna() | (spider_data['Win?'] == '')]
spider_hist = spider_data[spider_data['Win?'].isin(['Yes','No'])].copy()

if spider_upcoming.empty:
    st.write("No upcoming fights after spider filters.")
else:
    fight_counts = spider_upcoming.groupby('FightID').size()
    complete_ids = fight_counts[fight_counts == 2].index
    spider_upcoming = spider_upcoming[spider_upcoming['FightID'].isin(complete_ids)]
    if spider_upcoming.empty:
        st.warning("No upcoming fight has both fighters after spider filters.")
    else:
        sim_features = [c for c in new_features if c in spider_data.columns and c not in ['FightNumber', 'FightNumber_diff']]
        sim_features = [c for c in sim_features if c not in ['Win?', 'Method', 'Round', 'Title']]
        if not sim_features:
            st.warning("No numeric features for similarity.")
        else:
            selected_vars = st.multiselect("Select variables for similarity", sim_features, default=sim_features[:5], max_selections=8, key="spider_vars")
            if selected_vars:
                hist_sub = spider_hist[selected_vars].dropna()
                if len(hist_sub) < 2:
                    st.warning("Not enough historical data.")
                else:
                    scaler_sim = StandardScaler()
                    scaler_sim.fit(hist_sub)
                    up_ids = spider_upcoming['FightID'].unique()
                    selected_fight_spider = st.selectbox("Choose an upcoming fight for similarity", up_ids, key="spider_fight_select")
                    if selected_fight_spider:
                        fight_rows = spider_upcoming[spider_upcoming['FightID'] == selected_fight_spider]
                        f1 = fight_rows.iloc[0]
                        f2 = fight_rows.iloc[1]
                        st.write(f"### {f1['Fighter']} vs {f2['Fighter']}")

                        up_vals = []
                        for var in selected_vars:
                            raw = f1[var]
                            try:
                                v = float(raw) if pd.notna(raw) else 0.0
                            except:
                                v = 0.0
                            up_vals.append(v)
                        up_vec = np.array([up_vals], dtype=np.float64)
                        up_scaled = scaler_sim.transform(up_vec)

                        hist_scaled = scaler_sim.transform(hist_sub)
                        dists = cdist(up_scaled, hist_scaled, 'euclidean').flatten()
                        max_dist = dists.max() if dists.max() > 0 else 1.0
                        sim_scores = 100 * (1 - dists / max_dist)

                        sim_df = spider_hist.loc[hist_sub.index, ['FightDate', 'Fighter', 'Opponent', 'Win?']].copy()
                        sim_df['Similarity'] = sim_scores.round(1)
                        sim_df = sim_df.sort_values('Similarity', ascending=False)

                        st.subheader("Similarity Metrics")
                        n_top = st.slider("Number of top similar fights to consider", 5, 100, 50, step=5, key="spider_top_n")
                        top_n = sim_df.head(n_top)
                        count = len(top_n)
                        avg_sim = top_n['Similarity'].mean()
                        total_sim = top_n['Similarity'].sum()
                        composite = avg_sim * (count ** 0.5) / 100
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Count", count)
                        col2.metric("Avg Similarity", f"{avg_sim:.1f}%")
                        col3.metric("Total Similarity", f"{total_sim:.1f}")
                        col4.metric("Composite Score", f"{composite:.1f}")

                        high_sim = top_n[top_n['Similarity'] >= 90]
                        if len(high_sim) > 0:
                            wins_high = (high_sim['Win?'] == 'Yes').sum()
                            win_rate_high = wins_high / len(high_sim) * 100
                            st.metric("Win Rate (Similarity ≥ 90%)", f"{win_rate_high:.1f}%", delta=f"{len(high_sim)} fights")
                        else:
                            st.write("No historical fights with similarity ≥ 90% in the top selection.")

                        st.subheader("Similarity Distribution")
                        fig_hist = px.histogram(sim_df, x='Similarity', nbins=20, title="Similarity Scores (All)")
                        st.plotly_chart(fig_hist, use_container_width=True)

                        st.subheader(f"Top {n_top} Most Similar Historical Fights")
                        st.dataframe(top_n)
