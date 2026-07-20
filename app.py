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
from sklearn.metrics import log_loss, brier_score_loss, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cdist

st.set_page_config(page_title="UFC Pre‑Fight Dashboard (Adjusted)", layout="wide")

# ============================================================
# 🔑 YOUR PARQUET FILE ID – replace with your actual ID
# ============================================================
PARQUET_FILE_ID = "1UIAgg0cHBW5TMekpoohpiP23Fd6aeqg8"   # adjust to your new Parquet ID

@st.cache_data
def load_data():
    gdown.download(f"https://drive.google.com/uc?id={PARQUET_FILE_ID}", "data.parquet", quiet=True)
    return pd.read_parquet("data.parquet")

data = load_data()
original_data = data.copy()

# ---------- Helper functions ----------
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

def normalize_title_col(series):
    if series is None:
        return pd.Series('', index=series.index)
    return series.astype(str).str.strip().str.lower()

# ---------- Define expected columns ----------
adjperf_cols = [c for c in data.columns if c.startswith('adjperf_')]
new_features = [
    'Age', 'AgeDiff', 'HeightDiff', 'ReachDiff',
    'DaysSincePrev', 'DaysSincePrev_diff', 'Avg3DaysGap_diff',
    'FightNumber', 'FightNumber_diff',
    'FighterOddsNum', 'PrevFighterOddsNum',
    'CareerWinPct_diff', 'Prev7WinPct',
    'FighterColleyDecay', 'OpponentColleyDecay', 'ColleyDecayDiff',
    'FighterMasseyDecay', 'OpponentMasseyDecay', 'MasseyDecayDiff',
    'FighterWeightedMasseyDecay', 'OpponentWeightedMasseyDecay', 'WeightedMasseyDecayDiff'
]
for col in adjperf_cols:
    new_features.append(col)
    new_features.append(f'Opponent_{col}')
    new_features.append(f'{col}_diff')
# Keep only existing
new_features = [c for c in new_features if c in data.columns]

three_d_features = [c for c in new_features if data[c].nunique(dropna=True) >= 2 and np.issubdtype(data[c].dtype, np.number)]

# ---------- Sidebar Filters ----------
st.sidebar.title("Filters")
with st.sidebar.expander("General", expanded=True):
    wc = st.multiselect("Weight Class", sorted(data['WC'].dropna().unique())) if 'WC' in data.columns else []
    stance = st.multiselect("Stance", sorted(data['Stance'].dropna().unique())) if 'Stance' in data.columns else []
    country = st.multiselect("Country", sorted(data['Country'].dropna().unique())) if 'Country' in data.columns else []
    sched_rounds = st.multiselect("Scheduled Rounds", sorted(data['ScheduledRounds'].dropna().unique())) if 'ScheduledRounds' in data.columns else []
    title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"]) if 'Title' in data.columns else "All"
    hometown = st.selectbox("Hometown", ["All", "Yes", "No"]) if 'HometownFighter' in data.columns else "All"
    opp_hometown = st.selectbox("Opp Hometown", ["All", "Yes", "No"]) if 'Opponent_Hometown' in data.columns else "All"
    event_country = st.multiselect("Event Country", sorted(data['EventCountry'].dropna().unique())) if 'EventCountry' in data.columns else []

with st.sidebar.expander("Fight Numbers", expanded=False):
    if 'FightNumber' in data.columns:
        fn_min = st.number_input("Min Fight #", value=1, min_value=1, max_value=int(data['FightNumber'].max()))
        fn_max = st.number_input("Max Fight #", value=int(data['FightNumber'].max()))
    else:
        fn_min, fn_max = 1, 1000
    if 'Opponent_FightNumber' in data.columns:
        ofn_min = st.number_input("Opp Min Fight #", value=1)
        ofn_max = st.number_input("Opp Max Fight #", value=int(data['Opponent_FightNumber'].max()))
    else:
        ofn_min, ofn_max = 1, 1000

with st.sidebar.expander("Career Win % Diff", expanded=False):
    if 'CareerWinPct_diff' in data.columns:
        cwp_min = st.slider("Min Career Win % Diff", -100, 100, -100, step=5)
        cwp_max = st.slider("Max Career Win % Diff", -100, 100, 100, step=5)
    else:
        cwp_min, cwp_max = -100, 100

with st.sidebar.expander("Physical Attributes", expanded=False):
    if 'Age' in data.columns:
        age = st.slider("Age", int(data['Age'].min()), int(data['Age'].max()), (int(data['Age'].min()), int(data['Age'].max())))
    else:
        age = (0, 100)
    if 'AgeDiff' in data.columns:
        age_diff = st.slider("Age Diff", int(data['AgeDiff'].min()), int(data['AgeDiff'].max()), (int(data['AgeDiff'].min()), int(data['AgeDiff'].max())))
    else:
        age_diff = (-100, 100)
    if 'HeightDiff' in data.columns:
        height_diff = st.slider("Height Diff (in)", int(data['HeightDiff'].min()), int(data['HeightDiff'].max()), (int(data['HeightDiff'].min()), int(data['HeightDiff'].max())))
    else:
        height_diff = (-50, 50)
    if 'ReachDiff' in data.columns:
        reach_diff = st.slider("Reach Diff (in)", int(data['ReachDiff'].min()), int(data['ReachDiff'].max()), (int(data['ReachDiff'].min()), int(data['ReachDiff'].max())))
    else:
        reach_diff = (-50, 50)

with st.sidebar.expander("Days & Gaps", expanded=False):
    if 'DaysSincePrev' in data.columns:
        days = st.slider("Days Since Prev", int(data['DaysSincePrev'].min()), int(data['DaysSincePrev'].max()), (int(data['DaysSincePrev'].min()), int(data['DaysSincePrev'].max())))
    else:
        days = (0, 1000)
    if 'DaysSincePrev_diff' in data.columns:
        days_diff = st.slider("Days Since Prev Diff", int(data['DaysSincePrev_diff'].min()), int(data['DaysSincePrev_diff'].max()), (int(data['DaysSincePrev_diff'].min()), int(data['DaysSincePrev_diff'].max())))
    else:
        days_diff = (-1000, 1000)
    if 'Avg3DaysGap_diff' in data.columns:
        avg3_diff = st.slider("Avg3DaysGap Diff", int(data['Avg3DaysGap_diff'].min()), int(data['Avg3DaysGap_diff'].max()), (int(data['Avg3DaysGap_diff'].min()), int(data['Avg3DaysGap_diff'].max())))
    else:
        avg3_diff = (-1000, 1000)

with st.sidebar.expander("Odds", expanded=False):
    if 'FighterOddsNum' in data.columns:
        cur_odds = st.slider("Fighter Odds", int(data['FighterOddsNum'].min()), int(data['FighterOddsNum'].max()), (int(data['FighterOddsNum'].min()), int(data['FighterOddsNum'].max())), step=10)
    else:
        cur_odds = (-1000, 1000)
    if 'PrevFighterOddsNum' in data.columns:
        prev_odds = st.slider("Prev Fighter Odds", int(data['PrevFighterOddsNum'].min()), int(data['PrevFighterOddsNum'].max()), (int(data['PrevFighterOddsNum'].min()), int(data['PrevFighterOddsNum'].max())), step=10)
    else:
        prev_odds = (-1000, 1000)

new_wc = st.sidebar.checkbox("New Weight Class") if 'IsNewWeightClass' in data.columns else False
prev_title = st.sidebar.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"])
opp_prev_title = st.sidebar.selectbox("Opp Prev Fight Was Title?", ["All", "Yes", "No"])

# ---------- Apply main filters ----------
filtered = data.copy()

if wc and 'WC' in filtered.columns: filtered = filtered[filtered['WC'].isin(wc)]
if stance and 'Stance' in filtered.columns: filtered = filtered[filtered['Stance'].isin(stance)]
if country and 'Country' in filtered.columns: filtered = filtered[filtered['Country'].isin(country)]
if sched_rounds and 'ScheduledRounds' in filtered.columns: filtered = filtered[filtered['ScheduledRounds'].isin(sched_rounds)]
if title_fight != "All" and 'Title' in filtered.columns: filtered = filtered[filtered['Title'] == title_fight]
if hometown != "All" and 'HometownFighter' in filtered.columns: filtered = filtered[filtered['HometownFighter'] == hometown]
if opp_hometown != "All" and 'Opponent_Hometown' in filtered.columns: filtered = filtered[filtered['Opponent_Hometown'] == opp_hometown]
if event_country and 'EventCountry' in filtered.columns: filtered = filtered[filtered['EventCountry'].isin(event_country)]
if new_wc and 'IsNewWeightClass' in filtered.columns: filtered = filtered[filtered['IsNewWeightClass'] == True]

if 'Prev1_Title' in filtered.columns:
    filtered['Prev1_Title_clean'] = normalize_title_col(filtered['Prev1_Title'])
    if prev_title != "All":
        filtered = filtered[filtered['Prev1_Title_clean'] == prev_title.lower()]
if 'Opponent_Prev1_Title' in filtered.columns:
    filtered['Opp_Prev1_Title_clean'] = normalize_title_col(filtered['Opponent_Prev1_Title'])
    if opp_prev_title != "All":
        filtered = filtered[filtered['Opp_Prev1_Title_clean'] == opp_prev_title.lower()]

if 'FightNumber' in filtered.columns:
    filtered = filtered[(filtered['FightNumber'] >= fn_min) & (filtered['FightNumber'] <= fn_max)]
if 'Opponent_FightNumber' in filtered.columns:
    filtered = filtered[(filtered['Opponent_FightNumber'] >= ofn_min) & (filtered['Opponent_FightNumber'] <= ofn_max)]
if 'Age' in filtered.columns:
    filtered = filtered[(filtered['Age'] >= age[0]) & (filtered['Age'] <= age[1])]
if 'AgeDiff' in filtered.columns:
    filtered = filtered[(filtered['AgeDiff'] >= age_diff[0]) & (filtered['AgeDiff'] <= age_diff[1])]
if 'HeightDiff' in filtered.columns:
    filtered = filtered[(filtered['HeightDiff'] >= height_diff[0]) & (filtered['HeightDiff'] <= height_diff[1])]
if 'ReachDiff' in filtered.columns:
    filtered = filtered[(filtered['ReachDiff'] >= reach_diff[0]) & (filtered['ReachDiff'] <= reach_diff[1])]
if 'DaysSincePrev' in filtered.columns:
    filtered = filtered[(filtered['DaysSincePrev'] >= days[0]) & (filtered['DaysSincePrev'] <= days[1])]
if 'DaysSincePrev_diff' in filtered.columns:
    filtered = filtered[(filtered['DaysSincePrev_diff'] >= days_diff[0]) & (filtered['DaysSincePrev_diff'] <= days_diff[1])]
if 'Avg3DaysGap_diff' in filtered.columns:
    filtered = filtered[(filtered['Avg3DaysGap_diff'] >= avg3_diff[0]) & (filtered['Avg3DaysGap_diff'] <= avg3_diff[1])]
if 'CareerWinPct_diff' in filtered.columns:
    filtered = filtered[(filtered['CareerWinPct_diff'] >= cwp_min) & (filtered['CareerWinPct_diff'] <= cwp_max)]
if 'FighterOddsNum' in filtered.columns:
    filtered = filtered[(filtered['FighterOddsNum'] >= cur_odds[0]) & (filtered['FighterOddsNum'] <= cur_odds[1])]
if 'PrevFighterOddsNum' in filtered.columns:
    filtered = filtered[(filtered['PrevFighterOddsNum'] >= prev_odds[0]) & (filtered['PrevFighterOddsNum'] <= prev_odds[1])]

data = filtered
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
# Initialize session state
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

# Set default features: use first three from three_d_features
if len(three_d_features) >= 3:
    default_lr = three_d_features[:3]
    default_knn = three_d_features[:3]
else:
    default_lr = default_knn = []

if 'x_lr' not in st.session_state:
    st.session_state.x_lr = default_lr[0] if len(default_lr) > 0 else None
if 'y_lr' not in st.session_state:
    st.session_state.y_lr = default_lr[1] if len(default_lr) > 1 else None
if 'z_lr' not in st.session_state:
    st.session_state.z_lr = default_lr[2] if len(default_lr) > 2 else None
if 'x_knn' not in st.session_state:
    st.session_state.x_knn = default_knn[0] if len(default_knn) > 0 else None
if 'y_knn' not in st.session_state:
    st.session_state.y_knn = default_knn[1] if len(default_knn) > 1 else None
if 'z_knn' not in st.session_state:
    st.session_state.z_knn = default_knn[2] if len(default_knn) > 2 else None
if 'knn_model_k' not in st.session_state:
    st.session_state.knn_model_k = 5

# Compute overall/recent win rates on filtered data
full_hist = data[data['Win?'].isin(['Yes','No'])].sort_values('FightDate')
if len(full_hist) > 0:
    st.session_state.overall_wr = (full_hist['Win?'] == 'Yes').mean() * 100
    recent = full_hist.tail(recent_window)
    st.session_state.recent_wr = (recent['Win?'] == 'Yes').mean() * 100 if len(recent) > 0 else 0.0
    st.session_state.recent_count = len(recent)

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
    else:
        hist = data[data['Win?'].isin(['Yes','No'])].copy()
        sub = hist[[x_lr, y_lr, z_lr, 'Win?']].dropna()
        if len(sub) < 10:
            st.session_state.lr_model = None
            st.session_state.lr_train_status = f"LR: only {len(sub)} rows (need ≥10)."
            st.session_state.y_train_lr = None
        elif sub['Win?'].nunique() < 2:
            st.session_state.lr_model = None
            st.session_state.lr_train_status = "LR: need both Win and Loss."
            st.session_state.y_train_lr = None
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
            except Exception as e:
                st.session_state.lr_model = None
                st.session_state.lr_train_status = f"LR error: {str(e)}"
                st.session_state.y_train_lr = None

    # KNN
    if x_knn is None or y_knn is None or z_knn is None:
        st.session_state.calibrated_knn = None
        st.session_state.knn_train_status = "KNN features not set."
        st.session_state.y_train_knn = None
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
        elif train_df['Win?'].nunique() < 2:
            st.session_state.calibrated_knn = None
            st.session_state.knn_train_status = "KNN: need both Win and Loss."
            st.session_state.y_train_knn = None
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
            except Exception as e:
                st.session_state.calibrated_knn = None
                st.session_state.knn_train_status = f"KNN error: {str(e)}"
                st.session_state.y_train_knn = None

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

col1, col2 = st.columns(2)
for result, col in zip(['Yes', 'No'], [col1, col2]):
    subset = data[data['Win?'] == result]
    if len(subset) == 0: continue
    label = "Winners" if result == 'Yes' else "Losers"
    with col:
        st.subheader(label)
        # Show means of selected features for winners/losers
        for feat in ['CareerWinPct_diff', 'AgeDiff', 'HeightDiff', 'ReachDiff', 'DaysSincePrev_diff']:
            if feat in subset.columns:
                st.write(f"**{feat}:** {subset[feat].mean():.2f}")

# =========================================================================
# UPCOMING FIGHT MATCHUP
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
                # Only show columns that exist
                if 'Age' in row:
                    st.write(f"**Age:** {row['Age']}")
                if 'AgeDiff' in row:
                    st.write(f"**AgeDiff:** {row['AgeDiff']:.1f}")
                if 'HeightDiff' in row:
                    st.write(f"**HeightDiff:** {row['HeightDiff']:.1f} in")
                if 'ReachDiff' in row:
                    st.write(f"**ReachDiff:** {row['ReachDiff']:.1f} in")
                if 'DaysSincePrev' in row:
                    st.write(f"**DaysSincePrev:** {row['DaysSincePrev']:.0f}")
                if 'DaysSincePrev_diff' in row:
                    st.write(f"**DaysSincePrev_diff:** {row['DaysSincePrev_diff']:.0f}")
                if 'Avg3DaysGap_diff' in row:
                    st.write(f"**Avg3DaysGap_diff:** {row['Avg3DaysGap_diff']:.0f}")
                if 'CareerWinPct_diff' in row:
                    st.write(f"**CareerWinPct_diff:** {row['CareerWinPct_diff']:.1%}")
                if 'Prev7WinPct' in row:
                    st.write(f"**Prev7WinPct:** {row['Prev7WinPct']:.1%}")
                if 'FighterOddsNum' in row:
                    st.write(f"**FighterOddsNum:** {row['FighterOddsNum']:.0f}")
                if 'PrevFighterOddsNum' in row:
                    st.write(f"**PrevFighterOddsNum:** {row['PrevFighterOddsNum']:.0f}")
                if 'ColleyDecayDiff' in row:
                    st.write(f"**ColleyDecayDiff:** {row['ColleyDecayDiff']:.3f}")
                if 'MasseyDecayDiff' in row:
                    st.write(f"**MasseyDecayDiff:** {row['MasseyDecayDiff']:.3f}")
                if 'WeightedMasseyDecayDiff' in row:
                    st.write(f"**WeightedMasseyDecayDiff:** {row['WeightedMasseyDecayDiff']:.3f}")
                # Show some adjperf diffs for key stats
                key_stats = ['adjperf_KD', 'adjperf_SS', 'adjperf_TD', 'adjperf_Subs', 'adjperf_Ctrl']
                for ks in key_stats:
                    diff_col = f'{ks}_diff'
                    if diff_col in row:
                        st.write(f"**{diff_col}:** {row[diff_col]:.2f}")

            colA, colB = st.columns(2)
            with colA:
                show_fighter_stats(f1_row, f1_row['Fighter'])
            with colB:
                show_fighter_stats(f2_row, f2_row['Fighter'])

            st.subheader(f"Model Win Probabilities for {f1_row['Fighter']}")

            # LR
            lr_model = st.session_state.lr_model
            if lr_model is not None and st.session_state.x_lr is not None:
                def safe_val(row, col):
                    try:
                        val = row[col]
                        return val if pd.notna(val) else 0.0
                    except:
                        return 0.0
                v1 = safe_val(f1_row, st.session_state.x_lr)
                v2 = safe_val(f1_row, st.session_state.y_lr)
                v3 = safe_val(f1_row, st.session_state.z_lr)
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
                st.info("LR model not trained. Check status above.")

            # KNN
            calibrated_knn = st.session_state.calibrated_knn
            scaler = st.session_state.scaler
            X_train = st.session_state.X_train
            if calibrated_knn is not None and scaler is not None and st.session_state.x_knn is not None:
                means_knn = X_train.mean(axis=0) if X_train is not None else np.zeros(3)
                vals_knn = []
                for i, col_name in enumerate([st.session_state.x_knn, st.session_state.y_knn, st.session_state.z_knn]):
                    raw = get_first_col(pd.DataFrame(f1_row).T, col_name)[0]
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
                st.info("KNN model not trained. Check status above.")
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
                    v1 = f1_row[x_lr] if pd.notna(f1_row[x_lr]) else 0.0
                    v2 = f1_row[y_lr] if pd.notna(f1_row[y_lr]) else 0.0
                    v3 = f1_row[z_lr] if pd.notna(f1_row[z_lr]) else 0.0
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

        # LR combo builder
        st.subheader("LR 3‑Variable Combinations (Brier)")
        combo_candidates = [c for c in three_d_features if c != 'FighterOddsNum']
        importance_features = [c for c in three_d_features if not c.endswith('_diff') and not c.startswith('Opponent_')]
        @st.cache_data
        def numerical_importance(_data, features):
            hist = _data[_data['Win?'].isin(['Yes','No'])].copy()
            hist['Target'] = (hist['Win?'] == 'Yes').astype(int)
            X = hist[features].dropna()
            y = hist.loc[X.index, 'Target']
            if len(X) > 10:
                X_imp = SimpleImputer(strategy='median').fit_transform(X)
                mi = mutual_info_classif(X_imp, y, discrete_features=False)
                return pd.DataFrame({'Feature': features, 'Mutual Information': mi}).sort_values('Mutual Information', ascending=False).head(20)
            return pd.DataFrame()
        mi_df = numerical_importance(data, importance_features)
        top_feats = mi_df['Feature'].tolist() if not mi_df.empty else combo_candidates
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
            if st.button("Compute LR 3‑Var Combos", key="lr_combo_btn"):
                with st.spinner("Testing 3‑variable LR combos…"):
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
                            results.append({'Variables': ', '.join(combo), 'Brier': bs})
                        except:
                            pass
                    if results:
                        st.session_state.lr_combo_results = pd.DataFrame(results).sort_values('Brier').head(20)
                    else:
                        st.warning("Could not evaluate any combination.")
            if st.session_state.lr_combo_results is not None:
                st.write("**Top 20 3‑Variable Combinations (Brier)**")
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
                    means_knn = X_knn.mean(axis=0)
                    vals_knn = []
                    for col_name in [x_knn, y_knn, z_knn]:
                        raw = f1_row[col_name] if col_name in f1_row else np.nan
                        try:
                            v = float(raw) if pd.notna(raw) else means_knn[len(vals_knn)]
                        except:
                            v = means_knn[len(vals_knn)]
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

        # KNN combo builder
        st.subheader("KNN 3‑Variable Combinations (Brier, In‑Sample)")
        combo_candidates_knn = [c for c in three_d_features if c != 'FighterOddsNum']
        if not mi_df.empty:
            top_features_knn = mi_df['Feature'].tolist()
        else:
            top_features_knn = combo_candidates_knn
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
            if st.button("Compute KNN 3‑Var Combos (In‑Sample)", key="knn_combo_btn"):
                with st.spinner("Testing 3‑variable KNN combos (in‑sample)…"):
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
                            scaler_combo = StandardScaler()
                            X_scaled = scaler_combo.fit_transform(X)
                            base_knn_cv = KNeighborsClassifier(n_neighbors=k_combo, weights='distance')
                            calibrated = CalibratedClassifierCV(base_knn_cv, method='sigmoid', cv=5)
                            calibrated.fit(X_scaled, y_clean)
                            y_prob = calibrated.predict_proba(X_scaled)[:, 1]
                            y_prob = np.clip(y_prob, 0.1, 0.9)
                            bs = brier_score_loss(y_clean, y_prob)
                            results.append({'Variables': ', '.join(combo), 'Brier (In‑Sample)': bs})
                        except:
                            pass
                    if results:
                        st.session_state.knn_combo_results = pd.DataFrame(results).sort_values('Brier (In‑Sample)').head(20)
                    else:
                        st.warning("Could not evaluate any combination.")
            if st.session_state.knn_combo_results is not None:
                st.write("**Top 20 3‑Variable Combinations (Brier, In‑Sample)**")
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
# Add some columns if they exist
for col in ['Age','AgeDiff','HeightDiff','ReachDiff','CareerWinPct_diff','Prev7WinPct','ColleyDecayDiff','MasseyDecayDiff','WeightedMasseyDecayDiff']:
    if col in last20.columns:
        display_cols.append(col)
# Add some adjperf diffs
for ks in ['adjperf_KD', 'adjperf_SS', 'adjperf_TD']:
    if f'{ks}_diff' in last20.columns:
        display_cols.append(f'{ks}_diff')
display_cols = [c for c in display_cols if c in last20.columns]
st.dataframe(last20[display_cols])

# =========================================================================
# FEATURE IMPORTANCE
# =========================================================================
st.header("Top 20 Feature Importance (Current Filter Set)")
hist_imp = data[data['Win?'].isin(['Yes', 'No'])].copy()
if len(hist_imp) < 10:
    st.warning("Too few historical fights after filtering to compute importance.")
else:
    hist_imp['Target'] = (hist_imp['Win?'] == 'Yes').astype(int)
    eligible = [c for c in three_d_features if c in hist_imp.columns]
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
# SPIDER CHART (with column checks)
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

# We need Prev1_Title for spider – if it doesn't exist, skip that filter
# We'll also skip outcome filters if columns missing
spider_prev1_col = 'Prev1_Outcome_raw'
spider_prev2_col = 'Prev2_Outcome_raw'
spider_prev3_col = 'Prev3_Outcome_raw'
spider_career1_col = 'Career1_Outcome_raw'
spider_career2_col = 'Career2_Outcome_raw'
spider_career3_col = 'Career3_Outcome_raw'

# Only show outcome filters if columns exist
if all(c in original_data.columns for c in [spider_prev1_col, spider_prev2_col, spider_prev3_col,
                                            spider_career1_col, spider_career2_col, spider_career3_col]):
    all_outcomes_raw_spider = sorted(original_data[spider_prev1_col].dropna().unique())
    all_outcomes_career_spider = sorted(original_data[spider_career1_col].dropna().unique())
    with st.expander("Previous Outcomes (Spider)"):
        spider_prev1 = st.multiselect("Prev Fight 1", all_outcomes_raw_spider, key="spider_prev1")
        spider_prev2 = st.multiselect("Prev Fight 2", all_outcomes_raw_spider, key="spider_prev2")
        spider_prev3 = st.multiselect("Prev Fight 3", all_outcomes_raw_spider, key="spider_prev3")
        spider_career1 = st.multiselect("Career F1", all_outcomes_career_spider, key="spider_career1")
        spider_career2 = st.multiselect("Career F2", all_outcomes_career_spider, key="spider_career2")
        spider_career3 = st.multiselect("Career F3", all_outcomes_career_spider, key="spider_career3")
else:
    spider_prev1 = spider_prev2 = spider_prev3 = spider_career1 = spider_career2 = spider_career3 = []

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

if spider_prev1 and spider_prev1_col in spider_data.columns: mask &= spider_data[spider_prev1_col].isin(spider_prev1)
if spider_prev2 and spider_prev2_col in spider_data.columns: mask &= spider_data[spider_prev2_col].isin(spider_prev2)
if spider_prev3 and spider_prev3_col in spider_data.columns: mask &= spider_data[spider_prev3_col].isin(spider_prev3)
if spider_career1 and spider_career1_col in spider_data.columns: mask &= spider_data[spider_career1_col].isin(spider_career1)
if spider_career2 and spider_career2_col in spider_data.columns: mask &= spider_data[spider_career2_col].isin(spider_career2)
if spider_career3 and spider_career3_col in spider_data.columns: mask &= spider_data[spider_career3_col].isin(spider_career3)

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
        # Use only the new features for similarity (those that exist)
        sim_features = [c for c in new_features if c in spider_data.columns and c not in ['FightNumber', 'FightNumber_diff']]
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
