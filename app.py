import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import itertools
import gdown
import hashlib
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cdist

st.set_page_config(page_title="UFC Pre‑Fight Dashboard", layout="wide")

# -----------------------------------------------
# LOAD DATA
# -----------------------------------------------
PARQUET_FILE_ID = "1uIpfbGFmDolA8P2vc15VvA1qbNzWetxf"   # replace if needed

@st.cache_data
def load_data():
    gdown.download(f"https://drive.google.com/uc?id={PARQUET_FILE_ID}", "data.parquet", quiet=True)
    return pd.read_parquet("data.parquet")

data = load_data()

# Apply 2015+ filter
if 'FightDate' in data.columns:
    data = data[data['FightDate'] >= '2015-01-01'].copy()

original_data = data.copy()

# -----------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------
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

# -----------------------------------------------
# FEATURE LISTS
# -----------------------------------------------
adjperf_diff_cols = [c for c in data.columns if c.endswith('_diff') and c.startswith('adjperf_')]
base_cols = [
    'Age', 'AgeDiff', 'HeightDiff', 'ReachDiff',
    'DaysSincePrev', 'DaysSincePrev_diff', 'Avg3DaysGap_diff',
    'FightNumber', 'FightNumber_diff',
    'FighterOddsNum', 'PrevFighterOddsNum',
    'CareerWinPct_diff', 'Prev7WinPct',
    'FighterColleyDecay', 'OpponentColleyDecay', 'ColleyDecayDiff',
    'FighterMasseyDecay', 'OpponentMasseyDecay', 'MasseyDecayDiff',
    'FighterWeightedMasseyDecay', 'OpponentWeightedMasseyDecay', 'WeightedMasseyDecayDiff'
]

new_features = []
for col in base_cols:
    if col in data.columns:
        new_features.append(col)
for col in adjperf_diff_cols:
    if col in data.columns:
        new_features.append(col)
new_features = list(dict.fromkeys(new_features))  # unique, order preserved

three_d_features = [c for c in new_features if data[c].nunique(dropna=True) >= 2 and np.issubdtype(data[c].dtype, np.number)]

exclude_combo = [
    'CareerWinPct_diff', 'Prev7WinPct',
    'FighterColleyDecay', 'OpponentColleyDecay', 'ColleyDecayDiff',
    'FighterMasseyDecay', 'OpponentMasseyDecay', 'MasseyDecayDiff',
    'FighterWeightedMasseyDecay', 'OpponentWeightedMasseyDecay', 'WeightedMasseyDecayDiff'
]
combo_candidates = [c for c in new_features if c not in exclude_combo]

# -----------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------
for key, default in [
    ('lr_model', None), ('calibrated_knn', None), ('scaler', None),
    ('X_train', None), ('y_train_knn', None),
    ('overall_wr', 0.0), ('recent_wr', 0.0), ('recent_count', 0),
    ('lr_train_status', "Not trained"), ('knn_train_status', "Not trained"),
    ('selected_fight_row', None),
    ('lr_feature_names', []), ('knn_feature_names', []),
    ('x_lr', None), ('y_lr', None), ('z_lr', None),
    ('x_knn', None), ('y_knn', None), ('z_knn', None),
    ('knn_model_k', 5),
    ('lr_combo_results', None), ('knn_combo_results', None)
]:
    if key not in st.session_state:
        st.session_state[key] = default

if len(three_d_features) >= 3:
    if st.session_state.x_lr is None: st.session_state.x_lr = three_d_features[0]
    if st.session_state.y_lr is None: st.session_state.y_lr = three_d_features[1]
    if st.session_state.z_lr is None: st.session_state.z_lr = three_d_features[2]
    if st.session_state.x_knn is None: st.session_state.x_knn = three_d_features[0]
    if st.session_state.y_knn is None: st.session_state.y_knn = three_d_features[1]
    if st.session_state.z_knn is None: st.session_state.z_knn = three_d_features[2]

# -----------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------
st.sidebar.title("Filters")

with st.sidebar.expander("General", expanded=True):
    wc = st.multiselect("Weight Class", sorted(data['WC'].dropna().unique()), key="filter_wc") if 'WC' in data.columns else []
    stance = st.multiselect("Stance", sorted(data['Stance'].dropna().unique()), key="filter_stance") if 'Stance' in data.columns else []
    country = st.multiselect("Country", sorted(data['Country'].dropna().unique()), key="filter_country") if 'Country' in data.columns else []
    sched_rounds = st.multiselect("Scheduled Rounds", sorted(data['ScheduledRounds'].dropna().unique()), key="filter_sched") if 'ScheduledRounds' in data.columns else []
    title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"], key="filter_titlefight") if 'Title' in data.columns else "All"
    hometown_fighter = st.multiselect("Hometown (Fighter)", sorted(data['HometownFighter'].dropna().unique()), key="filter_hometown_fighter") if 'HometownFighter' in data.columns else []
    opp_hometown = st.multiselect("Opponent Hometown", sorted(data['Opponent_Hometown'].dropna().unique()), key="filter_opp_hometown") if 'Opponent_Hometown' in data.columns else []
    event_country = st.multiselect("Event Country", sorted(data['EventCountry'].dropna().unique()), key="filter_event") if 'EventCountry' in data.columns else []

with st.sidebar.expander("Fight Numbers", expanded=False):
    fn_min = st.number_input("Min Fight #", value=1, min_value=1, max_value=int(data['FightNumber'].max()), key="filter_fn_min") if 'FightNumber' in data.columns else 1
    fn_max = st.number_input("Max Fight #", value=int(data['FightNumber'].max()), key="filter_fn_max") if 'FightNumber' in data.columns else 1000
    ofn_min = st.number_input("Opp Min Fight #", value=1, key="filter_ofn_min") if 'Opponent_FightNumber' in data.columns else 1
    ofn_max = st.number_input("Opp Max Fight #", value=int(data['Opponent_FightNumber'].max()), key="filter_ofn_max") if 'Opponent_FightNumber' in data.columns else 1000

with st.sidebar.expander("Career Win % Diff", expanded=False):
    if 'CareerWinPct_diff' in data.columns:
        cwp_min = st.slider("Min Career Win % Diff", -100, 100, -100, step=5, key="filter_cwp_min")
        cwp_max = st.slider("Max Career Win % Diff", -100, 100, 100, step=5, key="filter_cwp_max")
    else:
        cwp_min, cwp_max = -100, 100

with st.sidebar.expander("Physical Attributes", expanded=False):
    age_min, age_max = st.slider("Age", int(data['Age'].min()), int(data['Age'].max()), (int(data['Age'].min()), int(data['Age'].max())), key="filter_age") if 'Age' in data.columns else (0,100)
    ad_min, ad_max = st.slider("Age Diff", int(data['AgeDiff'].min()), int(data['AgeDiff'].max()), (int(data['AgeDiff'].min()), int(data['AgeDiff'].max())), key="filter_age_diff") if 'AgeDiff' in data.columns else (-100,100)
    hd_min, hd_max = st.slider("Height Diff (in)", int(data['HeightDiff'].min()), int(data['HeightDiff'].max()), (int(data['HeightDiff'].min()), int(data['HeightDiff'].max())), key="filter_height_diff") if 'HeightDiff' in data.columns else (-50,50)
    rd_min, rd_max = st.slider("Reach Diff (in)", int(data['ReachDiff'].min()), int(data['ReachDiff'].max()), (int(data['ReachDiff'].min()), int(data['ReachDiff'].max())), key="filter_reach_diff") if 'ReachDiff' in data.columns else (-50,50)

with st.sidebar.expander("Days & Gaps", expanded=False):
    days_min, days_max = st.slider("Days Since Prev", int(data['DaysSincePrev'].min()), int(data['DaysSincePrev'].max()), (int(data['DaysSincePrev'].min()), int(data['DaysSincePrev'].max())), key="filter_days") if 'DaysSincePrev' in data.columns else (0,1000)
    ddiff_min, ddiff_max = st.slider("Days Since Prev Diff", int(data['DaysSincePrev_diff'].min()), int(data['DaysSincePrev_diff'].max()), (int(data['DaysSincePrev_diff'].min()), int(data['DaysSincePrev_diff'].max())), key="filter_days_diff") if 'DaysSincePrev_diff' in data.columns else (-1000,1000)
    avg3_min, avg3_max = st.slider("Avg3DaysGap Diff", int(data['Avg3DaysGap_diff'].min()), int(data['Avg3DaysGap_diff'].max()), (int(data['Avg3DaysGap_diff'].min()), int(data['Avg3DaysGap_diff'].max())), key="filter_avg3_diff") if 'Avg3DaysGap_diff' in data.columns else (-1000,1000)

with st.sidebar.expander("Odds", expanded=False):
    odds_min, odds_max = st.slider("Fighter Odds", int(data['FighterOddsNum'].min()), int(data['FighterOddsNum'].max()), (int(data['FighterOddsNum'].min()), int(data['FighterOddsNum'].max())), step=10, key="filter_cur_odds") if 'FighterOddsNum' in data.columns else (-1000,1000)
    podds_min, podds_max = st.slider("Prev Fighter Odds", int(data['PrevFighterOddsNum'].min()), int(data['PrevFighterOddsNum'].max()), (int(data['PrevFighterOddsNum'].min()), int(data['PrevFighterOddsNum'].max())), step=10, key="filter_prev_odds") if 'PrevFighterOddsNum' in data.columns else (-1000,1000)

skip_nc = st.sidebar.checkbox("Skip NC outcomes", key="filter_skip_nc")
if skip_nc:
    prev1_col = 'Prev1_Outcome_skipNC'; prev2_col = 'Prev2_Outcome_skipNC'; prev3_col = 'Prev3_Outcome_skipNC'
    career1_col = 'Career1_Outcome_skipNC'; career2_col = 'Career2_Outcome_skipNC'; career3_col = 'Career3_Outcome_skipNC'
    opp_career1_col = 'Opponent_Career1_Outcome_skipNC'; opp_career2_col = 'Opponent_Career2_Outcome_skipNC'; opp_career3_col = 'Opponent_Career3_Outcome_skipNC'
else:
    prev1_col = 'Prev1_Outcome_raw'; prev2_col = 'Prev2_Outcome_raw'; prev3_col = 'Prev3_Outcome_raw'
    career1_col = 'Career1_Outcome_raw'; career2_col = 'Career2_Outcome_raw'; career3_col = 'Career3_Outcome_raw'
    opp_career1_col = 'Opponent_Career1_Outcome_raw'; opp_career2_col = 'Opponent_Career2_Outcome_raw'; opp_career3_col = 'Opponent_Career3_Outcome_raw'

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

prior_weight = st.sidebar.slider("Bayesian prior weight", 0.0, 20.0, 5.0, step=0.5, key="prior_weight_global")
recent_window = st.sidebar.slider("Recent fights window", 1, 100, 50, key="recent_win_global")

# -----------------------------------------------
# BUILD FILTER MASK
# -----------------------------------------------
mask = pd.Series(True, index=data.index)

def add_filter(condition, keep_nan=False, col_name=None):
    if condition is None:
        return None
    if keep_nan and col_name and col_name in data.columns:
        return condition | data[col_name].isna()
    return condition

if wc: mask &= data['WC'].isin(wc)
if stance: mask &= data['Stance'].isin(stance)
if country: mask &= data['Country'].isin(country)
if sched_rounds: mask &= data['ScheduledRounds'].isin(sched_rounds)
if title_fight != "All": mask &= data['Title'] == title_fight
if hometown_fighter: mask &= data['HometownFighter'].isin(hometown_fighter)
if opp_hometown: mask &= data['Opponent_Hometown'].isin(opp_hometown)
if event_country: mask &= data['EventCountry'].isin(event_country)
if new_wc and 'IsNewWeightClass' in data.columns: mask &= data['IsNewWeightClass'] == True

if prev_title != "All" and 'Prev1_Title' in data.columns:
    mask &= normalize_title_col(data['Prev1_Title']) == prev_title.lower()
if opp_prev_title != "All" and 'Opponent_Prev1_Title' in data.columns:
    mask &= normalize_title_col(data['Opponent_Prev1_Title']) == opp_prev_title.lower()

if 'FightNumber' in data.columns:
    mask &= add_filter((data['FightNumber'] >= fn_min) & (data['FightNumber'] <= fn_max), keep_nan=True, col_name='FightNumber')
if 'Opponent_FightNumber' in data.columns:
    mask &= add_filter((data['Opponent_FightNumber'] >= ofn_min) & (data['Opponent_FightNumber'] <= ofn_max), keep_nan=True, col_name='Opponent_FightNumber')

if 'CareerWinPct_diff' in data.columns:
    mask &= add_filter((data['CareerWinPct_diff'] >= cwp_min) & (data['CareerWinPct_diff'] <= cwp_max), keep_nan=True, col_name='CareerWinPct_diff')

for col, (cmin, cmax) in [
    ('Age', (age_min, age_max)),
    ('AgeDiff', (ad_min, ad_max)),
    ('HeightDiff', (hd_min, hd_max)),
    ('ReachDiff', (rd_min, rd_max)),
    ('DaysSincePrev', (days_min, days_max)),
    ('DaysSincePrev_diff', (ddiff_min, ddiff_max)),
    ('Avg3DaysGap_diff', (avg3_min, avg3_max)),
    ('FighterOddsNum', (odds_min, odds_max)),
    ('PrevFighterOddsNum', (podds_min, podds_max))
]:
    if col in data.columns:
        mask &= add_filter((data[col] >= cmin) & (data[col] <= cmax), keep_nan=True, col_name=col)

for col, val in [(prev1_col, prev1), (prev2_col, prev2), (prev3_col, prev3),
                 (career1_col, career1), (career2_col, career2), (career3_col, career3)]:
    if val and col in data.columns:
        mask &= data[col].isin(val)

for shift, wlist in [(1, opp_prev1), (2, opp_prev2), (3, opp_prev3)]:
    col = f'Opponent_Prev{shift}_Outcome_raw'
    if wlist and col in data.columns:
        if skip_nc:
            col_use = f'Opponent_Prev{shift}_Outcome_skipNC'
            if col_use in data.columns:
                mask &= data[col_use].isin(wlist)
        else:
            mask &= data[col].isin(wlist)

for col, val in [(opp_career1_col, opp_career1), (opp_career2_col, opp_career2), (opp_career3_col, opp_career3)]:
    if val and col in data.columns:
        mask &= data[col].isin(val)

if use_colley and 'ColleyDecayDiff' in data.columns:
    mask &= add_filter((data['ColleyDecayDiff'] >= colley_range[0]) & (data['ColleyDecayDiff'] <= colley_range[1]), keep_nan=True, col_name='ColleyDecayDiff')
if use_massey and 'MasseyDecayDiff' in data.columns:
    mask &= add_filter((data['MasseyDecayDiff'] >= massey_range[0]) & (data['MasseyDecayDiff'] <= massey_range[1]), keep_nan=True, col_name='MasseyDecayDiff')
if use_wmd and 'WeightedMasseyDecayDiff' in data.columns:
    mask &= add_filter((data['WeightedMasseyDecayDiff'] >= wmd_range[0]) & (data['WeightedMasseyDecayDiff'] <= wmd_range[1]), keep_nan=True, col_name='WeightedMasseyDecayDiff')

filtered = data[mask].copy()
surviving_fight_ids = filtered['FightID'].unique()
matchup_data = original_data[original_data['FightID'].isin(surviving_fight_ids)]

# -----------------------------------------------
# FILTER STATUS
# -----------------------------------------------
st.write(f"**Filter status:** {len(filtered)} / {len(data)} rows ({len(filtered)/len(data)*100:.1f}%)  |  {len(surviving_fight_ids)} unique fights")
if len(filtered) == 0:
    st.warning("No data matches the selected filters.")
    st.stop()

# -----------------------------------------------
# DATA ENRICHMENT
# -----------------------------------------------
def detailed_result(row):
    win_raw = row.get('Win?')
    if pd.isna(win_raw) or str(win_raw).strip().lower() in ('', 'none', 'nan'):
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

filtered['DetailedResult'] = filtered.apply(detailed_result, axis=1)
filtered['Fight'] = filtered['Fighter'].astype(str) + ' vs ' + filtered['Opponent'].astype(str)

color_map = {
    'Win': 'green', 'Loss': 'red', 'Win by DQ': 'limegreen',
    'Loss by DQ': 'darkred', 'No Contest': 'purple', 'Upcoming': 'blue', 'Draw': 'gray'
}

# -----------------------------------------------
# CACHED MODEL TRAINING
# -----------------------------------------------
@st.cache_data(show_spinner="Training models...")
def train_models_cached(df, x_lr, y_lr, z_lr, x_knn, y_knn, z_knn, k_knn):
    """Train LR and KNN models and return a dict with results."""
    result = {
        'lr_model': None, 'lr_train_status': 'LR features not set.',
        'lr_feature_names': [],
        'calibrated_knn': None, 'scaler': None,
        'knn_train_status': 'KNN features not set.',
        'knn_feature_names': [],
        'X_train': None, 'y_train_knn': None,
        'y_train_lr': None, 'X_train_lr': None
    }

    # LR
    if x_lr and y_lr and z_lr and all(c in df.columns for c in [x_lr, y_lr, z_lr]):
        hist = df[df['Win?'].isin(['Yes','No'])].copy()
        sub = hist[[x_lr, y_lr, z_lr, 'Win?']].dropna()
        if len(sub) >= 10 and sub['Win?'].nunique() == 2:
            try:
                sub['target'] = (sub['Win?'] == 'Yes').astype(int)
                X = sub[[x_lr, y_lr, z_lr]].values
                y = sub['target'].values
                lr = LogisticRegression(max_iter=1000).fit(X, y)
                result['lr_model'] = lr
                result['lr_train_status'] = f"LR trained on {len(sub)} fights."
                result['y_train_lr'] = y
                result['X_train_lr'] = X
                result['lr_feature_names'] = [x_lr, y_lr, z_lr]
            except Exception as e:
                result['lr_train_status'] = f"LR error: {e}"
        else:
            result['lr_train_status'] = "LR needs ≥10 rows with both Win/Loss."

    # KNN
    if x_knn and y_knn and z_knn and all(c in df.columns for c in [x_knn, y_knn, z_knn]):
        hist = df[df['Win?'].isin(['Yes','No'])].copy()
        c1 = get_first_col(hist, x_knn); c2 = get_first_col(hist, y_knn); c3 = get_first_col(hist, z_knn)
        win_vals = (hist['Win?'] == 'Yes').values
        train_df = pd.DataFrame({'f1': c1, 'f2': c2, 'f3': c3, 'Win?': win_vals}).dropna()
        if len(train_df) >= 10 and train_df['Win?'].nunique() == 2:
            try:
                X = train_df[['f1','f2','f3']].values.astype(np.float64)
                y = train_df['Win?'].astype(int).values
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                base_knn = KNeighborsClassifier(n_neighbors=k_knn, weights='distance')
                calibrated = CalibratedClassifierCV(base_knn, method='sigmoid', cv=5).fit(X_scaled, y)
                result['calibrated_knn'] = calibrated
                result['scaler'] = scaler
                result['X_train'] = X
                result['y_train_knn'] = y
                result['knn_train_status'] = f"KNN trained on {len(train_df)} fights."
                result['knn_feature_names'] = [x_knn, y_knn, z_knn]
            except Exception as e:
                result['knn_train_status'] = f"KNN error: {e}"
        else:
            result['knn_train_status'] = "KNN needs ≥10 rows with both Win/Loss."

    return result

# Get current feature selections
x_lr = st.session_state.x_lr; y_lr = st.session_state.y_lr; z_lr = st.session_state.z_lr
x_knn = st.session_state.x_knn; y_knn = st.session_state.y_knn; z_knn = st.session_state.z_knn
k_knn = st.session_state.knn_model_k

# Train models (cached)
train_result = train_models_cached(filtered, x_lr, y_lr, z_lr, x_knn, y_knn, z_knn, k_knn)

# Update session state
st.session_state.lr_model = train_result['lr_model']
st.session_state.lr_train_status = train_result['lr_train_status']
st.session_state.lr_feature_names = train_result['lr_feature_names']
st.session_state.y_train_lr = train_result.get('y_train_lr')
st.session_state.X_train_lr = train_result.get('X_train_lr')

st.session_state.calibrated_knn = train_result['calibrated_knn']
st.session_state.scaler = train_result['scaler']
st.session_state.knn_train_status = train_result['knn_train_status']
st.session_state.knn_feature_names = train_result['knn_feature_names']
st.session_state.X_train = train_result['X_train']
st.session_state.y_train_knn = train_result['y_train_knn']

# Compute overall/recent win rates
hist_for_wr = filtered[filtered['Win?'].isin(['Yes','No'])].copy()
if len(hist_for_wr) > 0:
    st.session_state.overall_wr = (hist_for_wr['Win?'] == 'Yes').mean() * 100
    recent = hist_for_wr.sort_values('FightDate', ascending=False).head(recent_window)
    st.session_state.recent_wr = (recent['Win?'] == 'Yes').mean() * 100 if len(recent) > 0 else st.session_state.overall_wr
    st.session_state.recent_count = len(recent)
else:
    st.session_state.overall_wr = 0.0; st.session_state.recent_wr = 0.0; st.session_state.recent_count = 0

# -----------------------------------------------
# PERFORMANCE SUMMARY
# -----------------------------------------------
st.title("UFC Pre‑Fight Performance Dashboard")
st.header("Performance Summary")
total = len(filtered)
wins = (filtered['Win?'] == 'Yes').sum()
win_rate = wins / total * 100 if total > 0 else 0
col1, col2, col3 = st.columns(3)
col1.metric("Total Fights", total)
col2.metric("Wins", wins)
col3.metric("Win Rate", f"{win_rate:.1f}%")

# -----------------------------------------------
# UPCOMING FIGHT MATCHUP
# -----------------------------------------------
st.header("Upcoming Fight Matchup")
lr_status = st.session_state.lr_train_status
if lr_status and "error" not in lr_status.lower():
    st.success(f"✅ LR: {lr_status}")
else:
    st.error(f"❌ LR: {lr_status}")

knn_status = st.session_state.knn_train_status
if knn_status and "error" not in knn_status.lower():
    st.success(f"✅ KNN: {knn_status}")
else:
    st.error(f"❌ KNN: {knn_status}")

upcoming_display = matchup_data[matchup_data['Win?'].isna() | (matchup_data['Win?'] == '')]
st.write(f"**Upcoming fights after filters:** {len(upcoming_display['FightID'].unique())}")

if not upcoming_display.empty:
    upcoming_ids = sorted(upcoming_display['FightID'].unique())
    selected_fight = st.selectbox("Choose an upcoming fight", upcoming_ids, key="upcoming_select")
    if selected_fight:
        fight_rows = upcoming_display[upcoming_display['FightID'] == selected_fight]
        if len(fight_rows) == 2:
            f1 = fight_rows.iloc[0]; f2 = fight_rows.iloc[1]
            st.session_state.selected_fight_row = f1
            st.write(f"### {f1['Fighter']} vs {f2['Fighter']}")
            colA, colB = st.columns(2)
            with colA:
                st.subheader(f1['Fighter'])
                for c in ['Age','HeightDiff','ReachDiff','CareerWinPct_diff','Prev7WinPct','FighterOddsNum']:
                    if c in f1: st.write(f"**{c}:** {f1[c]:.2f}" if isinstance(f1[c], (int,float)) else f"**{c}:** {f1[c]}")
                adjperf_cols = [c for c in f1.index if c.startswith('adjperf_') and not c.endswith('_diff') and 'Opponent_' not in c]
                if adjperf_cols:
                    st.write("**Adjusted Performance**")
                    opp_adj = {c: f1.get(f'Opponent_{c}', np.nan) for c in adjperf_cols if f'Opponent_{c}' in f1.index}
                    df_adj = pd.DataFrame({'Fighter': [f1.get(c, np.nan) for c in adjperf_cols], 'Opponent': [opp_adj.get(c, np.nan) for c in adjperf_cols]}, index=adjperf_cols).T
                    st.dataframe(df_adj)
            with colB:
                st.subheader(f2['Fighter'])
                for c in ['Age','HeightDiff','ReachDiff','CareerWinPct_diff','Prev7WinPct','FighterOddsNum']:
                    if c in f2: st.write(f"**{c}:** {f2[c]:.2f}" if isinstance(f2[c], (int,float)) else f"**{c}:** {f2[c]}")
                adjperf_cols = [c for c in f2.index if c.startswith('adjperf_') and not c.endswith('_diff') and 'Opponent_' not in c]
                if adjperf_cols:
                    st.write("**Adjusted Performance**")
                    opp_adj = {c: f2.get(f'Opponent_{c}', np.nan) for c in adjperf_cols if f'Opponent_{c}' in f2.index}
                    df_adj = pd.DataFrame({'Fighter': [f2.get(c, np.nan) for c in adjperf_cols], 'Opponent': [opp_adj.get(c, np.nan) for c in adjperf_cols]}, index=adjperf_cols).T
                    st.dataframe(df_adj)

            st.subheader(f"Model Win Probabilities for {f1['Fighter']}")
            lr_model = st.session_state.lr_model; lr_feats = st.session_state.lr_feature_names
            if lr_model and len(lr_feats) == 3:
                vals = [f1[c] if c in f1 and pd.notna(f1[c]) else 0.0 for c in lr_feats]
                try:
                    prob = lr_model.predict_proba(np.array([vals]))[0,1]
                    shrunk = (prior_weight * st.session_state.overall_wr/100 + prob) / (prior_weight + 1)
                    st.write(f"**LR:** {prob:.1%} | Shrunken: {shrunk:.1%}")
                except Exception as e: st.error(f"LR prediction error: {e}")
            else: st.info("LR model not trained.")

            knn_model = st.session_state.calibrated_knn; scaler = st.session_state.scaler; knn_feats = st.session_state.knn_feature_names
            if knn_model and scaler and len(knn_feats) == 3:
                vals = [f1[c] if c in f1 and pd.notna(f1[c]) else 0.0 for c in knn_feats]
                try:
                    up_scaled = scaler.transform(np.array([vals]))
                    prob = np.clip(knn_model.predict_proba(up_scaled)[0,1], 0.1, 0.9)
                    shrunk = (prior_weight * st.session_state.overall_wr/100 + prob) / (prior_weight + 1)
                    st.write(f"**KNN:** {prob:.1%} | Shrunken: {shrunk:.1%}")
                except Exception as e: st.error(f"KNN prediction error: {e}")
            else: st.info("KNN model not trained.")
        else:
            st.warning("Fight data incomplete (expected 2 rows).")
else:
    st.info("No upcoming fights with current filters.")

# -----------------------------------------------
# 3D LR SCATTER & COMBO BUILDER
# -----------------------------------------------
st.header("3D LR Win/Loss Prediction & Best LR Combinations")
if len(three_d_features) >= 3:
    col1, col2, col3 = st.columns(3)
    with col1:
        x_lr = st.selectbox("X (LR)", three_d_features, index=three_d_features.index(st.session_state.x_lr) if st.session_state.x_lr in three_d_features else 0, key="lr_x")
    with col2:
        y_lr = st.selectbox("Y (LR)", three_d_features, index=three_d_features.index(st.session_state.y_lr) if st.session_state.y_lr in three_d_features else 1, key="lr_y")
    with col3:
        z_lr = st.selectbox("Z (LR)", three_d_features, index=three_d_features.index(st.session_state.z_lr) if st.session_state.z_lr in three_d_features else 2, key="lr_z")
    if (x_lr != st.session_state.x_lr or y_lr != st.session_state.y_lr or z_lr != st.session_state.z_lr):
        st.session_state.x_lr, st.session_state.y_lr, st.session_state.z_lr = x_lr, y_lr, z_lr
        # Will retrain via cache automatically on next run
        st.rerun()

    plot_data = filtered[[x_lr, y_lr, z_lr, 'DetailedResult', 'Fight']].dropna()
    if len(plot_data) >= 10:
        fig = px.scatter_3d(plot_data, x=x_lr, y=y_lr, z=z_lr, color='DetailedResult', color_discrete_map=color_map, hover_data=['Fight'], title="LR 3D Scatter")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Not enough data for 3D LR plot.")

    # LR metrics
    if st.session_state.lr_model and hasattr(st.session_state, 'y_train_lr') and st.session_state.y_train_lr is not None:
        X_lr = st.session_state.X_train_lr; y_true = st.session_state.y_train_lr
        if X_lr is not None and len(X_lr) > 0:
            y_prob = st.session_state.lr_model.predict_proba(X_lr)[:,1]
            ll = log_loss(y_true, y_prob); bs = brier_score_loss(y_true, y_prob)
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("LR Log‑loss", f"{ll:.3f}")
            col_m2.metric("LR Brier", f"{bs:.3f}")
            col_m3.metric("Overall Win%", f"{st.session_state.overall_wr:.1f}%")

    # LR combo builder
    st.subheader("LR 3‑Variable Combinations (Cross‑Validated Brier)")
    candidates = [c for c in combo_candidates if c in data.columns and c != 'FighterOddsNum']
    if len(candidates) >= 3:
        @st.cache_data
        def numerical_importance(_data, features):
            hist = _data[_data['Win?'].isin(['Yes','No'])].copy()
            hist['Target'] = (hist['Win?'] == 'Yes').astype(int)
            X = hist[features].dropna(); y = hist.loc[X.index, 'Target']
            if len(X) > 10:
                X_imp = SimpleImputer(strategy='median').fit_transform(X)
                mi = mutual_info_classif(X_imp, y, discrete_features=False)
                return pd.DataFrame({'Feature': features, 'MI': mi}).sort_values('MI', ascending=False).head(20)
            return pd.DataFrame()
        mi_df = numerical_importance(data, candidates)
        top_feats = mi_df['Feature'].tolist() if not mi_df.empty else candidates
        num_top = st.slider("Top features to test", 5, min(30, len(top_feats)), 10, key="lr_combo_top")
        candidates = top_feats[:num_top]

        if st.button("Compute LR Combos (Cross‑Validated)", key="lr_combo_btn"):
            with st.spinner("Computing..."):
                hist = data[data['Win?'].isin(['Yes','No'])].copy()
                hist['WinNum'] = (hist['Win?'] == 'Yes').astype(int)
                results = []
                for combo in itertools.combinations(candidates, 3):
                    sub = hist[list(combo) + ['WinNum']].dropna()
                    if len(sub) < 10 or sub['WinNum'].nunique() < 2: continue
                    X = sub[list(combo)].values; y = sub['WinNum'].values
                    try:
                        lr = LogisticRegression(max_iter=1000)
                        y_prob = cross_val_predict(lr, X, y, cv=5, method='predict_proba')[:,1]
                        results.append({'Variables': ', '.join(combo), 'CV Brier': brier_score_loss(y, y_prob)})
                    except: pass
                if results:
                    st.session_state.lr_combo_results = pd.DataFrame(results).sort_values('CV Brier').head(20)
                else: st.warning("No combinations evaluated.")
        if st.session_state.lr_combo_results is not None:
            st.dataframe(st.session_state.lr_combo_results, use_container_width=True)
else:
    st.warning("Need at least 3 numeric features for 3D LR plot.")

# -----------------------------------------------
# 3D KNN SCATTER & COMBO BUILDER
# -----------------------------------------------
st.header("3D Weighted KNN Win/Loss Prediction & Best KNN Combinations")
if len(three_d_features) >= 3:
    col1k, col2k, col3k = st.columns(3)
    with col1k:
        x_knn = st.selectbox("X (KNN)", three_d_features, index=three_d_features.index(st.session_state.x_knn) if st.session_state.x_knn in three_d_features else 0, key="knn_x")
    with col2k:
        y_knn = st.selectbox("Y (KNN)", three_d_features, index=three_d_features.index(st.session_state.y_knn) if st.session_state.y_knn in three_d_features else 1, key="knn_y")
    with col3k:
        z_knn = st.selectbox("Z (KNN)", three_d_features, index=three_d_features.index(st.session_state.z_knn) if st.session_state.z_knn in three_d_features else 2, key="knn_z")
    if (x_knn != st.session_state.x_knn or y_knn != st.session_state.y_knn or z_knn != st.session_state.z_knn):
        st.session_state.x_knn, st.session_state.y_knn, st.session_state.z_knn = x_knn, y_knn, z_knn
        st.rerun()

    plot_data_knn = filtered[[x_knn, y_knn, z_knn, 'DetailedResult', 'Fight']].dropna()
    if len(plot_data_knn) >= 10:
        fig_knn = px.scatter_3d(plot_data_knn, x=x_knn, y=y_knn, z=z_knn, color='DetailedResult', color_discrete_map=color_map, hover_data=['Fight'], title="KNN 3D Scatter")
        st.plotly_chart(fig_knn, use_container_width=True)
    else:
        st.warning("Not enough data for 3D KNN plot.")

    k_knn = st.slider("KNN neighbors", 1, 20, st.session_state.knn_model_k, key="knn_slider")
    if k_knn != st.session_state.knn_model_k:
        st.session_state.knn_model_k = k_knn
        st.rerun()

    # KNN combo builder
    st.subheader("KNN 3‑Variable Combinations (Cross‑Validated Brier)")
    candidates_knn = [c for c in combo_candidates if c in data.columns and c != 'FighterOddsNum']
    if len(candidates_knn) >= 3:
        k_combo = st.slider("KNN neighbors (combo)", 1, 20, 5, key="knn_combo_k")
        if st.button("Compute KNN Combos (Cross‑Validated)", key="knn_combo_btn"):
            with st.spinner("Computing..."):
                hist_combo = data[data['Win?'].isin(['Yes','No'])].copy()
                hist_combo['WinNum'] = (hist_combo['Win?'] == 'Yes').astype(int)
                results = []
                for combo in itertools.combinations(candidates_knn, 3):
                    c1 = get_first_col(hist_combo, combo[0]); c2 = get_first_col(hist_combo, combo[1]); c3 = get_first_col(hist_combo, combo[2])
                    y = hist_combo['WinNum'].values
                    mask_combo = ~(np.isnan(c1) | np.isnan(c2) | np.isnan(c3))
                    if mask_combo.sum() < 10 or np.unique(y[mask_combo]).size < 2: continue
                    X = np.column_stack([c1[mask_combo], c2[mask_combo], c3[mask_combo]]); y_clean = y[mask_combo]
                    try:
                        pipeline = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k_combo, weights='distance'))])
                        y_prob = cross_val_predict(pipeline, X, y_clean, cv=5, method='predict_proba')[:,1]
                        results.append({'Variables': ', '.join(combo), 'CV Brier': brier_score_loss(y_clean, y_prob)})
                    except: pass
                if results:
                    st.session_state.knn_combo_results = pd.DataFrame(results).sort_values('CV Brier').head(20)
                else: st.warning("No combinations evaluated.")
        if st.session_state.knn_combo_results is not None:
            st.dataframe(st.session_state.knn_combo_results, use_container_width=True)
else:
    st.warning("Need at least 3 numeric features for KNN plot.")

# -----------------------------------------------
# LAST 20 FIGHTS
# -----------------------------------------------
st.header("Last 20 Fights")
last20 = filtered.sort_values('FightDate', ascending=False).head(20)
cols = ['FightDate','Fighter','Opponent','Win?','Method','AgeDiff','HeightDiff','ReachDiff','CareerWinPct_diff']
cols = [c for c in cols if c in last20.columns]
st.dataframe(last20[cols], use_container_width=True)

# -----------------------------------------------
# FEATURE IMPORTANCE
# -----------------------------------------------
st.header("Top 20 Feature Importance")
hist_imp = filtered[filtered['Win?'].isin(['Yes','No'])].copy()
if len(hist_imp) >= 10:
    hist_imp['Target'] = (hist_imp['Win?'] == 'Yes').astype(int)
    feats = [c for c in three_d_features if c in hist_imp.columns]
    if feats:
        X = hist_imp[feats].dropna()
        if len(X) >= 10:
            imp = mutual_info_classif(X, hist_imp.loc[X.index, 'Target'], discrete_features=False, random_state=42)
            df_imp = pd.DataFrame({'Feature': feats, 'Mutual Information': imp}).sort_values('Mutual Information', ascending=False).head(20)
            fig = px.bar(df_imp, x='Mutual Information', y='Feature', orientation='h', title="Top 20 Mutual Information")
            st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Not enough complete rows for importance.")
    else: st.warning("No numeric features.")
else: st.warning("Too few historical fights for importance.")

# -----------------------------------------------
# FIGHT SIMILARITY (INDEPENDENT FILTERS)
# -----------------------------------------------
st.header("Fight Similarity (Independent Filters)")
st.write("These filters are separate from the main sidebar and do not affect the dashboard above.")
with st.expander("Similarity Filters", expanded=True):
    col_sp1, col_sp2 = st.columns(2)
    with col_sp1:
        spider_wc = st.multiselect("Weight Class", sorted(original_data['WC'].dropna().unique()), key="spider_wc") if 'WC' in original_data.columns else []
        spider_stance = st.multiselect("Stance", sorted(original_data['Stance'].dropna().unique()), key="spider_stance") if 'Stance' in original_data.columns else []
        spider_country = st.multiselect("Country", sorted(original_data['Country'].dropna().unique()), key="spider_country") if 'Country' in original_data.columns else []
        spider_sched_rounds = st.multiselect("Scheduled Rounds", sorted(original_data['ScheduledRounds'].dropna().unique()), key="spider_sched") if 'ScheduledRounds' in original_data.columns else []
        spider_event_country = st.multiselect("Event Country", sorted(original_data['EventCountry'].dropna().unique()), key="spider_eventc") if 'EventCountry' in original_data.columns else []
    with col_sp2:
        spider_title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"], key="spider_title") if 'Title' in original_data.columns else "All"
        spider_hometown_fighter = st.multiselect("Hometown (Fighter)", sorted(original_data['HometownFighter'].dropna().unique()), key="spider_hometown_fighter") if 'HometownFighter' in original_data.columns else []
        spider_opp_hometown = st.multiselect("Hometown (Opponent)", sorted(original_data['Opponent_Hometown'].dropna().unique()), key="spider_opp_hometown") if 'Opponent_Hometown' in original_data.columns else []
        spider_new_wc = st.checkbox("New Weight Class", key="spider_new_wc") if 'IsNewWeightClass' in original_data.columns else False
        spider_prev_title = st.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"], key="spider_prev_title")

    spider_prev1 = st.multiselect("Prev Fight 1", all_outcomes_raw, key="spider_prev1")
    spider_prev2 = st.multiselect("Prev Fight 2", all_outcomes_raw, key="spider_prev2")
    spider_prev3 = st.multiselect("Prev Fight 3", all_outcomes_raw, key="spider_prev3")
    spider_career1 = st.multiselect("Career F1", all_outcomes_career, key="spider_career1")
    spider_career2 = st.multiselect("Career F2", all_outcomes_career, key="spider_career2")
    spider_career3 = st.multiselect("Career F3", all_outcomes_career, key="spider_career3")

spider_mask = pd.Series(True, index=original_data.index)
if spider_wc: spider_mask &= original_data['WC'].isin(spider_wc)
if spider_stance: spider_mask &= original_data['Stance'].isin(spider_stance)
if spider_country: spider_mask &= original_data['Country'].isin(spider_country)
if spider_sched_rounds: spider_mask &= original_data['ScheduledRounds'].isin(spider_sched_rounds)
if spider_title_fight != "All": spider_mask &= original_data['Title'] == spider_title_fight
if spider_hometown_fighter: spider_mask &= original_data['HometownFighter'].isin(spider_hometown_fighter)
if spider_opp_hometown: spider_mask &= original_data['Opponent_Hometown'].isin(spider_opp_hometown)
if spider_event_country: spider_mask &= original_data['EventCountry'].isin(spider_event_country)
if spider_new_wc and 'IsNewWeightClass' in original_data.columns: spider_mask &= original_data['IsNewWeightClass'] == True
if spider_prev_title != "All" and 'Prev1_Title' in original_data.columns:
    spider_mask &= normalize_title_col(original_data['Prev1_Title']) == spider_prev_title.lower()
if spider_prev1: spider_mask &= original_data['Prev1_Outcome_raw'].isin(spider_prev1)
if spider_prev2: spider_mask &= original_data['Prev2_Outcome_raw'].isin(spider_prev2)
if spider_prev3: spider_mask &= original_data['Prev3_Outcome_raw'].isin(spider_prev3)
if spider_career1: spider_mask &= original_data['Career1_Outcome_raw'].isin(spider_career1)
if spider_career2: spider_mask &= original_data['Career2_Outcome_raw'].isin(spider_career2)
if spider_career3: spider_mask &= original_data['Career3_Outcome_raw'].isin(spider_career3)

spider_filtered = original_data[spider_mask].copy()
spider_fight_ids = spider_filtered['FightID'].unique()
spider_data = original_data[original_data['FightID'].isin(spider_fight_ids)]

spider_upcoming = spider_data[spider_data['Win?'].isna() | (spider_data['Win?'] == '')]
spider_hist = spider_data[spider_data['Win?'].isin(['Yes','No'])].copy()

if spider_upcoming.empty:
    st.write("No upcoming fights for similarity.")
else:
    fight_counts = spider_upcoming.groupby('FightID').size()
    complete_ids = fight_counts[fight_counts == 2].index
    spider_upcoming = spider_upcoming[spider_upcoming['FightID'].isin(complete_ids)]
    if spider_upcoming.empty:
        st.warning("No upcoming fight has both fighters after similarity filters.")
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
                        f1 = fight_rows.iloc[0]; f2 = fight_rows.iloc[1]
                        st.write(f"### {f1['Fighter']} vs {f2['Fighter']}")

                        up_vals = [float(f1.get(var, 0.0)) for var in selected_vars]
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
                        n_top = st.slider("Number of top similar fights", 5, 100, 50, step=5, key="spider_top_n")
                        top_n = sim_df.head(n_top)
                        count = len(top_n); avg_sim = top_n['Similarity'].mean(); total_sim = top_n['Similarity'].sum()
                        composite = avg_sim * (count ** 0.5) / 100
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Count", count); col2.metric("Avg Similarity", f"{avg_sim:.1f}%")
                        col3.metric("Total Similarity", f"{total_sim:.1f}"); col4.metric("Composite Score", f"{composite:.1f}")

                        high_sim = top_n[top_n['Similarity'] >= 90]
                        if len(high_sim) > 0:
                            wins_high = (high_sim['Win?'] == 'Yes').sum()
                            st.metric("Win Rate (≥90% sim)", f"{wins_high/len(high_sim)*100:.1f}%", delta=f"{len(high_sim)} fights")
                        else:
                            st.write("No historical fights with similarity ≥ 90%.")

                        fig_hist = px.histogram(sim_df, x='Similarity', nbins=20, title="Similarity Distribution")
                        st.plotly_chart(fig_hist, use_container_width=True)
                        st.subheader(f"Top {n_top} Most Similar Historical Fights")
                        st.dataframe(top_n, use_container_width=True)
