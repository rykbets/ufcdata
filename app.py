import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gdown
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist

st.set_page_config(page_title="UFC Pre‑Fight Dashboard", layout="wide")

# -----------------------------------------------
# LOAD DATA
# -----------------------------------------------
PARQUET_FILE_ID = "1uIpfbGFmDolA8P2vc15VvA1qbNzWetxf"   # <-- update with your file ID

@st.cache_data
def load_data():
    gdown.download(f"https://drive.google.com/uc?id={PARQUET_FILE_ID}", "data.parquet", quiet=True)
    df = pd.read_parquet("data.parquet")
    required_cols = ['FightID', 'Fighter', 'Opponent', 'FightDate', 'Win?', 'Age', 'Height', 'Reach', 'WC']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}")
    return df

try:
    data = load_data()
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

if 'FightDate' in data.columns:
    data = data[data['FightDate'] >= '2015-01-01'].copy()
original_data = data.copy()

# Helper functions
def normalize_title_col(series):
    if series is None: return pd.Series('', index=series.index)
    return series.astype(str).str.strip().str.lower()

def get_diff_range(df, col_name):
    if col_name not in df.columns: return -1.0, 1.0
    vals = df[col_name].dropna()
    if len(vals) == 0: return -1.0, 1.0
    return float(vals.min()), float(vals.max())

# Rating columns for display and spider chart
rating_raw_cols = [
    'FighterColleyDecay', 'OpponentColleyDecay', 'ColleyDecayDiff',
    'FighterMasseyFinishDecay', 'OpponentMasseyFinishDecay', 'MasseyFinishDecayDiff',
    'FighterMasseyStrikeDecay', 'OpponentMasseyStrikeDecay', 'MasseyStrikeDecayDiff',
    'FighterMasseyCtrlDecay', 'OpponentMasseyCtrlDecay', 'MasseyCtrlDecayDiff',
    'FighterWeightedMasseyDecay', 'OpponentWeightedMasseyDecay', 'WeightedMasseyDecayDiff'
]
rating_avg7_cols = [
    'FighterColleyDecay_avg7', 'Opponent_FighterColleyDecay_avg7', 'FighterColleyDecay_avg7_diff',
    'FighterMasseyFinishDecay_avg7', 'Opponent_FighterMasseyFinishDecay_avg7', 'FighterMasseyFinishDecay_avg7_diff',
    'FighterMasseyStrikeDecay_avg7', 'Opponent_FighterMasseyStrikeDecay_avg7', 'FighterMasseyStrikeDecay_avg7_diff',
    'FighterMasseyCtrlDecay_avg7', 'Opponent_FighterMasseyCtrlDecay_avg7', 'FighterMasseyCtrlDecay_avg7_diff',
    'FighterWeightedMasseyDecay_avg7', 'Opponent_FighterWeightedMasseyDecay_avg7', 'FighterWeightedMasseyDecay_avg7_diff'
]

# Feature lists for filters and similarity (spider will use only rating diffs)
base_cols = [c for c in data.columns if c not in ['FightID','Fighter','Opponent','FightDate','Win?','Method','Round',
                                                    'DetailedResult','Fight','FightDurationMinutes']]
new_features = list(dict.fromkeys(base_cols))

# For spider chart: only rating differentials
spider_rating_diff_cols = [c for c in rating_raw_cols if c.endswith('Diff')] + \
                          [c for c in rating_avg7_cols if c.endswith('_diff')]

# Session state (minimal, no models)
for key, default in [
    ('overall_wr', 0.0), ('recent_wr', 0.0), ('recent_count', 0),
]:
    if key not in st.session_state:
        st.session_state[key] = default

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
    if 'Opponent_FightNumber' in data.columns:
        ofn_min = st.number_input("Opp Min Fight #", value=1, key="filter_ofn_min")
        ofn_max = st.number_input("Opp Max Fight #", value=int(data['Opponent_FightNumber'].max()), key="filter_ofn_max")
    else:
        ofn_min = st.number_input("Opp Min Fight #", value=1, disabled=True, key="filter_ofn_min_disabled")
        ofn_max = st.number_input("Opp Max Fight #", value=1000, disabled=True, key="filter_ofn_max_disabled")
        st.warning("'Opponent_FightNumber' column missing. Re‑run data processing script.")

with st.sidebar.expander("Career Win % Diff", expanded=False):
    if 'CareerWinPct_diff' in data.columns:
        cwp_min, cwp_max = st.slider("Career Win % Diff", -100, 100, (-100, 100), step=5, key="filter_cwp")
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
    use_massey = st.checkbox("Filter MasseyFinishDecayDiff", value=False, key="filter_use_massey")
    if use_massey:
        min_md, max_md = get_diff_range(data, 'MasseyFinishDecayDiff')
        massey_range = st.slider("MasseyFinishDecayDiff range", min_md, max_md, (min_md, max_md), step=0.01, key="filter_massey")
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
# BUILD MAIN FILTER MASK
# -----------------------------------------------
mask = pd.Series(True, index=data.index)
def add_filter(condition, keep_nan=False, col_name=None):
    if condition is None: return None
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
if use_massey and 'MasseyFinishDecayDiff' in data.columns:
    mask &= add_filter((data['MasseyFinishDecayDiff'] >= massey_range[0]) & (data['MasseyFinishDecayDiff'] <= massey_range[1]), keep_nan=True, col_name='MasseyFinishDecayDiff')
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
    if win_val in ('No Contest', 'NC'): return 'No Contest'
    if win_val == 'Draw': return 'Draw'
    if win_val == 'Yes': return 'Win'
    if win_val == 'No': return 'Loss'
    return 'Upcoming'

filtered['DetailedResult'] = filtered.apply(detailed_result, axis=1)
filtered['Fight'] = filtered['Fighter'].astype(str) + ' vs ' + filtered['Opponent'].astype(str)
color_map = {
    'Win': 'green', 'Loss': 'red', 'Win by DQ': 'limegreen',
    'Loss by DQ': 'darkred', 'No Contest': 'purple', 'Upcoming': 'blue', 'Draw': 'gray'
}

# Win rates (for shrinkage if needed later)
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
col1.metric("Total Fights", total); col2.metric("Wins", wins); col3.metric("Win Rate", f"{win_rate:.1f}%")

# -----------------------------------------------
# LAST 20 FIGHTS (moved up)
# -----------------------------------------------
st.header("Last 20 Fights")
last20 = filtered.sort_values('FightDate', ascending=False).head(20)
cols = ['FightDate','Fighter','Opponent','Win?','Method','AgeDiff','HeightDiff','ReachDiff','CareerWinPct_diff']
cols = [c for c in cols if c in last20.columns]
st.dataframe(last20[cols], use_container_width=True)

# -----------------------------------------------
# UPCOMING FIGHT MATCHUP (FULL TABLE + TOP 5 DIFFS)
# -----------------------------------------------
st.header("Upcoming Fight Matchup")

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

            # ---- BUILD TABLE SECTIONS ----
            sections = {}

            identity_cols = ['WC','Title','ScheduledRounds','Stance','Country','HometownFighter','EventCountry']
            sections["Identity"] = [c for c in identity_cols if c in f1.index]

            physical_cols = ['Age','Height','Reach','AgeDiff','HeightDiff','ReachDiff']
            sections["Physical"] = [c for c in physical_cols if c in f1.index]

            fight_hist_cols = ['FightNumber','DaysSincePrev','Avg3DaysGap','Prev7WinPct','CareerWinPct',
                               'DaysSincePrev_diff','Avg3DaysGap_diff','CareerWinPct_diff','FightNumber_diff']
            sections["Fight History"] = [c for c in fight_hist_cols if c in f1.index]

            sections["Normalized Simple Stats (diff)"] = [
                c for c in f1.index if c.startswith('adj_') and c.endswith('_diff')
            ]

            odds_cols = ['FighterOddsNum','PrevFighterOddsNum']
            sections["Odds"] = [c for c in odds_cols if c in f1.index]

            sections["Ratings (Raw)"] = [c for c in f1.index if ('Colley' in c or 'Massey' in c) and 'avg7' not in c]
            sections["Ratings (7‑Fight Avg)"] = [c for c in f1.index if 'avg7' in c]

            sections["Striking & Grappling Final Differentials"] = [
                c for c in f1.index if c.endswith('_opp_diff')
            ]

            sections["Outcomes"] = [c for c in f1.index if 'Outcome' in c]

            other_cols = ['Prev1_Title','IsNewWeightClass','PrevFighterOddsNum']
            sections["Other"] = [c for c in other_cols if c in f1.index]

            rows = []
            for sec_name, cols in sections.items():
                if not cols:
                    continue
                rows.append({"Stat": f"--- {sec_name} ---", f1['Fighter']: "", f2['Fighter']: ""})
                for c in cols:
                    val1 = f1[c]
                    val2 = f2[c]
                    def fmt(v):
                        if isinstance(v, (int, float)) and pd.notna(v):
                            return f"{v:.2f}"
                        elif pd.isna(v):
                            return ""
                        else:
                            return str(v)
                    rows.append({"Stat": c, f1['Fighter']: fmt(val1), f2['Fighter']: fmt(val2)})

            df_stats = pd.DataFrame(rows)
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

            # Top 5 Differentials (signed)
            st.subheader("Top 5 Differentials")
            for fighter, row in [(f1['Fighter'], f1), (f2['Fighter'], f2)]:
                diffs = {}
                for c in row.index:
                    if (c.endswith('_opp_diff') or (c.startswith('adj_') and c.endswith('_diff'))):
                        val = row[c]
                        if pd.notna(val):
                            diffs[c] = val
                top5 = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:5]
                if top5:
                    st.write(f"**{fighter}**")
                    for col, val in top5:
                        st.write(f"{col}: {val:+.2f}" if isinstance(val, float) else f"{col}: {val}")
                else:
                    st.write(f"**{fighter}**: No eligible differentials available.")
        else:
            st.warning("Fight data incomplete (expected 2 rows).")
else:
    st.info("No upcoming fights with current filters.")

# -----------------------------------------------
# FIGHT SIMILARITY (INDEPENDENT FILTERS, RATING DIFFS ONLY, NO COMBO BUILDER)
# -----------------------------------------------
st.header("Fight Similarity (Independent Filters)")
st.write("These filters are separate from the main sidebar and do not affect the dashboard above.")

with st.expander("Similarity Filters", expanded=True):
    with st.expander("General", expanded=True):
        spider_wc = st.multiselect("Weight Class", sorted(original_data['WC'].dropna().unique()), key="spider_wc") if 'WC' in original_data.columns else []
        spider_stance = st.multiselect("Stance", sorted(original_data['Stance'].dropna().unique()), key="spider_stance") if 'Stance' in original_data.columns else []
        spider_country = st.multiselect("Country", sorted(original_data['Country'].dropna().unique()), key="spider_country") if 'Country' in original_data.columns else []
        spider_sched_rounds = st.multiselect("Scheduled Rounds", sorted(original_data['ScheduledRounds'].dropna().unique()), key="spider_sched") if 'ScheduledRounds' in original_data.columns else []
        spider_title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"], key="spider_title") if 'Title' in original_data.columns else "All"
        spider_hometown_fighter = st.multiselect("Hometown (Fighter)", sorted(original_data['HometownFighter'].dropna().unique()), key="spider_hometown_fighter") if 'HometownFighter' in original_data.columns else []
        spider_opp_hometown = st.multiselect("Opponent Hometown", sorted(original_data['Opponent_Hometown'].dropna().unique()), key="spider_opp_hometown") if 'Opponent_Hometown' in original_data.columns else []
        spider_event_country = st.multiselect("Event Country", sorted(original_data['EventCountry'].dropna().unique()), key="spider_eventc") if 'EventCountry' in original_data.columns else []

    with st.expander("Physical Attributes & Fight Numbers", expanded=False):
        spider_fn_min = st.number_input("Min Fight #", value=1, min_value=1, max_value=int(original_data['FightNumber'].max()), key="spider_fn_min") if 'FightNumber' in original_data.columns else 1
        spider_fn_max = st.number_input("Max Fight #", value=int(original_data['FightNumber'].max()), key="spider_fn_max") if 'FightNumber' in original_data.columns else 1000
        spider_ofn_min = st.number_input("Opp Min Fight #", value=1, key="spider_ofn_min") if 'Opponent_FightNumber' in original_data.columns else 1
        spider_ofn_max = st.number_input("Opp Max Fight #", value=int(original_data['Opponent_FightNumber'].max()), key="spider_ofn_max") if 'Opponent_FightNumber' in original_data.columns else 1000
        spider_cwp_min, spider_cwp_max = st.slider("Career Win % Diff", -100, 100, (-100, 100), step=5, key="spider_cwp") if 'CareerWinPct_diff' in original_data.columns else (-100,100)
        spider_age_min, spider_age_max = st.slider("Age", int(original_data['Age'].min()), int(original_data['Age'].max()), (int(original_data['Age'].min()), int(original_data['Age'].max())), key="spider_age") if 'Age' in original_data.columns else (0,100)
        spider_ad_min, spider_ad_max = st.slider("Age Diff", int(original_data['AgeDiff'].min()), int(original_data['AgeDiff'].max()), (int(original_data['AgeDiff'].min()), int(original_data['AgeDiff'].max())), key="spider_age_diff") if 'AgeDiff' in original_data.columns else (-100,100)
        spider_hd_min, spider_hd_max = st.slider("Height Diff", int(original_data['HeightDiff'].min()), int(original_data['HeightDiff'].max()), (int(original_data['HeightDiff'].min()), int(original_data['HeightDiff'].max())), key="spider_height_diff") if 'HeightDiff' in original_data.columns else (-50,50)
        spider_rd_min, spider_rd_max = st.slider("Reach Diff", int(original_data['ReachDiff'].min()), int(original_data['ReachDiff'].max()), (int(original_data['ReachDiff'].min()), int(original_data['ReachDiff'].max())), key="spider_reach_diff") if 'ReachDiff' in original_data.columns else (-50,50)
        spider_days_min, spider_days_max = st.slider("Days Since Prev", int(original_data['DaysSincePrev'].min()), int(original_data['DaysSincePrev'].max()), (int(original_data['DaysSincePrev'].min()), int(original_data['DaysSincePrev'].max())), key="spider_days") if 'DaysSincePrev' in original_data.columns else (0,1000)
        spider_ddiff_min, spider_ddiff_max = st.slider("Days Since Prev Diff", int(original_data['DaysSincePrev_diff'].min()), int(original_data['DaysSincePrev_diff'].max()), (int(original_data['DaysSincePrev_diff'].min()), int(original_data['DaysSincePrev_diff'].max())), key="spider_days_diff") if 'DaysSincePrev_diff' in original_data.columns else (-1000,1000)
        spider_avg3_min, spider_avg3_max = st.slider("Avg3DaysGap Diff", int(original_data['Avg3DaysGap_diff'].min()), int(original_data['Avg3DaysGap_diff'].max()), (int(original_data['Avg3DaysGap_diff'].min()), int(original_data['Avg3DaysGap_diff'].max())), key="spider_avg3_diff") if 'Avg3DaysGap_diff' in original_data.columns else (-1000,1000)

    with st.expander("Odds", expanded=False):
        spider_odds_min, spider_odds_max = st.slider("Fighter Odds", int(original_data['FighterOddsNum'].min()), int(original_data['FighterOddsNum'].max()), (int(original_data['FighterOddsNum'].min()), int(original_data['FighterOddsNum'].max())), step=10, key="spider_cur_odds") if 'FighterOddsNum' in original_data.columns else (-1000,1000)
        spider_podds_min, spider_podds_max = st.slider("Prev Fighter Odds", int(original_data['PrevFighterOddsNum'].min()), int(original_data['PrevFighterOddsNum'].max()), (int(original_data['PrevFighterOddsNum'].min()), int(original_data['PrevFighterOddsNum'].max())), step=10, key="spider_prev_odds") if 'PrevFighterOddsNum' in original_data.columns else (-1000,1000)

    with st.expander("Previous Outcomes", expanded=False):
        spider_skip_nc = st.checkbox("Skip NC outcomes", key="spider_skip_nc")
        if spider_skip_nc:
            spider_prev1_col = 'Prev1_Outcome_skipNC'; spider_prev2_col = 'Prev2_Outcome_skipNC'; spider_prev3_col = 'Prev3_Outcome_skipNC'
            spider_career1_col = 'Career1_Outcome_skipNC'; spider_career2_col = 'Career2_Outcome_skipNC'; spider_career3_col = 'Career3_Outcome_skipNC'
        else:
            spider_prev1_col = 'Prev1_Outcome_raw'; spider_prev2_col = 'Prev2_Outcome_raw'; spider_prev3_col = 'Prev3_Outcome_raw'
            spider_career1_col = 'Career1_Outcome_raw'; spider_career2_col = 'Career2_Outcome_raw'; spider_career3_col = 'Career3_Outcome_raw'

        spider_prev1 = st.multiselect("Prev Fight 1", all_outcomes_raw, key="spider_prev1")
        spider_prev2 = st.multiselect("Prev Fight 2", all_outcomes_raw, key="spider_prev2")
        spider_prev3 = st.multiselect("Prev Fight 3", all_outcomes_raw, key="spider_prev3")
        spider_career1 = st.multiselect("Career F1", all_outcomes_career, key="spider_career1")
        spider_career2 = st.multiselect("Career F2", all_outcomes_career, key="spider_career2")
        spider_career3 = st.multiselect("Career F3", all_outcomes_career, key="spider_career3")

        # Opponent previous outcomes
        spider_opp_prev1 = st.multiselect("Opp Prev 1", all_outcomes_raw, key="spider_opp_prev1")
        spider_opp_prev2 = st.multiselect("Opp Prev 2", all_outcomes_raw, key="spider_opp_prev2")
        spider_opp_prev3 = st.multiselect("Opp Prev 3", all_outcomes_raw, key="spider_opp_prev3")

        # Opponent career outcomes
        spider_opp_career1 = st.multiselect("Opp Career F1", all_outcomes_career, key="spider_opp_career1")
        spider_opp_career2 = st.multiselect("Opp Career F2", all_outcomes_career, key="spider_opp_career2")
        spider_opp_career3 = st.multiselect("Opp Career F3", all_outcomes_career, key="spider_opp_career3")

    with st.expander("Other", expanded=False):
        spider_prev_title = st.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"], key="spider_prev_title")
        spider_opp_prev_title = st.selectbox("Opp Prev Fight Was Title?", ["All", "Yes", "No"], key="spider_opp_prev_title")
        spider_new_wc = st.checkbox("New Weight Class", key="spider_new_wc") if 'IsNewWeightClass' in original_data.columns else False

    with st.expander("Ratings", expanded=False):
        spider_use_colley = st.checkbox("Filter ColleyDecayDiff", value=False, key="spider_use_colley")
        if spider_use_colley:
            min_cd, max_cd = get_diff_range(original_data, 'ColleyDecayDiff')
            spider_colley_range = st.slider("ColleyDecayDiff range", min_cd, max_cd, (min_cd, max_cd), step=0.01, key="spider_colley")
        spider_use_massey = st.checkbox("Filter MasseyFinishDecayDiff", value=False, key="spider_use_massey")
        if spider_use_massey:
            min_md, max_md = get_diff_range(original_data, 'MasseyFinishDecayDiff')
            spider_massey_range = st.slider("MasseyFinishDecayDiff range", min_md, max_md, (min_md, max_md), step=0.01, key="spider_massey")
        spider_use_wmd = st.checkbox("Filter WeightedMasseyDecayDiff", value=False, key="spider_use_wmd")
        if spider_use_wmd:
            min_wmd, max_wmd = get_diff_range(original_data, 'WeightedMasseyDecayDiff')
            spider_wmd_range = st.slider("WeightedMasseyDecayDiff range", min_wmd, max_wmd, (min_wmd, max_wmd), step=0.01, key="spider_wmd")

# Build spider mask
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
if spider_opp_prev_title != "All" and 'Opponent_Prev1_Title' in original_data.columns:
    spider_mask &= normalize_title_col(original_data['Opponent_Prev1_Title']) == spider_opp_prev_title.lower()

def spider_add_filter(condition, col_name):
    if condition is None: return None
    if col_name in original_data.columns:
        return condition | original_data[col_name].isna()
    return condition

if 'FightNumber' in original_data.columns:
    spider_mask &= spider_add_filter((original_data['FightNumber'] >= spider_fn_min) & (original_data['FightNumber'] <= spider_fn_max), 'FightNumber')
if 'Opponent_FightNumber' in original_data.columns:
    spider_mask &= spider_add_filter((original_data['Opponent_FightNumber'] >= spider_ofn_min) & (original_data['Opponent_FightNumber'] <= spider_ofn_max), 'Opponent_FightNumber')
if 'CareerWinPct_diff' in original_data.columns:
    spider_mask &= spider_add_filter((original_data['CareerWinPct_diff'] >= spider_cwp_min) & (original_data['CareerWinPct_diff'] <= spider_cwp_max), 'CareerWinPct_diff')

for col, (cmin, cmax) in [
    ('Age', (spider_age_min, spider_age_max)),
    ('AgeDiff', (spider_ad_min, spider_ad_max)),
    ('HeightDiff', (spider_hd_min, spider_hd_max)),
    ('ReachDiff', (spider_rd_min, spider_rd_max)),
    ('DaysSincePrev', (spider_days_min, spider_days_max)),
    ('DaysSincePrev_diff', (spider_ddiff_min, spider_ddiff_max)),
    ('Avg3DaysGap_diff', (spider_avg3_min, spider_avg3_max)),
    ('FighterOddsNum', (spider_odds_min, spider_odds_max)),
    ('PrevFighterOddsNum', (spider_podds_min, spider_podds_max))
]:
    if col in original_data.columns:
        spider_mask &= spider_add_filter((original_data[col] >= cmin) & (original_data[col] <= cmax), col)

for col, val in [(spider_prev1_col, spider_prev1), (spider_prev2_col, spider_prev2), (spider_prev3_col, spider_prev3),
                 (spider_career1_col, spider_career1), (spider_career2_col, spider_career2), (spider_career3_col, spider_career3)]:
    if val and col in original_data.columns:
        spider_mask &= original_data[col].isin(val)

# Opponent shifted previous outcomes
for shift, wlist in [(1, spider_opp_prev1), (2, spider_opp_prev2), (3, spider_opp_prev3)]:
    col = f'Opponent_Prev{shift}_Outcome_raw'
    if wlist and col in original_data.columns:
        if spider_skip_nc:
            col_use = f'Opponent_Prev{shift}_Outcome_skipNC'
            if col_use in original_data.columns:
                spider_mask &= original_data[col_use].isin(wlist)
        else:
            spider_mask &= original_data[col].isin(wlist)

# Opponent career outcomes
for col, val in [('Opponent_Career1_Outcome_raw', spider_opp_career1),
                 ('Opponent_Career2_Outcome_raw', spider_opp_career2),
                 ('Opponent_Career3_Outcome_raw', spider_opp_career3)]:
    if val and col in original_data.columns:
        spider_mask &= original_data[col].isin(val)

if spider_use_colley and 'ColleyDecayDiff' in original_data.columns:
    spider_mask &= spider_add_filter((original_data['ColleyDecayDiff'] >= spider_colley_range[0]) & (original_data['ColleyDecayDiff'] <= spider_colley_range[1]), 'ColleyDecayDiff')
if spider_use_massey and 'MasseyFinishDecayDiff' in original_data.columns:
    spider_mask &= spider_add_filter((original_data['MasseyFinishDecayDiff'] >= spider_massey_range[0]) & (original_data['MasseyFinishDecayDiff'] <= spider_massey_range[1]), 'MasseyFinishDecayDiff')
if spider_use_wmd and 'WeightedMasseyDecayDiff' in original_data.columns:
    spider_mask &= spider_add_filter((original_data['WeightedMasseyDecayDiff'] >= spider_wmd_range[0]) & (original_data['WeightedMasseyDecayDiff'] <= spider_wmd_range[1]), 'WeightedMasseyDecayDiff')

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
        # Only rating differentials for similarity
        sim_features = [c for c in spider_rating_diff_cols if c in spider_data.columns]
        if not sim_features:
            st.warning("No rating differential features available.")
        else:
            selected_vars = st.multiselect("Select rating differentials for similarity", sim_features, default=sim_features[:5], max_selections=8, key="spider_vars")
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

                        total_hist_count = len(sim_df)
                        st.metric("Total historical fights matching filters", total_hist_count)

                        st.subheader("Similarity Metrics (Top N)")
                        n_top = st.slider("Number of top similar fights", 5, 100, 50, step=5, key="spider_top_n")
                        top_n = sim_df.head(n_top)
                        count = len(top_n); avg_sim = top_n['Similarity'].mean(); total_sim = top_n['Similarity'].sum()
                        composite = avg_sim * (count ** 0.5) / 100
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Count (Top N)", count); col2.metric("Avg Similarity", f"{avg_sim:.1f}%")
                        col3.metric("Total Similarity", f"{total_sim:.1f}"); col4.metric("Composite Score", f"{composite:.1f}")

                        # 90% similarity metrics
                        high_sim_90 = top_n[top_n['Similarity'] >= 90]
                        if len(high_sim_90) > 0:
                            wins_90 = (high_sim_90['Win?'] == 'Yes').sum()
                            win_rate_90 = wins_90 / len(high_sim_90) * 100
                            # weighted win rate
                            if wins_90 > 0:
                                weight_sum_wins = high_sim_90.loc[high_sim_90['Win?'] == 'Yes', 'Similarity'].sum()
                                weight_sum_all = high_sim_90['Similarity'].sum()
                                weighted_wr_90 = (weight_sum_wins / weight_sum_all) * 100 if weight_sum_all > 0 else 0.0
                            else:
                                weighted_wr_90 = 0.0
                            st.metric("Win Rate (≥90% sim)", f"{win_rate_90:.1f}%",
                                      delta=f"{len(high_sim_90)} fights")
                            st.metric("Weighted Win Rate (≥90% sim)", f"{weighted_wr_90:.1f}%")
                        else:
                            st.write("No historical fights with similarity ≥ 90% in the top selection.")

                        # 80% similarity metrics
                        high_sim_80 = top_n[top_n['Similarity'] >= 80]
                        if len(high_sim_80) > 0:
                            wins_80 = (high_sim_80['Win?'] == 'Yes').sum()
                            win_rate_80 = wins_80 / len(high_sim_80) * 100
                            if wins_80 > 0:
                                weight_sum_wins_80 = high_sim_80.loc[high_sim_80['Win?'] == 'Yes', 'Similarity'].sum()
                                weight_sum_all_80 = high_sim_80['Similarity'].sum()
                                weighted_wr_80 = (weight_sum_wins_80 / weight_sum_all_80) * 100 if weight_sum_all_80 > 0 else 0.0
                            else:
                                weighted_wr_80 = 0.0
                            st.metric("Win Rate (≥80% sim)", f"{win_rate_80:.1f}%",
                                      delta=f"{len(high_sim_80)} fights")
                            st.metric("Weighted Win Rate (≥80% sim)", f"{weighted_wr_80:.1f}%")
                        else:
                            st.write("No historical fights with similarity ≥ 80% in the top selection.")

                        fig_hist = px.histogram(sim_df, x='Similarity', nbins=20, title="Similarity Distribution (All)")
                        st.plotly_chart(fig_hist, use_container_width=True, key="sim_hist_chart")

                        st.subheader(f"Top {n_top} Most Similar Historical Fights")
                        st.dataframe(top_n, use_container_width=True)

# -----------------------------------------------
# FEATURE IMPORTANCE (MOVED TO BOTTOM)
# -----------------------------------------------
st.header("Top 20 Feature Importance & Global Model Ranking")
hist_imp = filtered[filtered['Win?'].isin(['Yes','No'])].copy()
if len(hist_imp) < 10:
    st.warning("Too few historical fights after filtering to compute importance.")
else:
    hist_imp['Target'] = (hist_imp['Win?'] == 'Yes').astype(int)
    # Use all numeric features that are in the data (from new_features, excluding internal)
    feats = [c for c in new_features if pd.api.types.is_numeric_dtype(data[c]) and c in hist_imp.columns]
    if feats:
        X_mi = hist_imp[feats].dropna()
        if len(X_mi) >= 10:
            imputer = SimpleImputer(strategy='median')
            X_imp = imputer.fit_transform(X_mi)
            y_mi = hist_imp.loc[X_mi.index, 'Target']
            mi = mutual_info_classif(X_imp, y_mi, discrete_features=False, random_state=42)
            mi_df = pd.DataFrame({'Feature': feats, 'MI': mi}).sort_values('MI', ascending=False).head(20)
            fig_mi = px.bar(mi_df, x='MI', y='Feature', orientation='h',
                            title="Top 20 Mutual Information")
            st.plotly_chart(fig_mi, use_container_width=True, key="mi_plot")
        else:
            st.warning("Not enough complete rows for MI.")

        if st.button("Compute Lasso Importance (all features)"):
            with st.spinner("Fitting LassoCV..."):
                X_lasso = hist_imp[feats].copy(); y_lasso = hist_imp['Target']
                imp = SimpleImputer(strategy='median')
                X_lasso_imp = imp.fit_transform(X_lasso)
                scaler_lasso = StandardScaler()
                X_lasso_scaled = scaler_lasso.fit_transform(X_lasso_imp)
                lasso = LogisticRegressionCV(
                    penalty='l1', solver='saga', cv=5,
                    scoring='neg_brier_score', max_iter=2000,
                    Cs=10, n_jobs=-1, random_state=42
                )
                lasso.fit(X_lasso_scaled, y_lasso)
                coef = lasso.coef_.flatten()
                coef_df = pd.DataFrame({'Feature': feats, 'Coefficient': coef})
                coef_df = coef_df[coef_df['Coefficient'] != 0].sort_values('Coefficient', key=abs, ascending=False)
                st.subheader("Lasso Non‑Zero Coefficients")
                if len(coef_df) > 0:
                    fig_lasso = px.bar(coef_df.head(30), x='Coefficient', y='Feature', orientation='h',
                                       title="Lasso Coefficients")
                    st.plotly_chart(fig_lasso, use_container_width=True, key="lasso_plot")
                else:
                    st.write("Lasso eliminated all features.")

        if st.button("Compute Random Forest Importance (all features)"):
            with st.spinner("Training Random Forest..."):
                X_rf = hist_imp[feats].copy(); y_rf = hist_imp['Target']
                imp = SimpleImputer(strategy='median')
                X_rf_imp = imp.fit_transform(X_rf)
                rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
                rf.fit(X_rf_imp, y_rf)
                rf_imp = pd.DataFrame({'Feature': feats, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False).head(30)
                st.subheader("Random Forest Feature Importance (Gini)")
                fig_rf = px.bar(rf_imp, x='Importance', y='Feature', orientation='h',
                                title="Random Forest Feature Importance")
                st.plotly_chart(fig_rf, use_container_width=True, key="rf_plot")
    else:
        st.warning("No numeric features.")
