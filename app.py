import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import gdown
import streamlit as st
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier

# ----------------------------- Configuration -----------------------------
PARQUET_FILE_ID = "1uIpfbGFmDolA8P2vc15VvA1qbNzWetxf"
SAVED_SEARCHES_FILE = "saved_searches.json"

# ----------------------------- Data Loading -----------------------------
@st.cache_data
def load_data():
    if os.path.exists("data.parquet"):
        os.remove("data.parquet")
    gdown.download(f"https://drive.google.com/uc?id={PARQUET_FILE_ID}", "data.parquet", quiet=True)
    df = pd.read_parquet("data.parquet")
    required_cols = ['FightID', 'Fighter', 'Opponent', 'FightDate', 'Win?', 'Age', 'Height', 'Reach', 'WC']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}")
    df = df[df['FightDate'] >= '2015-01-01'].copy()
    df['FightDate'] = pd.to_datetime(df['FightDate'])
    df['Win?'] = df['Win?'].replace('', np.nan)
    return df

df_all = load_data()

# ----------------------------- Feature Groups -----------------------------
SLIDER_FEATURES_SPEC = [
    ('Age', 'Age'), ('AgeDiff', 'AgeDiff'),
    ('MasseyStrikeDecayDiff', 'MasseyStrikeDecayDiff'),
    ('Prev7WinPct_diff', 'Prev7WinPct_diff'),
    ('Prev7WinPct', 'Prev7WinPct'),
    ('MasseyCtrlDecayDiff', 'MasseyCtrlDecayDiff'),
    ('MasseyFinishDecayDiff', 'MasseyFinishDecayDiff'),
    ('CareerWinPct', 'CareerWinPct'),
    ('CareerWinPct_diff', 'CareerWinPct_diff'),
    ('ReachDiff', 'ReachDiff'),
    ('DaysSincePrev', 'DaysSincePrev'),
    ('ColleyDecayDiff', 'ColleyDecayDiff')
]
SLIDER_COLUMNS = [col for col, _ in SLIDER_FEATURES_SPEC if col in df_all.columns]
SLIDER_LABELS = [label for col, label in SLIDER_FEATURES_SPEC if col in df_all.columns]

slider_min_max = {}
for col in SLIDER_COLUMNS:
    mn, mx = df_all[col].min(), df_all[col].max()
    if pd.isna(mn) or pd.isna(mx):
        mn, mx = 0, 1
    slider_min_max[col] = (mn, mx)

ABS_MAPPING = {}
for col in df_all.columns:
    if col.startswith('Abs_'):
        orig = col[4:]
        if orig in df_all.columns:
            ABS_MAPPING[orig] = col

# ----------------------------- Exclusions (pattern-based) -----------------------------
raw_stats = ['KD','SS','SSA','TS','TSA','TD','TDA','Subs','Reversals',
             'HSL','HSA','BSL','BSA','LSL','LSA','DSL','DSA','CSL','CSA','GSL','GSA','Ctrl']
exclude_set = set()
exclude_set.update([
    'FightID','Fighter','Opponent','FightDate','Win?','Method','Round','WC','Stance','Country',
    'EventCountry','HometownFighter','Opponent_Hometown','ScheduledRounds','Title','Prev1_Title',
    'Prev2_Title','Prev3_Title','Opponent_Prev1_Title','FightNumber','TotalTimeSec',
    'FighterOddsNum','OpponentOddsNum','PrevFighterOddsNum',
    'KD_per_SS','Sub_per_Ctrl','SubWin_per_Ctrl','Ctrl_per_TD',
    'SS%_TS','DS%_SS','CS%_SS','GS%_SS','HS%_SS','BS%_SS','LS%_SS',
    'Completed3Rounds','FightDurationMinutes',
])
for stat in raw_stats:
    exclude_set.add(stat)
    exclude_set.add(f'Def_{stat}')
    exclude_set.add(f'ratio_off_{stat}')
    exclude_set.add(f'R1_ratio_off_{stat}')
    exclude_set.add(f'adjperf_ratio_{stat}')
    exclude_set.add(f'R1_adjperf_ratio_{stat}')
    exclude_set.add(f'Def_adjperf_ratio_{stat}')
    exclude_set.add(f'R1_Def_adjperf_ratio_{stat}')
    exclude_set.add(f'log_{stat}')
    exclude_set.add(f'adjperf_log_{stat}')
    exclude_set.add(f'Def_adjperf_log_{stat}')
    exclude_set.add(f'R1_log_{stat}')
    exclude_set.add(f'R1_adjperf_log_{stat}')
    exclude_set.add(f'R1_Def_adjperf_log_{stat}')
for method in ['KO','Sub','Dec']:
    exclude_set.add(f'WinBy{method}')
    exclude_set.add(f'LossBy{method}')
for surv in ['Survived1R','Survived15','Survived2R','Survived3R']:
    exclude_set.add(surv)
    exclude_set.add(f'FinishedOpp{surv[-2:]}')
for prefix in ['Prev1_','Prev2_','Prev3_']:
    exclude_set.add(prefix+'AgeDiff')
    exclude_set.add(prefix+'ReachDiff')
    exclude_set.add(prefix+'HeightDiff')
    exclude_set.add('Abs_'+prefix+'AgeDiff')
    exclude_set.add('Abs_'+prefix+'ReachDiff')
    exclude_set.add('Abs_'+prefix+'HeightDiff')

numeric_cols = df_all.select_dtypes(include=[np.number]).columns.tolist()
FEATURES_WINNER = [c for c in numeric_cols if c not in exclude_set]

# ----------------------------- Helper Functions -----------------------------
# (Same as before – all filtering/metrics functions)

# ----------------------------- Saved Search Management -----------------------------
# (Same as before)

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(layout="wide")
st.title("UFC Spider Filter Dashboard")

# Sidebar with saved searches, target, metrics (same as previous)
with st.sidebar:
    st.header("Saved Searches")
    search_name = st.text_input("Search Name", key="search_name")
    col1, col2, col3 = st.columns(3)
    if col1.button("Save"):
        if search_name:
            st.session_state.saved_searches[search_name] = {'tab': 'spider', 'params': get_params_from_widgets()}
            save_saved_searches(st.session_state.saved_searches)
            st.experimental_rerun()
    if col2.button("Load"):
        selected = st.session_state.get('saved_search_select', None)
        if selected and selected in st.session_state.saved_searches:
            apply_params_to_widgets(st.session_state.saved_searches[selected]['params'])
            st.experimental_rerun()
    if col3.button("Delete"):
        selected = st.session_state.get('saved_search_select', None)
        if selected and selected in st.session_state.saved_searches:
            del st.session_state.saved_searches[selected]
            save_saved_searches(st.session_state.saved_searches)
            st.experimental_rerun()
    rename_input = st.text_input("Rename to", key="rename_input")
    if st.button("Rename"):
        selected = st.session_state.get('saved_search_select', None)
        new_name = rename_input.strip()
        if selected and new_name and selected in st.session_state.saved_searches:
            st.session_state.saved_searches[new_name] = st.session_state.saved_searches.pop(selected)
            save_saved_searches(st.session_state.saved_searches)
            st.experimental_rerun()
    saved_search_select = st.selectbox("Saved Searches", options=sorted(st.session_state.saved_searches.keys()), key='saved_search_select')
    st.markdown("---")
    st.header("Global Target")
    target = st.selectbox("Target", options=["win", "complete3rds", "complete1.5rounds"], key='target')
    lambda_penalty = st.number_input("Penalty λ", value=0.1, step=0.05, key='lambda_penalty')

    # Metrics in sidebar
    st.markdown("---")
    st.subheader("Metrics")
    params = get_params_from_widgets()

    @st.cache_data
    def filter_data(params_json, target):
        params = json.loads(params_json)
        return apply_spider_filters(params, target)

    params_json = json.dumps(params, sort_keys=True)
    df_filtered = filter_data(params_json, target)

    if df_filtered.empty:
        st.write("No fights match.")
    else:
        adv = compute_advanced_metrics(df_filtered, df_all.copy(), target, params, lambda_penalty)
        n_fights, n_apps, metric_val, ci_low, ci_high, p_val, flag, z = compute_metrics(df_filtered, target)
        if target == 'win':
            st.write(f"Unique Fights: {n_fights}, Appearances (Side A): {n_apps}, Win Rate: {metric_val:.1%}, Z‑score: {z:.2f}")
        else:
            metric_name = 'Completion 3 Rounds' if target=='complete3rds' else 'Completion 1.5 Rounds'
            st.write(f"Unique Fights: {n_fights}, {metric_name}: {metric_val:.1%}, Z‑score: {z:.2f}")
        if adv:
            st.write(f"Eff: {adv['efficiency']:.6f} | |z|‑λ·k penalty: {adv['penalty_score']:.2f} (k={adv['k']})")

# Main area
st.subheader("Shared Filters")
col1, col2, col3, col4, col5 = st.columns(5)
wc = col1.multiselect("Weight Class", options=sorted(df_all['WC'].dropna().unique()), key='wc')
event_country = col2.multiselect("Event Country", options=sorted(df_all['EventCountry'].dropna().unique()), key='event_country')
title_fight = col3.selectbox("Title Fight", options=["All","Yes","No"], key='title_fight')
sched_rounds = col4.multiselect("Scheduled Rounds", options=sorted(df_all['ScheduledRounds'].dropna().unique()), key='sched_rounds')
new_wc = col5.checkbox("New Weight Class", key='new_wc')

# Side A
st.subheader("Side A Criteria")
with st.container():
    cols = st.columns(6)
    fa_country = cols[0].multiselect("Country A", options=sorted(df_all['Country'].dropna().unique()), key='fa_country')
    fa_stance = cols[1].multiselect("Stance A", options=sorted(df_all['Stance'].dropna().unique()), key='fa_stance')
    fa_hometown = cols[2].multiselect("Hometown A", options=sorted(df_all['HometownFighter'].dropna().unique()), key='fa_hometown')
    fa_fn_min = cols[3].number_input("Min Fight # A", value=1, key='fa_fn_min')
    fa_fn_max = cols[4].number_input("Max Fight # A", value=int(df_all['FightNumber'].max()), key='fa_fn_max')
    fa_prev_title = cols[5].selectbox("Prev Title A", options=["All","Yes","No"], key='fa_prev_title')

    st.markdown("**Previous Outcomes**")
    cols = st.columns(6)
    fa_prev1 = cols[0].multiselect("Prev 1 A", options=sorted(df_all['Prev1_Outcome_raw'].dropna().unique()), key='fa_prev1')
    fa_prev2 = cols[1].multiselect("Prev 2 A", options=sorted(df_all['Prev2_Outcome_raw'].dropna().unique()), key='fa_prev2')
    fa_prev3 = cols[2].multiselect("Prev 3 A", options=sorted(df_all['Prev3_Outcome_raw'].dropna().unique()), key='fa_prev3')
    fa_career1 = cols[3].multiselect("Career F1 A", options=sorted(df_all['Career1_Outcome_raw'].dropna().unique()), key='fa_career1')
    fa_career2 = cols[4].multiselect("Career F2 A", options=sorted(df_all['Career2_Outcome_raw'].dropna().unique()), key='fa_career2')
    fa_career3 = cols[5].multiselect("Career F3 A", options=sorted(df_all['Career3_Outcome_raw'].dropna().unique()), key='fa_career3')

    st.markdown("**Continuous Filters A**")
    for col, label in zip(SLIDER_COLUMNS, SLIDER_LABELS):
        mn, mx = slider_min_max[col]
        # Ensure range state exists
        range_key = f'fa_{col}_range'
        if range_key not in st.session_state:
            st.session_state[range_key] = (mn, mx)

        # Manual min/max inputs (these update range_key on change)
        c1, c2 = st.columns(2)
        min_val = c1.number_input(f"{label} A Min", min_value=mn, max_value=mx,
                                  value=st.session_state[range_key][0],
                                  key=f'fa_{col}_min')
        max_val = c2.number_input(f"{label} A Max", min_value=mn, max_value=mx,
                                  value=st.session_state[range_key][1],
                                  key=f'fa_{col}_max')
        if (min_val, max_val) != st.session_state[range_key]:
            st.session_state[range_key] = (min_val, max_val)

        # Slider (uses range_key, and after moving we update range_key)
        slider_val = st.slider(f"{label} A", min_value=mn, max_value=mx,
                               value=st.session_state[range_key],
                               key=f'fa_{col}_slider')
        if slider_val != st.session_state[range_key]:
            st.session_state[range_key] = slider_val
            # Also update min/max input states to reflect slider move
            st.session_state[f'fa_{col}_min'] = slider_val[0]
            st.session_state[f'fa_{col}_max'] = slider_val[1]

    st.markdown("**Dynamic Sliders A**")
    for i in range(3):
        feat = st.selectbox(f"Dyn Feat {i+1} A", options=["None"]+FEATURES_WINNER, key=f'fa_dyn_{i}_feat')
        if feat == "None":
            mn, mx = 0, 1
        else:
            mn, mx = df_all[feat].min(), df_all[feat].max()

        # Reset range if feature changed
        if st.session_state.get(f'fa_dyn_{i}_last_feat') != feat:
            st.session_state[f'fa_dyn_{i}_range'] = (mn, mx)
            st.session_state[f'fa_dyn_{i}_last_feat'] = feat
        if f'fa_dyn_{i}_range' not in st.session_state:
            st.session_state[f'fa_dyn_{i}_range'] = (mn, mx)

        c1, c2 = st.columns(2)
        min_val = c1.number_input(f"Min {i+1} A", min_value=mn, max_value=mx,
                                  value=st.session_state[f'fa_dyn_{i}_range'][0],
                                  key=f'fa_dyn_{i}_min')
        max_val = c2.number_input(f"Max {i+1} A", min_value=mn, max_value=mx,
                                  value=st.session_state[f'fa_dyn_{i}_range'][1],
                                  key=f'fa_dyn_{i}_max')
        if (min_val, max_val) != st.session_state[f'fa_dyn_{i}_range']:
            st.session_state[f'fa_dyn_{i}_range'] = (min_val, max_val)

        slider_val = st.slider(f"Dyn Range {i+1} A", min_value=mn, max_value=mx,
                               value=st.session_state[f'fa_dyn_{i}_range'],
                               key=f'fa_dyn_{i}_slider')
        if slider_val != st.session_state[f'fa_dyn_{i}_range']:
            st.session_state[f'fa_dyn_{i}_range'] = slider_val
            st.session_state[f'fa_dyn_{i}_min'] = slider_val[0]
            st.session_state[f'fa_dyn_{i}_max'] = slider_val[1]

# Side B
st.subheader("Side B Criteria")
with st.container():
    cols = st.columns(6)
    fb_country = cols[0].multiselect("Country B", options=sorted(df_all['Country'].dropna().unique()), key='fb_country')
    fb_stance = cols[1].multiselect("Stance B", options=sorted(df_all['Stance'].dropna().unique()), key='fb_stance')
    fb_hometown = cols[2].multiselect("Hometown B", options=sorted(df_all['HometownFighter'].dropna().unique()), key='fb_hometown')
    fb_fn_min = cols[3].number_input("Min Fight # B", value=1, key='fb_fn_min')
    fb_fn_max = cols[4].number_input("Max Fight # B", value=int(df_all['FightNumber'].max()), key='fb_fn_max')
    fb_prev_title = cols[5].selectbox("Prev Title B", options=["All","Yes","No"], key='fb_prev_title')

    st.markdown("**Previous Outcomes**")
    cols = st.columns(6)
    fb_prev1 = cols[0].multiselect("Prev 1 B", options=sorted(df_all['Prev1_Outcome_raw'].dropna().unique()), key='fb_prev1')
    fb_prev2 = cols[1].multiselect("Prev 2 B", options=sorted(df_all['Prev2_Outcome_raw'].dropna().unique()), key='fb_prev2')
    fb_prev3 = cols[2].multiselect("Prev 3 B", options=sorted(df_all['Prev3_Outcome_raw'].dropna().unique()), key='fb_prev3')
    fb_career1 = cols[3].multiselect("Career F1 B", options=sorted(df_all['Career1_Outcome_raw'].dropna().unique()), key='fb_career1')
    fb_career2 = cols[4].multiselect("Career F2 B", options=sorted(df_all['Career2_Outcome_raw'].dropna().unique()), key='fb_career2')
    fb_career3 = cols[5].multiselect("Career F3 B", options=sorted(df_all['Career3_Outcome_raw'].dropna().unique()), key='fb_career3')

    st.markdown("**Continuous Filters B**")
    for col, label in zip(SLIDER_COLUMNS, SLIDER_LABELS):
        mn, mx = slider_min_max[col]
        range_key = f'fb_{col}_range'
        if range_key not in st.session_state:
            st.session_state[range_key] = (mn, mx)

        c1, c2 = st.columns(2)
        min_val = c1.number_input(f"{label} B Min", min_value=mn, max_value=mx,
                                  value=st.session_state[range_key][0],
                                  key=f'fb_{col}_min')
        max_val = c2.number_input(f"{label} B Max", min_value=mn, max_value=mx,
                                  value=st.session_state[range_key][1],
                                  key=f'fb_{col}_max')
        if (min_val, max_val) != st.session_state[range_key]:
            st.session_state[range_key] = (min_val, max_val)

        slider_val = st.slider(f"{label} B", min_value=mn, max_value=mx,
                               value=st.session_state[range_key],
                               key=f'fb_{col}_slider')
        if slider_val != st.session_state[range_key]:
            st.session_state[range_key] = slider_val
            st.session_state[f'fb_{col}_min'] = slider_val[0]
            st.session_state[f'fb_{col}_max'] = slider_val[1]

    st.markdown("**Dynamic Sliders B**")
    for i in range(3):
        feat = st.selectbox(f"Dyn Feat {i+1} B", options=["None"]+FEATURES_WINNER, key=f'fb_dyn_{i}_feat')
        if feat == "None":
            mn, mx = 0, 1
        else:
            mn, mx = df_all[feat].min(), df_all[feat].max()

        if st.session_state.get(f'fb_dyn_{i}_last_feat') != feat:
            st.session_state[f'fb_dyn_{i}_range'] = (mn, mx)
            st.session_state[f'fb_dyn_{i}_last_feat'] = feat
        if f'fb_dyn_{i}_range' not in st.session_state:
            st.session_state[f'fb_dyn_{i}_range'] = (mn, mx)

        c1, c2 = st.columns(2)
        min_val = c1.number_input(f"Min {i+1} B", min_value=mn, max_value=mx,
                                  value=st.session_state[f'fb_dyn_{i}_range'][0],
                                  key=f'fb_dyn_{i}_min')
        max_val = c2.number_input(f"Max {i+1} B", min_value=mn, max_value=mx,
                                  value=st.session_state[f'fb_dyn_{i}_range'][1],
                                  key=f'fb_dyn_{i}_max')
        if (min_val, max_val) != st.session_state[f'fb_dyn_{i}_range']:
            st.session_state[f'fb_dyn_{i}_range'] = (min_val, max_val)

        slider_val = st.slider(f"Dyn Range {i+1} B", min_value=mn, max_value=mx,
                               value=st.session_state[f'fb_dyn_{i}_range'],
                               key=f'fb_dyn_{i}_slider')
        if slider_val != st.session_state[f'fb_dyn_{i}_range']:
            st.session_state[f'fb_dyn_{i}_range'] = slider_val
            st.session_state[f'fb_dyn_{i}_min'] = slider_val[0]
            st.session_state[f'fb_dyn_{i}_max'] = slider_val[1]

# Last X Fights
st.markdown("---")
st.subheader("Last X Fights")
last_x = st.number_input("Show last", min_value=1, value=100, key='last_x')
if not df_filtered.empty:
    df_show = df_filtered.drop_duplicates(subset='FightID', keep='first')
    df_show = df_show.sort_values('FightDate', ascending=False).head(last_x)
    st.dataframe(df_show[['FightDate','Fighter','Opponent','Win?','Method','WC','Round']])

# Feature Importance
st.subheader("Feature Importance")
if st.button("Compute Feature Importance"):
    with st.spinner("Training Random Forest..."):
        df = df_all[df_all['Win?'].isin(['Yes','No'])]
        target = st.session_state.get('target', 'win')
        if target == 'win':
            y = (df['Win?']=='Yes').astype(int)
        elif target == 'complete3rds':
            y = df['Completed3Rounds'].fillna(0).astype(int)
        else:
            y = df['Survived15'].fillna(0).astype(int)
        feat_cols = [c for c in FEATURES_WINNER if c in df.columns]
        if target in ('complete3rds','complete1.5rounds'):
            feat_cols = [c for c in feat_cols if (c.startswith('Abs') or c.startswith('Mean'))]
        else:
            feat_cols = [c for c in feat_cols if not (c.startswith('Abs') or c.startswith('Mean'))]
        if feat_cols:
            X = df[feat_cols].fillna(0)
            rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importances = pd.DataFrame({'Feature': feat_cols, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
            st.dataframe(importances.head(200))
        else:
            st.write("No numeric features available.")
