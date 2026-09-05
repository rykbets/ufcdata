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

# ----------------------------- Exclusions -----------------------------
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
def apply_range_filter(df, mask, col, range_vals, target='win'):
    if col not in df.columns or range_vals is None:
        return mask
    if target in ('complete3rds','complete1.5rounds') and col in ABS_MAPPING:
        col = ABS_MAPPING[col]
    min_val, max_val = range_vals
    if min_val is not None:
        mask &= df[col] >= min_val
    if max_val is not None:
        mask &= df[col] <= max_val
    return mask

def apply_dynamic_filters(df, mask, dynamic_sliders, target='win'):
    for slider in dynamic_sliders:
        col = slider.get('col')
        rng = slider.get('range')
        if col and rng:
            mask = apply_range_filter(df, mask, col, rng, target)
    return mask

def build_side_mask(data, side_params, target='win'):
    mask = pd.Series(True, index=data.index)
    if 'FightNumber' in data.columns:
        mask &= (data['FightNumber'] >= side_params.get('fn_min',1)) & (data['FightNumber'] <= side_params.get('fn_max',1e6))
    if side_params.get('stance'):
        mask &= data['Stance'].isin(side_params['stance'])
    skip_nc = side_params.get('skip_nc', False)
    prev_cols = ['Prev1_Outcome_raw','Prev2_Outcome_raw','Prev3_Outcome_raw',
                 'Career1_Outcome_raw','Career2_Outcome_raw','Career3_Outcome_raw']
    if skip_nc:
        prev_cols = [c.replace('raw','skipNC') for c in prev_cols]
    for col, key in zip(prev_cols, ['prev1','prev2','prev3','career1','career2','career3']):
        selected = side_params.get(key, [])
        if selected and col in data.columns:
            cond = pd.Series(False, index=data.index)
            if "Win (any)" in selected: cond |= data[col].str.startswith('Win', na=False)
            if "Loss (any)" in selected: cond |= data[col].str.startswith('Loss', na=False)
            exact = [s for s in selected if s not in ("Win (any)","Loss (any)")]
            if exact: cond |= data[col].isin(exact)
            mask &= cond
    if side_params.get('prev_title','All') != 'All' and 'Prev1_Title' in data.columns:
        mask &= data['Prev1_Title'].str.strip().str.lower() == side_params['prev_title'].lower()
    if side_params.get('hometown') and 'HometownFighter' in data.columns:
        mask &= data['HometownFighter'].isin(side_params['hometown'])
    if side_params.get('country') and 'Country' in data.columns:
        mask &= data['Country'].isin(side_params['country'])
    for col in SLIDER_COLUMNS:
        key = f'{col}_range'
        if key in side_params:
            mask = apply_range_filter(data, mask, col, side_params[key], target)
    dyn = side_params.get('dynamic_sliders', [])
    mask = apply_dynamic_filters(data, mask, dyn, target)
    return mask

def apply_spider_filters(params, target='win'):
    if not params:
        return pd.DataFrame()
    mask_strict = pd.Series(True, index=df_all.index)
    if params.get('wc'): mask_strict &= df_all['WC'].isin(params['wc'])
    if params.get('event_country'): mask_strict &= df_all['EventCountry'].isin(params['event_country'])
    if params.get('sched_rounds'): mask_strict &= df_all['ScheduledRounds'].isin(params['sched_rounds'])
    if params.get('title_fight','All') != 'All': mask_strict &= df_all['Title'] == params['title_fight']
    if params.get('new_wc') and 'IsNewWeightClass' in df_all.columns:
        mask_strict &= df_all['IsNewWeightClass'] == True

    fight_ok = mask_strict.groupby(df_all['FightID']).transform('all')
    df_strict = df_all[fight_ok].copy()

    def extract_side(prefix):
        p = {}
        for k, v in params.items():
            if k.startswith(prefix):
                p[k.replace(prefix, '')] = v
        if 'fn_min' not in p: p['fn_min'] = 1
        if 'fn_max' not in p: p['fn_max'] = 1e6
        return p

    fa = extract_side('sideA_')
    fb = extract_side('sideB_')
    fa['dynamic_sliders'] = params.get('sideA_dynamic_sliders', [])
    fb['dynamic_sliders'] = params.get('sideB_dynamic_sliders', [])

    mask_a = build_side_mask(df_strict, fa, target)
    mask_b = build_side_mask(df_strict, fb, target)

    grouped = df_strict.groupby('FightID').groups
    keep = pd.Series(False, index=df_strict.index)
    for fid, idx_list in grouped.items():
        if len(idx_list) != 2:
            continue
        i0, i1 = idx_list[0], idx_list[1]
        if (mask_a.loc[i0] and mask_b.loc[i1]) or (mask_a.loc[i1] and mask_b.loc[i0]):
            keep[i0] = True
            keep[i1] = True
    side_a = keep & mask_a
    return df_strict[side_a].copy()

def compute_metrics(df, target='win'):
    completed = df[df['Win?'].isin(['Yes','No'])]
    n_fights = completed['FightID'].nunique()
    n_appearances = len(completed)
    if target == 'win':
        wins = (completed['Win?'] == 'Yes').sum()
        wr = wins / n_appearances if n_appearances > 0 else 0.0
        if n_appearances > 0:
            se = np.sqrt(0.5*0.5/n_appearances)
            z = (wr - 0.5)/se if se > 0 else 0.0
            result = binomtest(wins, n_appearances, p=0.5, alternative='two-sided')
            p_val = result.pvalue
            ci_low, ci_high = result.proportion_ci(confidence_level=0.95)
            flag = 'significant' if p_val < 0.05 else 'not significant'
        else:
            z = 0.0; p_val = 1.0; ci_low = np.nan; ci_high = np.nan; flag = 'not significant'
        return n_fights, n_appearances, wr, ci_low, ci_high, p_val, flag, z
    elif target in ['complete3rds','complete1.5rounds']:
        fight_ids = completed['FightID'].unique()
        if len(fight_ids) == 0:
            return 0, 0, 0.0, np.nan, np.nan, 1.0, 'not significant', 0.0
        df_fights = df_all[df_all['FightID'].isin(fight_ids) & df_all['Win?'].isin(['Yes','No'])]
        if target == 'complete3rds':
            comp = df_fights.groupby('FightID')['Survived3R'].min()
        else:
            comp = df_fights.groupby('FightID')['Survived15'].min()
        n_fights = len(comp)
        n_completed = comp.sum()
        rate = n_completed / n_fights if n_fights > 0 else 0.0
        if n_fights > 0:
            z = (rate - 0.5) / np.sqrt(0.5*0.5/n_fights)
            result = binomtest(n_completed, n_fights, p=0.5, alternative='two-sided')
            p_val = result.pvalue
            ci_low, ci_high = result.proportion_ci(confidence_level=0.95)
            flag = 'significant' if p_val < 0.05 else 'not significant'
        else:
            z = 0.0; p_val = 1.0; ci_low = np.nan; ci_high = np.nan; flag = 'not significant'
        return n_fights, n_fights, rate, ci_low, ci_high, p_val, flag, z
    else:
        return compute_metrics(df, 'win')

def count_active_filters(params):
    if not params: return 0
    k = 0
    for key in ['wc','event_country','title_fight','sched_rounds','new_wc']:
        if key in params:
            val = params[key]
            if isinstance(val, list) and len(val)>0: k += 1
            elif val not in ('All','',None,False): k += 1
    for prefix in ['sideA_','sideB_']:
        for key in ['country','stance','hometown','prev_title']:
            full = prefix+key
            if full in params:
                val = params[full]
                if isinstance(val, list) and len(val)>0: k += 1
                elif val not in ('All','',None,False): k += 1
        for col in SLIDER_COLUMNS:
            key = f'{prefix}{col}_range'
            if key in params and params[key] != [slider_min_max[col][0], slider_min_max[col][1]]:
                k += 1
        dyn = params.get(f'{prefix}dynamic_sliders', [])
        for slot in dyn:
            if slot.get('col') and slot.get('range'):
                col = slot['col']
                mn, mx = slider_min_max.get(col, (0,1))
                if slot['range'] != [mn, mx]:
                    k += 1
    return k

def compute_advanced_metrics(df_filtered, df_baseline, target='win', params=None, lambda_penalty=0.1):
    base = df_baseline[df_baseline['Win?'].isin(['Yes','No'])]
    filt = df_filtered[df_filtered['Win?'].isin(['Yes','No'])]
    if base.empty or filt.empty: return None
    n_base = base['FightID'].nunique()
    n_filt = filt['FightID'].nunique()
    if n_filt == 0: return None
    if target == 'win':
        p_base = (base['Win?']=='Yes').mean(); p_filt = (filt['Win?']=='Yes').mean()
        z_base = (p_base-0.5)/np.sqrt(0.5*0.5/n_base) if n_base>0 else 0
        z_filt = (p_filt-0.5)/np.sqrt(0.5*0.5/n_filt) if n_filt>0 else 0
    else:
        col = 'Survived3R' if target=='complete3rds' else 'Survived15'
        base_fights = base.drop_duplicates('FightID'); filt_fights = filt.drop_duplicates('FightID')
        base_comp = df_all[df_all['FightID'].isin(base_fights['FightID'])].groupby('FightID')[col].min()
        filt_comp = df_all[df_all['FightID'].isin(filt_fights['FightID'])].groupby('FightID')[col].min()
        n_base = len(base_comp); n_filt = len(filt_comp)
        if n_filt==0: return None
        p_base = base_comp.mean(); p_filt = filt_comp.mean()
        z_base = (p_base-0.5)/np.sqrt(0.5*0.5/n_base) if n_base>0 else 0
        z_filt = (p_filt-0.5)/np.sqrt(0.5*0.5/n_filt) if n_filt>0 else 0
    if n_base > n_filt:
        efficiency = (z_filt - z_base) / (n_base - n_filt)
    else:
        efficiency = 0.0
    k = count_active_filters(params)
    penalty_score = abs(z_filt) - lambda_penalty * k
    return {'n_base': n_base, 'n_filtered': n_filt, 'z_base': z_base, 'z_filtered': z_filt,
            'efficiency': efficiency, 'penalty_score': penalty_score, 'k': k}

# ----------------------------- Saved Search Management -----------------------------
def load_saved_searches():
    if os.path.exists(SAVED_SEARCHES_FILE):
        with open(SAVED_SEARCHES_FILE, 'r') as f:
            raw = json.load(f)
        normalized = {}
        for name, entry in raw.items():
            if isinstance(entry, dict) and 'params' in entry:
                normalized[name] = entry
            elif isinstance(entry, dict) and 'spider' in entry:
                normalized[name] = {'tab': 'spider', 'params': entry['spider']}
        return normalized
    return {}

def save_saved_searches(searches):
    with open(SAVED_SEARCHES_FILE, 'w') as f:
        json.dump(searches, f)

saved_searches = load_saved_searches()
if 'saved_searches' not in st.session_state:
    st.session_state.saved_searches = saved_searches

def get_params_from_widgets():
    params = {
        'wc': st.session_state.get('wc', []),
        'event_country': st.session_state.get('event_country', []),
        'title_fight': st.session_state.get('title_fight', 'All'),
        'sched_rounds': st.session_state.get('sched_rounds', []),
        'new_wc': st.session_state.get('new_wc', False),
    }
    params['sideA_country'] = st.session_state.get('fa_country', [])
    params['sideA_stance'] = st.session_state.get('fa_stance', [])
    params['sideA_hometown'] = st.session_state.get('fa_hometown', [])
    params['sideA_fn_min'] = st.session_state.get('fa_fn_min', 1)
    params['sideA_fn_max'] = st.session_state.get('fa_fn_max', int(df_all['FightNumber'].max()))
    params['sideA_prev_title'] = st.session_state.get('fa_prev_title', 'All')
    for i in range(1,4):
        params[f'sideA_prev{i}'] = st.session_state.get(f'fa_prev{i}', [])
        params[f'sideA_career{i}'] = st.session_state.get(f'fa_career{i}', [])
    params['sideB_country'] = st.session_state.get('fb_country', [])
    params['sideB_stance'] = st.session_state.get('fb_stance', [])
    params['sideB_hometown'] = st.session_state.get('fb_hometown', [])
    params['sideB_fn_min'] = st.session_state.get('fb_fn_min', 1)
    params['sideB_fn_max'] = st.session_state.get('fb_fn_max', int(df_all['FightNumber'].max()))
    params['sideB_prev_title'] = st.session_state.get('fb_prev_title', 'All')
    for i in range(1,4):
        params[f'sideB_prev{i}'] = st.session_state.get(f'fb_prev{i}', [])
        params[f'sideB_career{i}'] = st.session_state.get(f'fb_career{i}', [])
    for col in SLIDER_COLUMNS:
        params[f'sideA_{col}_range'] = list(st.session_state.get(f'fa_{col}_range', (slider_min_max[col][0], slider_min_max[col][1])))
        params[f'sideB_{col}_range'] = list(st.session_state.get(f'fb_{col}_range', (slider_min_max[col][0], slider_min_max[col][1])))
    dyn_a = []
    for i in range(3):
        feat = st.session_state.get(f'fa_dyn_{i}_feat', 'None')
        rng = st.session_state.get(f'fa_dyn_{i}_range', (0,1))
        if feat == 'None':
            dyn_a.append({})
        else:
            dyn_a.append({'col': feat, 'range': list(rng)})
    dyn_b = []
    for i in range(3):
        feat = st.session_state.get(f'fb_dyn_{i}_feat', 'None')
        rng = st.session_state.get(f'fb_dyn_{i}_range', (0,1))
        if feat == 'None':
            dyn_b.append({})
        else:
            dyn_b.append({'col': feat, 'range': list(rng)})
    params['sideA_dynamic_sliders'] = dyn_a
    params['sideB_dynamic_sliders'] = dyn_b
    return params

def apply_params_to_widgets(params):
    st.session_state['wc'] = params.get('wc', [])
    st.session_state['event_country'] = params.get('event_country', [])
    st.session_state['title_fight'] = params.get('title_fight', 'All')
    st.session_state['sched_rounds'] = params.get('sched_rounds', [])
    st.session_state['new_wc'] = params.get('new_wc', False)

    st.session_state['fa_country'] = params.get('sideA_country', [])
    st.session_state['fa_stance'] = params.get('sideA_stance', [])
    st.session_state['fa_hometown'] = params.get('sideA_hometown', [])
    st.session_state['fa_fn_min'] = params.get('sideA_fn_min', 1)
    st.session_state['fa_fn_max'] = params.get('sideA_fn_max', int(df_all['FightNumber'].max()))
    st.session_state['fa_prev_title'] = params.get('sideA_prev_title', 'All')
    for i in range(1,4):
        st.session_state[f'fa_prev{i}'] = params.get(f'sideA_prev{i}', [])
        st.session_state[f'fa_career{i}'] = params.get(f'sideA_career{i}', [])

    st.session_state['fb_country'] = params.get('sideB_country', [])
    st.session_state['fb_stance'] = params.get('sideB_stance', [])
    st.session_state['fb_hometown'] = params.get('sideB_hometown', [])
    st.session_state['fb_fn_min'] = params.get('sideB_fn_min', 1)
    st.session_state['fb_fn_max'] = params.get('sideB_fn_max', int(df_all['FightNumber'].max()))
    st.session_state['fb_prev_title'] = params.get('sideB_prev_title', 'All')
    for i in range(1,4):
        st.session_state[f'fb_prev{i}'] = params.get(f'sideB_prev{i}', [])
        st.session_state[f'fb_career{i}'] = params.get(f'sideB_career{i}', [])

    for col in SLIDER_COLUMNS:
        st.session_state[f'fa_{col}_range'] = tuple(params.get(f'sideA_{col}_range', [slider_min_max[col][0], slider_min_max[col][1]]))
        st.session_state[f'fb_{col}_range'] = tuple(params.get(f'sideB_{col}_range', [slider_min_max[col][0], slider_min_max[col][1]]))
        st.session_state[f'fa_{col}_min'] = st.session_state[f'fa_{col}_range'][0]
        st.session_state[f'fa_{col}_max'] = st.session_state[f'fa_{col}_range'][1]
        st.session_state[f'fb_{col}_min'] = st.session_state[f'fb_{col}_range'][0]
        st.session_state[f'fb_{col}_max'] = st.session_state[f'fb_{col}_range'][1]

    for i in range(3):
        dyn_a = params.get('sideA_dynamic_sliders', [])
        if i < len(dyn_a) and dyn_a[i].get('col'):
            st.session_state[f'fa_dyn_{i}_feat'] = dyn_a[i]['col']
            st.session_state[f'fa_dyn_{i}_range'] = tuple(dyn_a[i].get('range', [0,1]))
        else:
            st.session_state[f'fa_dyn_{i}_feat'] = 'None'
            st.session_state[f'fa_dyn_{i}_range'] = (0,1)

        dyn_b = params.get('sideB_dynamic_sliders', [])
        if i < len(dyn_b) and dyn_b[i].get('col'):
            st.session_state[f'fb_dyn_{i}_feat'] = dyn_b[i]['col']
            st.session_state[f'fb_dyn_{i}_range'] = tuple(dyn_b[i].get('range', [0,1]))
        else:
            st.session_state[f'fb_dyn_{i}_feat'] = 'None'
            st.session_state[f'fb_dyn_{i}_range'] = (0,1)

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(layout="wide")
st.title("UFC Spider Filter Dashboard")

# Sidebar
with st.sidebar:
    st.header("Saved Searches")
    search_name = st.text_input("Search Name", key="search_name")
    col1, col2 = st.columns(2)
    if col1.button("Save"):
        if search_name:
            st.session_state.saved_searches[search_name] = {'tab': 'spider', 'params': get_params_from_widgets()}
            save_saved_searches(st.session_state.saved_searches)
            st.rerun()
    if col2.button("Delete"):
        selected = st.session_state.get('saved_search_select', None)
        if selected and selected in st.session_state.saved_searches:
            del st.session_state.saved_searches[selected]
            save_saved_searches(st.session_state.saved_searches)
            st.rerun()

    def on_saved_select():
        selected = st.session_state.saved_search_select
        if selected in st.session_state.saved_searches:
            apply_params_to_widgets(st.session_state.saved_searches[selected]['params'])

    saved_search_select = st.selectbox(
        "Saved Searches",
        options=sorted(st.session_state.saved_searches.keys()),
        key='saved_search_select',
        on_change=on_saved_select
    )

    st.markdown("---")
    st.header("Global Target")
    target = st.selectbox("Target", options=["win", "complete3rds", "complete1.5rounds"], key='target')
    lambda_penalty = st.number_input("Penalty λ", value=0.1, step=0.05, key='lambda_penalty')

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

# Main area: filters
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

    with st.expander("Previous Outcomes A", expanded=False):
        cols = st.columns(6)
        fa_prev1 = cols[0].multiselect("Prev 1 A", options=sorted(df_all['Prev1_Outcome_raw'].dropna().unique()), key='fa_prev1')
        fa_prev2 = cols[1].multiselect("Prev 2 A", options=sorted(df_all['Prev2_Outcome_raw'].dropna().unique()), key='fa_prev2')
        fa_prev3 = cols[2].multiselect("Prev 3 A", options=sorted(df_all['Prev3_Outcome_raw'].dropna().unique()), key='fa_prev3')
        fa_career1 = cols[3].multiselect("Career F1 A", options=sorted(df_all['Career1_Outcome_raw'].dropna().unique()), key='fa_career1')
        fa_career2 = cols[4].multiselect("Career F2 A", options=sorted(df_all['Career2_Outcome_raw'].dropna().unique()), key='fa_career2')
        fa_career3 = cols[5].multiselect("Career F3 A", options=sorted(df_all['Career3_Outcome_raw'].dropna().unique()), key='fa_career3')

    with st.expander("Continuous Filters A", expanded=False):
        for col, label in zip(SLIDER_COLUMNS, SLIDER_LABELS):
            mn, mx = slider_min_max[col]
            range_key = f'fa_{col}_range'
            slider_key = f'fa_{col}_slider'
            min_key = f'fa_{col}_min'
            max_key = f'fa_{col}_max'

            # Initialise range if missing
            if range_key not in st.session_state:
                st.session_state[range_key] = (mn, mx)
            # Ensure min/max keys exist (fix KeyError)
            if min_key not in st.session_state:
                st.session_state[min_key] = st.session_state[range_key][0]
            if max_key not in st.session_state:
                st.session_state[max_key] = st.session_state[range_key][1]

            def on_slider_change(col=col, slider_key=slider_key, min_key=min_key, max_key=max_key, range_key=range_key):
                val = st.session_state[slider_key]
                st.session_state[range_key] = val
                st.session_state[min_key] = val[0]
                st.session_state[max_key] = val[1]

            def on_min_change(col=col, min_key=min_key, max_key=max_key, range_key=range_key):
                new_min = st.session_state[min_key]
                current_max = st.session_state[max_key]
                if new_min > current_max:
                    new_min = current_max
                    st.session_state[min_key] = new_min
                st.session_state[range_key] = (new_min, current_max)

            def on_max_change(col=col, min_key=min_key, max_key=max_key, range_key=range_key):
                new_max = st.session_state[max_key]
                current_min = st.session_state[min_key]
                if new_max < current_min:
                    new_max = current_min
                    st.session_state[max_key] = new_max
                st.session_state[range_key] = (current_min, new_max)

            slider_val = st.slider(
                f"{label} A",
                min_value=mn,
                max_value=mx,
                value=st.session_state[range_key],
                key=slider_key,
                on_change=on_slider_change
            )
            st.session_state[range_key] = slider_val
            st.session_state[min_key] = slider_val[0]
            st.session_state[max_key] = slider_val[1]

            with st.expander(f"Set exact range for {label} A", expanded=False):
                c1, c2 = st.columns(2)
                c1.number_input(
                    "Min",
                    min_value=mn,
                    max_value=mx,
                    value=st.session_state[min_key],
                    key=min_key,
                    on_change=on_min_change
                )
                c2.number_input(
                    "Max",
                    min_value=mn,
                    max_value=mx,
                    value=st.session_state[max_key],
                    key=max_key,
                    on_change=on_max_change
                )

    with st.expander("Dynamic Sliders A", expanded=False):
        for i in range(3):
            feat = st.selectbox(f"Dyn Feat {i+1} A", options=["None"]+FEATURES_WINNER, key=f'fa_dyn_{i}_feat')
            if feat == "None":
                mn, mx = 0, 1
            else:
                mn, mx = df_all[feat].min(), df_all[feat].max()

            if st.session_state.get(f'fa_dyn_{i}_last_feat') != feat:
                st.session_state[f'fa_dyn_{i}_range'] = (mn, mx)
                st.session_state[f'fa_dyn_{i}_last_feat'] = feat
            if f'fa_dyn_{i}_range' not in st.session_state:
                st.session_state[f'fa_dyn_{i}_range'] = (mn, mx)

            slider_val = st.slider(f"Dyn Range {i+1} A", min_value=mn, max_value=mx,
                                   value=st.session_state[f'fa_dyn_{i}_range'],
                                   key=f'fa_dyn_{i}_slider')
            st.session_state[f'fa_dyn_{i}_range'] = slider_val

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

    with st.expander("Previous Outcomes B", expanded=False):
        cols = st.columns(6)
        fb_prev1 = cols[0].multiselect("Prev 1 B", options=sorted(df_all['Prev1_Outcome_raw'].dropna().unique()), key='fb_prev1')
        fb_prev2 = cols[1].multiselect("Prev 2 B", options=sorted(df_all['Prev2_Outcome_raw'].dropna().unique()), key='fb_prev2')
        fb_prev3 = cols[2].multiselect("Prev 3 B", options=sorted(df_all['Prev3_Outcome_raw'].dropna().unique()), key='fb_prev3')
        fb_career1 = cols[3].multiselect("Career F1 B", options=sorted(df_all['Career1_Outcome_raw'].dropna().unique()), key='fb_career1')
        fb_career2 = cols[4].multiselect("Career F2 B", options=sorted(df_all['Career2_Outcome_raw'].dropna().unique()), key='fb_career2')
        fb_career3 = cols[5].multiselect("Career F3 B", options=sorted(df_all['Career3_Outcome_raw'].dropna().unique()), key='fb_career3')

    with st.expander("Continuous Filters B", expanded=False):
        for col, label in zip(SLIDER_COLUMNS, SLIDER_LABELS):
            mn, mx = slider_min_max[col]
            range_key = f'fb_{col}_range'
            slider_key = f'fb_{col}_slider'
            min_key = f'fb_{col}_min'
            max_key = f'fb_{col}_max'

            if range_key not in st.session_state:
                st.session_state[range_key] = (mn, mx)
            if min_key not in st.session_state:
                st.session_state[min_key] = st.session_state[range_key][0]
            if max_key not in st.session_state:
                st.session_state[max_key] = st.session_state[range_key][1]

            def on_slider_change(col=col, slider_key=slider_key, min_key=min_key, max_key=max_key, range_key=range_key):
                val = st.session_state[slider_key]
                st.session_state[range_key] = val
                st.session_state[min_key] = val[0]
                st.session_state[max_key] = val[1]

            def on_min_change(col=col, min_key=min_key, max_key=max_key, range_key=range_key):
                new_min = st.session_state[min_key]
                current_max = st.session_state[max_key]
                if new_min > current_max:
                    new_min = current_max
                    st.session_state[min_key] = new_min
                st.session_state[range_key] = (new_min, current_max)

            def on_max_change(col=col, min_key=min_key, max_key=max_key, range_key=range_key):
                new_max = st.session_state[max_key]
                current_min = st.session_state[min_key]
                if new_max < current_min:
                    new_max = current_min
                    st.session_state[max_key] = new_max
                st.session_state[range_key] = (current_min, new_max)

            slider_val = st.slider(
                f"{label} B",
                min_value=mn,
                max_value=mx,
                value=st.session_state[range_key],
                key=slider_key,
                on_change=on_slider_change
            )
            st.session_state[range_key] = slider_val
            st.session_state[min_key] = slider_val[0]
            st.session_state[max_key] = slider_val[1]

            with st.expander(f"Set exact range for {label} B", expanded=False):
                c1, c2 = st.columns(2)
                c1.number_input(
                    "Min",
                    min_value=mn,
                    max_value=mx,
                    value=st.session_state[min_key],
                    key=min_key,
                    on_change=on_min_change
                )
                c2.number_input(
                    "Max",
                    min_value=mn,
                    max_value=mx,
                    value=st.session_state[max_key],
                    key=max_key,
                    on_change=on_max_change
                )

    with st.expander("Dynamic Sliders B", expanded=False):
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

            slider_val = st.slider(f"Dyn Range {i+1} B", min_value=mn, max_value=mx,
                                   value=st.session_state[f'fb_dyn_{i}_range'],
                                   key=f'fb_dyn_{i}_slider')
            st.session_state[f'fb_dyn_{i}_range'] = slider_val

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
    with st.spinner("Training Random Forest on filtered data..."):
        if df_filtered.empty:
            st.write("No filtered data to train on.")
        else:
            if target in ('complete3rds','complete1.5rounds'):
                fight_ids = df_filtered['FightID'].unique()
                train_df = df_all[df_all['FightID'].isin(fight_ids)]
            else:
                train_df = df_filtered
            train_df = train_df[train_df['Win?'].isin(['Yes','No'])]
            if train_df.empty:
                st.write("No completed fights in filtered set.")
            else:
                if target == 'win':
                    y = (train_df['Win?']=='Yes').astype(int)
                elif target == 'complete3rds':
                    y = train_df.groupby('FightID')['Survived3R'].transform('min')
                else:
                    y = train_df.groupby('FightID')['Survived15'].transform('min')

                feat_cols = [c for c in FEATURES_WINNER if c in train_df.columns]
                if target in ('complete3rds','complete1.5rounds'):
                    feat_cols = [c for c in feat_cols if (c.startswith('Abs') or c.startswith('Mean'))]
                else:
                    feat_cols = [c for c in feat_cols if not (c.startswith('Abs') or c.startswith('Mean'))]

                if feat_cols:
                    X = train_df[feat_cols].fillna(0)
                    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
                    rf.fit(X, y)
                    importances = pd.DataFrame({'Feature': feat_cols, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)
                    st.dataframe(importances.head(200))
                else:
                    st.write("No suitable numeric features available.")
