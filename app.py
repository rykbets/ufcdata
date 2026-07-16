import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import os
import gdown
import itertools
import hashlib
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score, log_loss, brier_score_loss
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist

st.set_page_config(page_title="UFC Pre‑Fight Dashboard", layout="wide")

# ============================================================
# 🔑 YOUR GOOGLE DRIVE FILE IDS
# ============================================================
MAIN_FILE_ID      = "1eWDGGS8qQdLWvS_dgJ-HqObsr4ie9RcD"
UPCOMING_FILE_ID  = "1mUyNR2WLHQjC8IuvG7RA6LoXPjJq3aZ1"

# ---------- Cached data loader ----------
@st.cache_data
def load_full_data():
    gdown.download(f"https://drive.google.com/uc?id={MAIN_FILE_ID}", "ufc_all_data.csv", quiet=True)
    df = pd.read_csv("ufc_all_data.csv", low_memory=False)

    if UPCOMING_FILE_ID.strip():
        gdown.download(f"https://drive.google.com/uc?id={UPCOMING_FILE_ID}", "upcoming.csv", quiet=True)
        if os.path.exists("upcoming.csv"):
            df_up = pd.read_csv("upcoming.csv")
            if 'FightDate' not in df_up.columns:
                df_up['FightDate'] = df_up['FightID'].str[:10]

            def canonical_fightid(fid):
                if not isinstance(fid, str) or ' - ' not in fid:
                    return fid
                parts = fid.split(' - ')
                if len(parts) != 2:
                    return fid
                date_part, names_part = parts
                names = names_part.split(' vs ')
                if len(names) != 2:
                    return fid
                sorted_names = ' vs '.join(sorted([n.strip() for n in names], key=str.lower))
                return f"{date_part} - {sorted_names}"

            main_canonical = set(df['FightID'].apply(canonical_fightid).unique())
            df_up = df_up[~df_up['FightID'].apply(canonical_fightid).isin(main_canonical)]

            if not df_up.empty:
                for col in df.columns:
                    if col not in df_up.columns:
                        df_up[col] = '' if df[col].dtype == object else 0
                df_up = df_up[df.columns]
                df = pd.concat([df, df_up], ignore_index=True)

    # ---------- Preprocessing ----------
    df['FightDate'] = pd.to_datetime(df['FightID'].str[:10])

    for col in ['FighterOddsBFO', 'OpponentOddsBFO']:
        if col not in df.columns:
            df[col] = ''

    if 'Ctrl' in df.columns:
        df['Ctrl'] = pd.to_numeric(df['Ctrl'], errors='coerce').fillna(0)

    agg_cols = ['KD','SS','SSA','TS','TSA','TD','TDA','Subs','Reversals',
                'HSL','HSA','BSL','BSA','LSL','LSA','DSL','DSA','CSL','CSA','GSL','GSA']
    if 'Ctrl' in df.columns:
        agg_cols.append('Ctrl')

    optional_agg = {}
    if 'FighterOddsBFO' in df.columns:
        optional_agg['FighterOddsBFO'] = 'first'
    if 'OpponentOddsBFO' in df.columns:
        optional_agg['OpponentOddsBFO'] = 'first'

    fight_totals = df.groupby(['FightID','Fighter','FightDate'], as_index=False).agg({
        'Opponent':'first', 'Win?':'first', 'Method':'first',
        'WC':'first', 'Stance':'first', 'Country':'first',
        'ScheduledRounds':'first', 'Title':'first', 'Age':'first',
        'Height':'first', 'Reach':'first', 'EventCountry':'first', 'HometownFighter':'first',
        'Round':'max',
        **optional_agg,
        **{col:'sum' for col in agg_cols}
    })
    fight_totals.sort_values(['Fighter','FightDate'], inplace=True)
    fight_totals['FightNumber'] = fight_totals.groupby('Fighter').cumcount() + 1

    # DaysSincePrev, Avg3DaysGap
    fight_totals['DaysSincePrev'] = fight_totals.groupby('Fighter')['FightDate'].diff().dt.days
    def avg_last_3_gaps(group):
        diffs = group['DaysSincePrev']
        return diffs.rolling(3, min_periods=1).mean()
    fight_totals['Avg3DaysGap'] = fight_totals.groupby('Fighter', group_keys=False).apply(
        avg_last_3_gaps, include_groups=False
    ).reset_index(level=0, drop=True)

    # Opponent FightNumber & Hometown
    opp_info = fight_totals[['FightID','Fighter','FightNumber','HometownFighter']].rename(
        columns={'Fighter':'Opponent', 'FightNumber':'Opponent_FightNumber', 'HometownFighter':'Opponent_Hometown'}
    )
    fight_totals = fight_totals.merge(opp_info, on=['FightID','Opponent'], how='left')

    def parse_american_odds(odds_val):
        if pd.isna(odds_val):
            return np.nan
        if isinstance(odds_val, (int, float, np.integer, np.floating)):
            return int(odds_val)
        if isinstance(odds_val, str):
            s = odds_val.strip()
            if s == '':
                return np.nan
            try:
                return int(s.replace('+',''))
            except:
                try:
                    return int(float(s.replace('+','')))
                except:
                    return np.nan
        return np.nan

    if 'FighterOddsBFO' in fight_totals.columns:
        fight_totals['FighterOddsNum'] = fight_totals['FighterOddsBFO'].apply(parse_american_odds)
    else:
        fight_totals['FighterOddsNum'] = np.nan
    if 'OpponentOddsBFO' in fight_totals.columns:
        fight_totals['OpponentOddsNum'] = fight_totals['OpponentOddsBFO'].apply(parse_american_odds)
    else:
        fight_totals['OpponentOddsNum'] = np.nan

    # Differences with opponent (physical)
    pairs = fight_totals.merge(
        fight_totals[['FightID','Fighter','Age','Height','Reach']],
        left_on=['FightID','Opponent'], right_on=['FightID','Fighter'],
        suffixes=('','_opp'), how='left'
    )
    pairs.drop(columns=['Fighter_opp'], inplace=True)
    pairs['AgeDiff'] = pairs['Age'] - pairs['Age_opp']
    pairs['HeightDiff'] = pairs['Height'] - pairs['Height_opp']
    pairs['ReachDiff'] = pairs['Reach'] - pairs['Reach_opp']
    fight_totals = pairs.copy()

    # ---------- Career averages (computed on historical rows only, then forward‑filled) ----------
    career_stat_cols = ['SS','SSA','TS','TSA','TD','TDA','Subs','Reversals','KD','DSL']
    if 'Ctrl' in fight_totals.columns:
        career_stat_cols.append('Ctrl')

    has_stats = fight_totals['SS'].notna()
    stats_df = fight_totals[has_stats].copy()
    stats_df.sort_values(['Fighter','FightDate'], inplace=True)

    for col in career_stat_cols:
        if col in stats_df.columns:
            stats_df[f'cum_{col}'] = stats_df.groupby('Fighter')[col].cumsum()
            stats_df[f'prev_cum_{col}'] = stats_df.groupby('Fighter')[f'cum_{col}'].shift(1).fillna(0)

    stats_df['prev_fights_count'] = stats_df.groupby('Fighter').cumcount()

    for col in career_stat_cols:
        if col in stats_df.columns:
            stats_df[f'CareerAvg_{col}'] = (
                stats_df[f'prev_cum_{col}'] / stats_df['prev_fights_count'].replace(0, np.nan)
            )

    # Career win %
    stats_df['cum_wins'] = stats_df.groupby('Fighter')['Win?'].apply(
        lambda x: (x == 'Yes').cumsum()
    ).reset_index(level=0, drop=True)
    stats_df['prev_wins'] = stats_df.groupby('Fighter')['cum_wins'].shift(1).fillna(0)
    stats_df['CareerWinPct'] = (stats_df['prev_wins'] / stats_df['prev_fights_count'].replace(0, np.nan)) * 100

    merge_cols = ['FightID','Fighter'] + [f'CareerAvg_{c}' for c in career_stat_cols] + ['CareerWinPct']
    for col in merge_cols:
        if col in fight_totals.columns and col not in ['FightID','Fighter']:
            fight_totals.drop(columns=col, inplace=True)
    fight_totals = fight_totals.merge(stats_df[merge_cols], on=['FightID','Fighter'], how='left')

    # Forward‑fill career averages (so upcoming rows have the latest pre‑fight values)
    avg_cols = [f'CareerAvg_{c}' for c in career_stat_cols] + ['CareerWinPct']
    for col in avg_cols:
        if col in fight_totals.columns:
            fight_totals[col] = fight_totals.groupby('Fighter')[col].ffill().bfill()

    # ---------- Opponent career averages ----------
    opp_career = fight_totals[['FightID','Fighter'] + avg_cols].copy()
    opp_career.rename(columns={'Fighter':'Opponent', **{c: f'Opponent_{c}' for c in avg_cols}}, inplace=True)
    fight_totals = fight_totals.merge(opp_career, on=['FightID','Opponent'], how='left')

    # ---------- Opponent days & gaps ----------
    opp_days = fight_totals[['FightID','Fighter','DaysSincePrev','Avg3DaysGap']].copy()
    opp_days.rename(columns={
        'Fighter':'Opponent',
        'DaysSincePrev':'Opponent_DaysSincePrev',
        'Avg3DaysGap':'Opponent_Avg3DaysGap'
    }, inplace=True)
    fight_totals = fight_totals.merge(opp_days, on=['FightID','Opponent'], how='left')

    # ---------- CREATE DIFFERENTIAL COLUMNS (Fighter − Opponent) ----------
    diff_pairs = [
        ('CareerAvg_SS', 'Opponent_CareerAvg_SS'),
        ('CareerAvg_SSA', 'Opponent_CareerAvg_SSA'),
        ('CareerAvg_TS', 'Opponent_CareerAvg_TS'),
        ('CareerAvg_TSA', 'Opponent_CareerAvg_TSA'),
        ('CareerAvg_TD', 'Opponent_CareerAvg_TD'),
        ('CareerAvg_TDA', 'Opponent_CareerAvg_TDA'),
        ('CareerAvg_Subs', 'Opponent_CareerAvg_Subs'),
        ('CareerAvg_Reversals', 'Opponent_CareerAvg_Reversals'),
        ('CareerAvg_KD', 'Opponent_CareerAvg_KD'),
        ('CareerAvg_DSL', 'Opponent_CareerAvg_DSL'),
        ('CareerAvg_Ctrl', 'Opponent_CareerAvg_Ctrl'),
        ('CareerWinPct', 'Opponent_CareerWinPct'),
        ('DaysSincePrev', 'Opponent_DaysSincePrev'),
        ('Avg3DaysGap', 'Opponent_Avg3DaysGap'),
    ]
    for f_col, o_col in diff_pairs:
        if f_col in fight_totals.columns and o_col in fight_totals.columns:
            fight_totals[f'{f_col}_Diff'] = fight_totals[f_col] - fight_totals[o_col]

    # Accuracy
    if 'CareerAvg_SS' in fight_totals.columns and 'CareerAvg_SSA' in fight_totals.columns:
        fight_totals['CareerAvg_SS_Acc'] = (
            (fight_totals['CareerAvg_SS'] / fight_totals['CareerAvg_SSA'].replace(0, np.nan)) * 100
        ).round(1)

    fight_totals['prev_fights_count'] = fight_totals['FightNumber'] - 1

    # Previous fight stats (shifts) – unchanged
    for shift in [1,2,3]:
        for col in ['Win?','Method','Round','WC','Title'] + agg_cols + ['AgeDiff','HeightDiff','ReachDiff']:
            fight_totals[f'Prev{shift}_{col}'] = fight_totals.groupby('Fighter')[col].shift(shift)
        fight_totals.rename(columns={f'Prev{shift}_Win?': f'Prev{shift}_Win'}, inplace=True)

    # Outcome classification helpers (unchanged)
    def extract_round_from_method(method_str):
        if not isinstance(method_str, str): return None
        m = re.search(r'[Rr]ound\s*(\d)', method_str)
        if m: return int(m.group(1))
        numbers = re.findall(r'\d+', method_str)
        if numbers:
            for n in reversed(numbers):
                n_int = int(n)
                if 1 <= n_int <= 5: return n_int
        return None

    def classify_method_detailed(method_str, win, end_round=None):
        if not isinstance(method_str, str): return 'Other'
        method_lower = method_str.lower()
        round_num = extract_round_from_method(method_str)
        if round_num is None: round_num = end_round
        if 'draw' in method_lower: return 'Draw'
        if 'no contest' in method_lower or 'nc' in method_lower: return 'No Contest'
        if win == 'Draw': return 'Draw'
        if win in ('No Contest', 'NC'): return 'No Contest'
        prefix = 'Win' if win == 'Yes' else ('Loss' if win == 'No' else None)
        if 'dq' in method_lower or 'disqualif' in method_lower:
            return f'{prefix} by DQ' if prefix else 'DQ'
        if 'ko' in method_lower or 'tko' in method_lower:
            return f'{prefix} by KO (R{round_num})' if prefix and round_num else f'{prefix} by KO'
        if 'sub' in method_lower:
            return f'{prefix} by Sub (R{round_num})' if prefix and round_num else f'{prefix} by Sub'
        if 'dec' in method_lower: return f'{prefix} by Decision'
        return f'{prefix} by Other' if prefix else 'Other'

    for shift in [1,2,3]:
        fight_totals[f'Prev{shift}_Outcome_raw'] = fight_totals.apply(
            lambda r: classify_method_detailed(r[f'Prev{shift}_Method'], r[f'Prev{shift}_Win'], end_round=r[f'Prev{shift}_Round'])
            if pd.notna(r[f'Prev{shift}_Method']) else None, axis=1
        )

    def get_skip_nc_outcomes(group):
        results = {1: [], 2: [], 3: []}
        methods = group['Method'].tolist()
        wins = group['Win?'].tolist()
        rounds = group['Round'].tolist()
        for i in range(len(group)):
            for shift in [1,2,3]:
                outcome = None
                target_idx = i - shift
                while target_idx >= 0:
                    method = methods[target_idx]
                    if not isinstance(method, str): break
                    win = wins[target_idx]
                    rnd = rounds[target_idx]
                    outcome = classify_method_detailed(method, win, end_round=rnd)
                    if outcome != 'No Contest': break
                    target_idx -= 1
                results[shift].append(outcome if target_idx >= 0 else None)
        return pd.DataFrame({
            'Prev1_Outcome_skipNC': results[1],
            'Prev2_Outcome_skipNC': results[2],
            'Prev3_Outcome_skipNC': results[3]
        }, index=group.index)

    skip_nc_dfs = fight_totals.groupby('Fighter').apply(get_skip_nc_outcomes).reset_index(level=1, drop=True)
    fight_totals = fight_totals.join(skip_nc_dfs)

    def get_career_outcome(group, k, skip_nc=False):
        if skip_nc:
            non_nc_count = 0
            for _, row in group.iterrows():
                if row['Method'] is None or not isinstance(row['Method'], str): continue
                outcome = classify_method_detailed(row['Method'], row['Win?'], end_round=row['Round'])
                if outcome != 'No Contest':
                    non_nc_count += 1
                    if non_nc_count == k: return outcome
            return None
        else:
            row = group[group['FightNumber'] == k]
            if not row.empty:
                r = row.iloc[0]
                return classify_method_detailed(r['Method'], r['Win?'], end_round=r['Round'])
            return None

    career_raw = {}; career_skip = {}
    for fighter, group in fight_totals.groupby('Fighter'):
        career_raw[fighter] = {
            'Career1_Outcome_raw': get_career_outcome(group, 1, False),
            'Career2_Outcome_raw': get_career_outcome(group, 2, False),
            'Career3_Outcome_raw': get_career_outcome(group, 3, False)
        }
        career_skip[fighter] = {
            'Career1_Outcome_skipNC': get_career_outcome(group, 1, True),
            'Career2_Outcome_skipNC': get_career_outcome(group, 2, True),
            'Career3_Outcome_skipNC': get_career_outcome(group, 3, True)
        }

    career_raw_df = pd.DataFrame.from_dict(career_raw, orient='index')
    career_skip_df = pd.DataFrame.from_dict(career_skip, orient='index')
    fight_totals = fight_totals.join(career_raw_df, on='Fighter')
    fight_totals = fight_totals.join(career_skip_df, on='Fighter')

    for shift in [1,2,3]:
        col = f'Prev{shift}_Outcome_raw'
        title_col = f'Prev{shift}_Title'
        opp_df = fight_totals[['FightID','Fighter',col]].dropna(subset=[col])
        opp_df = opp_df.rename(columns={'Fighter':'Opponent', col:f'Opponent_Prev{shift}_Outcome_raw'})
        fight_totals = fight_totals.merge(opp_df, on=['FightID','Opponent'], how='left')
        opp_title_df = fight_totals[['FightID','Fighter',title_col]].dropna(subset=[title_col])
        opp_title_df = opp_title_df.rename(columns={'Fighter':'Opponent', title_col:f'Opponent_Prev{shift}_Title'})
        fight_totals = fight_totals.merge(opp_title_df, on=['FightID','Opponent'], how='left')

    opp_career_raw = pd.DataFrame.from_dict({fighter: {'Opponent_Career1_Outcome_raw': career_raw[fighter]['Career1_Outcome_raw'],
                                                       'Opponent_Career2_Outcome_raw': career_raw[fighter]['Career2_Outcome_raw'],
                                                       'Opponent_Career3_Outcome_raw': career_raw[fighter]['Career3_Outcome_raw']}
                                            for fighter in career_raw}, orient='index')
    opp_career_skip = pd.DataFrame.from_dict({fighter: {'Opponent_Career1_Outcome_skipNC': career_skip[fighter]['Career1_Outcome_skipNC'],
                                                        'Opponent_Career2_Outcome_skipNC': career_skip[fighter]['Career2_Outcome_skipNC'],
                                                        'Opponent_Career3_Outcome_skipNC': career_skip[fighter]['Career3_Outcome_skipNC']}
                                            for fighter in career_skip}, orient='index')
    fight_totals = fight_totals.join(opp_career_raw, on='Opponent')
    fight_totals = fight_totals.join(opp_career_skip, on='Opponent')

    if 'FighterOddsNum' in fight_totals.columns:
        fight_totals['PrevFighterOddsNum'] = fight_totals.groupby('Fighter')['FighterOddsNum'].shift(1)
    else:
        fight_totals['PrevFighterOddsNum'] = np.nan

    for i in range(2,4):
        fight_totals[f'Prev{i}_WC'] = fight_totals.groupby('Fighter')['WC'].shift(i)
    def is_new_weight_class(row):
        if pd.isna(row['Prev1_WC']) or pd.isna(row['Prev2_WC']) or pd.isna(row['Prev3_WC']): return False
        return row['WC'] != row['Prev1_WC'] and row['WC'] != row['Prev2_WC'] and row['WC'] != row['Prev3_WC']
    fight_totals['IsNewWeightClass'] = fight_totals.apply(is_new_weight_class, axis=1)

    return fight_totals

all_fights = load_full_data()
all_fights_display = all_fights[all_fights['FightDate'] >= '2015-01-01'].copy()

# ---------- Clean categorical columns ----------
for col in ['EventCountry', 'Country', 'Stance', 'WC', 'Title', 'ScheduledRounds']:
    if col in all_fights_display.columns:
        all_fights_display[col] = all_fights_display[col].fillna('').astype(str)

# ---------- Sidebar Filters ----------
st.sidebar.title("Filters")

with st.sidebar.expander("General", expanded=True):
    wc = st.multiselect("Weight Class", sorted(all_fights_display['WC'].dropna().unique()))
    stance = st.multiselect("Stance", sorted(all_fights_display['Stance'].dropna().unique()))
    country = st.multiselect("Country", sorted(all_fights_display['Country'].dropna().unique()))
    sched_rounds = st.multiselect("Scheduled Rounds", sorted(all_fights_display['ScheduledRounds'].dropna().unique()))
    title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"])
    hometown = st.selectbox("Hometown", ["All", "Yes", "No"])
    opp_hometown = st.selectbox("Opp Hometown", ["All", "Yes", "No"])
    event_country = st.multiselect("Event Country", sorted(all_fights_display['EventCountry'].dropna().unique()))

with st.sidebar.expander("Fight Numbers", expanded=False):
    fn_min = st.number_input("Min Fight #", value=1, min_value=1, max_value=int(all_fights_display['FightNumber'].max()))
    fn_max = st.number_input("Max Fight #", value=int(all_fights_display['FightNumber'].max()))
    ofn_min = st.number_input("Opp Min Fight #", value=1)
    ofn_max = st.number_input("Opp Max Fight #", value=int(all_fights_display['Opponent_FightNumber'].max()))

with st.sidebar.expander("Career Win %", expanded=False):
    career_win_pct = st.slider("Career Win %", 0, 100, (0, 100))

with st.sidebar.expander("Physical Attributes", expanded=False):
    age = st.slider("Age", int(all_fights_display['Age'].min()), int(all_fights_display['Age'].max()), (int(all_fights_display['Age'].min()), int(all_fights_display['Age'].max())))
    height = st.slider("Height (in)", int(all_fights_display['Height'].min()), int(all_fights_display['Height'].max()), (int(all_fights_display['Height'].min()), int(all_fights_display['Height'].max())))
    reach = st.slider("Reach (in)", int(all_fights_display['Reach'].min()), int(all_fights_display['Reach'].max()), (int(all_fights_display['Reach'].min()), int(all_fights_display['Reach'].max())))

with st.sidebar.expander("Opponent Physical Attributes", expanded=False):
    if 'Age_opp' in all_fights_display.columns:
        age_opp_min = int(all_fights_display['Age_opp'].min()) if not all_fights_display['Age_opp'].isna().all() else 0
        age_opp_max = int(all_fights_display['Age_opp'].max()) if not all_fights_display['Age_opp'].isna().all() else 0
        age_opp = st.slider("Opponent Age", age_opp_min, age_opp_max, (age_opp_min, age_opp_max))
    else:
        age_opp = (0, 0)
        st.write("Opponent age data unavailable.")
    if 'Height_opp' in all_fights_display.columns:
        h_opp_min = int(all_fights_display['Height_opp'].min()) if not all_fights_display['Height_opp'].isna().all() else 0
        h_opp_max = int(all_fights_display['Height_opp'].max()) if not all_fights_display['Height_opp'].isna().all() else 0
        height_opp = st.slider("Opponent Height (in)", h_opp_min, h_opp_max, (h_opp_min, h_opp_max))
    else:
        height_opp = (0, 0)
    if 'Reach_opp' in all_fights_display.columns:
        r_opp_min = int(all_fights_display['Reach_opp'].min()) if not all_fights_display['Reach_opp'].isna().all() else 0
        r_opp_max = int(all_fights_display['Reach_opp'].max()) if not all_fights_display['Reach_opp'].isna().all() else 0
        reach_opp = st.slider("Opponent Reach (in)", r_opp_min, r_opp_max, (r_opp_min, r_opp_max))
    else:
        reach_opp = (0, 0)

with st.sidebar.expander("Differences", expanded=False):
    age_diff = st.slider("Age Diff", int(all_fights_display['AgeDiff'].min()), int(all_fights_display['AgeDiff'].max()), (int(all_fights_display['AgeDiff'].min()), int(all_fights_display['AgeDiff'].max())))
    height_diff = st.slider("Height Diff (in)", int(all_fights_display['HeightDiff'].min()), int(all_fights_display['HeightDiff'].max()), (int(all_fights_display['HeightDiff'].min()), int(all_fights_display['HeightDiff'].max())))
    reach_diff = st.slider("Reach Diff (in)", int(all_fights_display['ReachDiff'].min()), int(all_fights_display['ReachDiff'].max()), (int(all_fights_display['ReachDiff'].min()), int(all_fights_display['ReachDiff'].max())))

with st.sidebar.expander("Days", expanded=False):
    days = st.slider("Days Since Prev", int(all_fights_display['DaysSincePrev'].min()), int(all_fights_display['DaysSincePrev'].max()), (int(all_fights_display['DaysSincePrev'].min()), int(all_fights_display['DaysSincePrev'].max())))
    avg3 = st.slider("Avg 3‑Fight Gap", int(all_fights_display['Avg3DaysGap'].min()), int(all_fights_display['Avg3DaysGap'].max()), (int(all_fights_display['Avg3DaysGap'].min()), int(all_fights_display['Avg3DaysGap'].max())))

with st.sidebar.expander("Odds", expanded=False):
    cur_min = int(all_fights_display['FighterOddsNum'].min()) if not all_fights_display['FighterOddsNum'].isna().all() else 0
    cur_max = int(all_fights_display['FighterOddsNum'].max()) if not all_fights_display['FighterOddsNum'].isna().all() else 0
    if cur_min != cur_max:
        cur_odds = st.slider("Fighter Odds", cur_min, cur_max, (cur_min, cur_max), step=10)
    else:
        st.write("No odds data")
        cur_odds = (0, 0)
    prev_min = int(all_fights_display['PrevFighterOddsNum'].min()) if not all_fights_display['PrevFighterOddsNum'].isna().all() else 0
    prev_max = int(all_fights_display['PrevFighterOddsNum'].max()) if not all_fights_display['PrevFighterOddsNum'].isna().all() else 0
    if prev_min != prev_max:
        prev_odds = st.slider("Prev Fight Odds", prev_min, prev_max, (prev_min, prev_max), step=10)
    else:
        st.write("No previous odds data")
        prev_odds = (0, 0)

new_wc = st.sidebar.checkbox("New Weight Class")
skip_nc = st.sidebar.checkbox("Skip NC outcomes")

prev_title = st.sidebar.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"])
opp_prev_title = st.sidebar.selectbox("Opp Prev Fight Was Title?", ["All", "Yes", "No"])

# Previous outcome columns
if skip_nc:
    prev1_col = 'Prev1_Outcome_skipNC'; prev2_col = 'Prev2_Outcome_skipNC'; prev3_col = 'Prev3_Outcome_skipNC'
    career1_col = 'Career1_Outcome_skipNC'; career2_col = 'Career2_Outcome_skipNC'; career3_col = 'Career3_Outcome_skipNC'
    opp_career1_col = 'Opponent_Career1_Outcome_skipNC'; opp_career2_col = 'Opponent_Career2_Outcome_skipNC'; opp_career3_col = 'Opponent_Career3_Outcome_skipNC'
else:
    prev1_col = 'Prev1_Outcome_raw'; prev2_col = 'Prev2_Outcome_raw'; prev3_col = 'Prev3_Outcome_raw'
    career1_col = 'Career1_Outcome_raw'; career2_col = 'Career2_Outcome_raw'; career3_col = 'Career3_Outcome_raw'
    opp_career1_col = 'Opponent_Career1_Outcome_raw'; opp_career2_col = 'Opponent_Career2_Outcome_raw'; opp_career3_col = 'Opponent_Career3_Outcome_raw'

all_outcomes_raw = sorted(all_fights[prev1_col].dropna().unique())
all_outcomes_career = sorted(all_fights[career1_col].dropna().unique())

with st.sidebar.expander("Previous Outcomes", expanded=False):
    prev1 = st.multiselect("Prev Fight 1", all_outcomes_raw)
    prev2 = st.multiselect("Prev Fight 2", all_outcomes_raw)
    prev3 = st.multiselect("Prev Fight 3", all_outcomes_raw)
    opp_prev1 = st.multiselect("Opp Prev 1", all_outcomes_raw)
    opp_prev2 = st.multiselect("Opp Prev 2", all_outcomes_raw)
    opp_prev3 = st.multiselect("Opp Prev 3", all_outcomes_raw)
    career1 = st.multiselect("Career F1", all_outcomes_career)
    career2 = st.multiselect("Career F2", all_outcomes_career)
    career3 = st.multiselect("Career F3", all_outcomes_career)
    opp_career1 = st.multiselect("Opp Career F1", all_outcomes_career)
    opp_career2 = st.multiselect("Opp Career F2", all_outcomes_career)
    opp_career3 = st.multiselect("Opp Career F3", all_outcomes_career)

# ---------- Apply filters ----------
data = all_fights_display.copy()

if wc: data = data[data['WC'].isin(wc)]
if stance: data = data[data['Stance'].isin(stance)]
if country: data = data[data['Country'].isin(country)]
if sched_rounds: data = data[data['ScheduledRounds'].isin(sched_rounds)]
if title_fight != "All": data = data[data['Title'] == title_fight]
if hometown != "All": data = data[data['HometownFighter'] == hometown]
if opp_hometown != "All": data = data[data['Opponent_Hometown'] == opp_hometown]
if event_country: data = data[data['EventCountry'].isin(event_country)]
if new_wc: data = data[data['IsNewWeightClass'] == True]
if prev_title != "All":
    data = data[data['Prev1_Title'] == prev_title]
if opp_prev_title != "All":
    data = data[data['Opponent_Prev1_Title'] == opp_prev_title]
if prev1: data = data[data[prev1_col].isin(prev1)]
if prev2: data = data[data[prev2_col].isin(prev2)]
if prev3: data = data[data[prev3_col].isin(prev3)]
if career1: data = data[data[career1_col].isin(career1)]
if career2: data = data[data[career2_col].isin(career2)]
if career3: data = data[data[career3_col].isin(career3)]
if opp_career1: data = data[data[opp_career1_col].isin(opp_career1)]
if opp_career2: data = data[data[opp_career2_col].isin(opp_career2)]
if opp_career3: data = data[data[opp_career3_col].isin(opp_career3)]

for opp_shift, opp_widget in [(1, opp_prev1), (2, opp_prev2), (3, opp_prev3)]:
    raw_col = f'Opponent_Prev{opp_shift}_Outcome_raw'
    if raw_col in data.columns:
        use_col = f'Opponent_Prev{opp_shift}_Outcome_skipNC' if skip_nc else raw_col
        if use_col in data.columns and opp_widget:
            data = data[data[use_col].isin(opp_widget)]

data = data[(data['FightNumber'] >= fn_min) & (data['FightNumber'] <= fn_max)]
data = data[(data['Opponent_FightNumber'] >= ofn_min) & (data['Opponent_FightNumber'] <= ofn_max)]
data = data[(data['Age'] >= age[0]) & (data['Age'] <= age[1])]
data = data[(data['Height'] >= height[0]) & (data['Height'] <= height[1])]
data = data[(data['Reach'] >= reach[0]) & (data['Reach'] <= reach[1])]

if 'Age_opp' in data.columns:
    data = data[(data['Age_opp'] >= age_opp[0]) & (data['Age_opp'] <= age_opp[1])]
if 'Height_opp' in data.columns:
    data = data[(data['Height_opp'] >= height_opp[0]) & (data['Height_opp'] <= height_opp[1])]
if 'Reach_opp' in data.columns:
    data = data[(data['Reach_opp'] >= reach_opp[0]) & (data['Reach_opp'] <= reach_opp[1])]

data = data[(data['AgeDiff'] >= age_diff[0]) & (data['AgeDiff'] <= age_diff[1])]
data = data[(data['HeightDiff'] >= height_diff[0]) & (data['HeightDiff'] <= height_diff[1])]
data = data[(data['ReachDiff'] >= reach_diff[0]) & (data['ReachDiff'] <= reach_diff[1])]
data = data[(data['DaysSincePrev'] >= days[0]) & (data['DaysSincePrev'] <= days[1])]
data = data[(data['Avg3DaysGap'] >= avg3[0]) & (data['Avg3DaysGap'] <= avg3[1])]
data = data[(data['CareerWinPct'] >= career_win_pct[0]) & (data['CareerWinPct'] <= career_win_pct[1])]

if not all_fights_display['FighterOddsNum'].isna().all() and cur_odds != (0,0):
    data = data.dropna(subset=['FighterOddsNum'])
    data = data[(data['FighterOddsNum'] >= cur_odds[0]) & (data['FighterOddsNum'] <= cur_odds[1])]
if not all_fights_display['PrevFighterOddsNum'].isna().all() and prev_odds != (0,0):
    data = data.dropna(subset=['PrevFighterOddsNum'])
    data = data[(data['PrevFighterOddsNum'] >= prev_odds[0]) & (data['PrevFighterOddsNum'] <= prev_odds[1])]

# ---------- Main Dashboard ----------
st.title("UFC Pre‑Fight Performance Dashboard")

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
        avg_ss = subset['CareerAvg_SS'].mean() if 'CareerAvg_SS' in subset else 0
        avg_ssa = subset['CareerAvg_SSA'].mean() if 'CareerAvg_SSA' in subset else 0
        avg_ss_acc = subset['CareerAvg_SS_Acc'].mean() if 'CareerAvg_SS_Acc' in subset else 0
        avg_td = subset['CareerAvg_TD'].mean() if 'CareerAvg_TD' in subset else 0
        avg_tda = subset['CareerAvg_TDA'].mean() if 'CareerAvg_TDA' in subset else 0
        avg_subs = subset['CareerAvg_Subs'].mean() if 'CareerAvg_Subs' in subset else 0
        avg_rev = subset['CareerAvg_Reversals'].mean() if 'CareerAvg_Reversals' in subset else 0
        avg_kd = subset['CareerAvg_KD'].mean() if 'CareerAvg_KD' in subset else 0
        avg_dsl = subset['CareerAvg_DSL'].mean() if 'CareerAvg_DSL' in subset else 0
        kdsl = (avg_kd / avg_dsl) if avg_dsl and avg_dsl > 0 else 0
        avg_ctrl = subset['CareerAvg_Ctrl'].mean() if 'CareerAvg_Ctrl' in subset else 0
        ctrtd = (avg_ctrl / avg_td) if avg_td and avg_td > 0 else 0
        age_diff_mean = subset['AgeDiff'].mean()
        height_diff_mean = subset['HeightDiff'].mean()
        reach_diff_mean = subset['ReachDiff'].mean()
        win_pct = subset['CareerWinPct'].mean()

        st.write(f"**Career Win %:** {win_pct:.1f}%")
        st.write(f"**Career Avg SS:** {avg_ss:.1f} / {avg_ssa:.1f} (Acc: {avg_ss_acc:.1f}%)")
        st.write(f"**Career Avg TD:** {avg_td:.1f} / {avg_tda:.1f}")
        st.write(f"**Career Avg Subs:** {avg_subs:.1f} | Rev: {avg_rev:.1f}")
        st.write(f"**Career Avg KD:** {avg_kd:.1f} | K/DSL: {kdsl:.3f}")
        if 'CareerAvg_Ctrl' in subset.columns:
            st.write(f"**Career Avg Ctrl Time:** {avg_ctrl:.0f}s | CTR/TD: {ctrtd:.1f}s")
        st.write(f"**Avg Age Diff:** {age_diff_mean:.1f} | **Avg Height Diff:** {height_diff_mean:.1f} in | **Avg Reach Diff:** {reach_diff_mean:.1f} in")

# ---------- Matchup area (upcoming fights – unfiltered) ----------
st.header("Upcoming Fight Matchup")
upcoming_data_unfiltered = all_fights_display[all_fights_display['Win?'].isna() | (all_fights_display['Win?'] == '')]
if not upcoming_data_unfiltered.empty:
    upcoming_fight_ids = upcoming_data_unfiltered['FightID'].unique()
    selected_fight = st.selectbox("Choose an upcoming fight", sorted(upcoming_fight_ids))
    if selected_fight:
        fight_rows = upcoming_data_unfiltered[upcoming_data_unfiltered['FightID'] == selected_fight]
        if len(fight_rows) == 2:
            f1_row = fight_rows.iloc[0]
            f2_row = fight_rows.iloc[1]
            st.write(f"### {f1_row['Fighter']} vs {f2_row['Fighter']}")

            def show_fighter_stats(row, label):
                st.subheader(label)
                st.write(f"**Age:** {row['Age']}  | **Height:** {row['Height']} in | **Reach:** {row['Reach']} in")
                st.write(f"**Stance:** {row['Stance']} | **Country:** {row['Country']}")
                st.write(f"**Fight #:** {row['FightNumber']} | **Opp Fight #:** {row['Opponent_FightNumber']}")
                st.write(f"**Days Since Prev:** {row['DaysSincePrev']:.0f} days  | **Avg 3‑Fight Gap:** {row['Avg3DaysGap']:.0f} days")
                st.write(f"**Career Win %:** {row['CareerWinPct']:.1f}%")
                st.write(f"**Odds (Fighter/Opp):** {row['FighterOddsBFO']} / {row['OpponentOddsBFO']}")

                st.write("**Career Averages (before this fight):**")
                avg_items = []
                for col_name in ['CareerAvg_SS','CareerAvg_SSA','CareerAvg_KD','CareerAvg_TD','CareerAvg_TDA',
                                 'CareerAvg_Subs','CareerAvg_Reversals','CareerAvg_Ctrl','CareerAvg_DSL']:
                    if col_name in row:
                        val = row[col_name]
                        avg_items.append(f"{col_name.replace('CareerAvg_','')}: {val:.1f}" if pd.notna(val) else f"{col_name.replace('CareerAvg_','')}: --")
                st.write(" · ".join(avg_items) if avg_items else "No career data")

                st.write("**Current Bout Differences:**")
                diff_items = []
                for diff_col, unit in [('AgeDiff','yrs'),('HeightDiff','in'),('ReachDiff','in')]:
                    if diff_col in row:
                        diff_items.append(f"{diff_col}: {row[diff_col]:+.1f} {unit}" if pd.notna(row[diff_col]) else f"{diff_col}: --")
                st.write(" · ".join(diff_items) if diff_items else "N/A")

                st.write("**Previous Outcomes (Fighter):**")
                prev_outs = []
                for shift, col in [(1, prev1_col), (2, prev2_col), (3, prev3_col)]:
                    val = row[col] if pd.notna(row[col]) else '--'
                    prev_outs.append(f"Prev {shift}: {val}")
                st.write(" · ".join(prev_outs))

                st.write("**Career Milestones (Fighter):**")
                career_outs = []
                for shift, col in [(1, career1_col), (2, career2_col), (3, career3_col)]:
                    val = row[col] if pd.notna(row[col]) else '--'
                    career_outs.append(f"F{shift}: {val}")
                st.write(" · ".join(career_outs))

                st.write("**Opponent Previous Outcomes:**")
                opp_prev_outs = []
                for shift in [1,2,3]:
                    raw_col = f'Opponent_Prev{shift}_Outcome_raw'
                    if raw_col in row:
                        use_col = f'Opponent_Prev{shift}_Outcome_skipNC' if skip_nc else raw_col
                        val = row[use_col] if use_col in row and pd.notna(row[use_col]) else '--'
                        opp_prev_outs.append(f"Prev {shift}: {val}")
                st.write(" · ".join(opp_prev_outs) if opp_prev_outs else "N/A")

                st.write("**Opponent Career Milestones:**")
                opp_career_outs = []
                for shift in [1,2,3]:
                    col = f'Opponent_Career{shift}_Outcome_skipNC' if skip_nc else f'Opponent_Career{shift}_Outcome_raw'
                    val = row[col] if col in row and pd.notna(row[col]) else '--'
                    opp_career_outs.append(f"F{shift}: {val}")
                st.write(" · ".join(opp_career_outs) if opp_career_outs else "N/A")

                st.write("**Title History:**")
                title_items = []
                for shift, label in [(1,'Prev Fight'),(2,'Fight‑2'),(3,'Fight‑3')]:
                    col_fighter = f'Prev{shift}_Title'
                    col_opp = f'Opponent_Prev{shift}_Title'
                    f_val = row[col_fighter] if col_fighter in row and pd.notna(row[col_fighter]) else '--'
                    o_val = row[col_opp] if col_opp in row and pd.notna(row[col_opp]) else '--'
                    title_items.append(f"{label}: Fighter={f_val}, Opp={o_val}")
                st.write(" · ".join(title_items))
                st.write("---")

            colA, colB = st.columns(2)
            with colA:
                show_fighter_stats(f1_row, f1_row['Fighter'])
            with colB:
                show_fighter_stats(f2_row, f2_row['Fighter'])
else:
    st.write("No upcoming fights in the dataset.")

# ---------- Last 20 Fights ----------
st.header("Last 20 Fights")
last20 = data.sort_values('FightDate', ascending=False).head(20)
display_cols = ['FightDate','Fighter','Opponent','WC','Win?','Method','Age','Height','Reach',
                'CareerAvg_SS','CareerAvg_KD','DaysSincePrev','Avg3DaysGap','Title',
                'FighterOddsBFO','OpponentOddsBFO']
if 'CareerAvg_Ctrl' in data.columns: display_cols.append('CareerAvg_Ctrl')
display_cols = [c for c in display_cols if c in last20.columns]
st.dataframe(last20[display_cols])

# =========================================================================
# COMMON DEFINITIONS
# =========================================================================
core = ['Age', 'Height', 'Reach', 'Age_opp', 'Height_opp', 'Reach_opp',
        'AgeDiff', 'HeightDiff', 'ReachDiff', 'DaysSincePrev', 'Avg3DaysGap',
        'FightNumber', 'Opponent_FightNumber', 'FighterOddsNum', 'PrevFighterOddsNum',
        'CareerWinPct', 'Opponent_CareerWinPct']
career_avg = [c for c in data.columns if c.startswith('CareerAvg_') and not c.startswith('Opponent_CareerAvg_')]
opp_career_avg = [c for c in data.columns if c.startswith('Opponent_CareerAvg_')]
diff_cols = [c for c in data.columns if c.endswith('_Diff')]
numerical_features = list(dict.fromkeys(
    c for c in core + career_avg + opp_career_avg + diff_cols
    if c in data.columns and not re.match(r'Prev\d+_', c) and not c.startswith('Opponent_Prev')
    and data[c].nunique(dropna=True) >= 2
))

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

# ---------- Bayesian Shrinkage Helper ----------
def compute_win_rates(fighter_name, hist_df, recent_window=5):
    """Return (overall_winrate, recent_winrate, recent_count) for a fighter in hist_df."""
    fighter_fights = hist_df[hist_df['Fighter'] == fighter_name].sort_values('FightDate')
    if len(fighter_fights) == 0:
        return 0, 0, 0
    overall_winrate = (fighter_fights['Win?'] == 'Yes').mean() * 100
    recent_fights = fighter_fights.tail(recent_window)
    recent_count = len(recent_fights)
    recent_winrate = (recent_fights['Win?'] == 'Yes').mean() * 100 if recent_count > 0 else 0
    return overall_winrate, recent_winrate, recent_count

# Shrinkage slider
prior_weight = st.sidebar.slider("Bayesian prior weight", 0.0, 20.0, 5.0, step=0.5, key="prior_weight_global")
recent_window = st.sidebar.slider("Recent fights window", 1, 100, 5, key="recent_win_global")

# ---------- 2D Win/Loss Prediction (Logistic Regression + KNN) ----------
st.header("2D Win/Loss Prediction")
st.markdown("Select two predictor variables. Logistic Regression and KNN models are fitted. Log‑loss, Brier score, win probability, and dataset‑wide win rates are shown.")

available_pred = [c for c in numerical_features if c in data.columns and data[c].nunique(dropna=True) >= 2]

if len(available_pred) >= 2:
    col1, col2 = st.columns(2)
    with col1:
        pred_x = st.selectbox("Predictor X", available_pred, key="pred_x")
    with col2:
        pred_y = st.selectbox("Predictor Y", available_pred, key="pred_y")

    if pred_x and pred_y:
        if pred_x == pred_y:
            st.warning("Please select two different variables for the plot.")
        else:
            # Historical data for model training
            hist = data[data['Win?'].isin(['Yes','No'])].copy()
            hist = hist[[pred_x, pred_y, 'Win?']].dropna()
            if len(hist) < 10 or hist['Win?'].nunique() < 2:
                st.warning("Not enough historical data.")
            else:
                hist['target'] = (hist['Win?'] == 'Yes').astype(int)
                X_hist = hist[[pred_x, pred_y]].values
                y_hist = hist['target'].values

                # Logistic Regression
                logreg = LogisticRegression(max_iter=1000)
                logreg.fit(X_hist, y_hist)
                y_prob_lr = logreg.predict_proba(X_hist)[:, 1]
                ll_lr = log_loss(y_hist, y_prob_lr)
                bs_lr = brier_score_loss(y_hist, y_prob_lr)

                # KNN
                k = st.slider("KNN neighbors", min_value=1, max_value=20, value=5, key="knn_2d")
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_hist, y_hist)
                y_prob_knn = knn.predict_proba(X_hist)[:, 1]
                ll_knn = log_loss(y_hist, y_prob_knn)
                bs_knn = brier_score_loss(y_hist, y_prob_knn)

                # Scatter plot
                plot_data = data[[pred_x, pred_y, 'DetailedResult', 'Fight', 'Win?']].copy()
                fig = px.scatter(
                    plot_data, x=pred_x, y=pred_y,
                    color='DetailedResult',
                    color_discrete_map=color_map,
                    hover_data=['Fight'],
                    title=f"Logistic Regression (LR) & KNN (k={k})"
                )

                # Decision boundary (robust)
                try:
                    x_min, x_max = X_hist[:, 0].min() - 0.5, X_hist[:, 0].max() + 0.5
                    y_min, y_max = X_hist[:, 1].min() - 0.5, X_hist[:, 1].max() + 0.5
                    if x_min < x_max and y_min < y_max:
                        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                             np.linspace(y_min, y_max, 100))
                        grid_points = np.c_[xx.ravel(), yy.ravel()]
                        # Ensure grid has two columns
                        if grid_points.shape[1] == 2:
                            Z = logreg.predict_proba(grid_points)[:, 1].reshape(xx.shape)
                            fig.add_trace(go.Contour(
                                x=np.linspace(x_min, x_max, 100),
                                y=np.linspace(y_min, y_max, 100),
                                z=Z,
                                contours_coloring='lines',
                                line_width=1,
                                showscale=False,
                                name='LR decision boundary'
                            ))
                except Exception as e:
                    st.caption("Could not draw decision boundary (non‑critical).")

                st.plotly_chart(fig, use_container_width=True)

                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("LR Log‑loss", f"{ll_lr:.3f}")
                    st.metric("LR Brier", f"{bs_lr:.3f}")
                with col_m2:
                    st.metric("KNN Log‑loss", f"{ll_knn:.3f}")
                    st.metric("KNN Brier", f"{bs_knn:.3f}")

                st.subheader("Win Probability Estimate")
                all_upcoming_reg = all_fights_display[all_fights_display['Win?'].isna() | (all_fights_display['Win?'] == '')]
                if not all_upcoming_reg.empty:
                    up_ids = all_upcoming_reg['FightID'].unique()
                    chosen_up = st.selectbox("Select upcoming fight", sorted(up_ids), key="prob_2d_up")
                    if chosen_up:
                        up_rows = all_upcoming_reg[all_upcoming_reg['FightID'] == chosen_up]
                        if len(up_rows) == 2:
                            fighter_row = up_rows.iloc[0]
                            fighter_name = fighter_row['Fighter']
                            if all(pd.notna(fighter_row[f]) for f in [pred_x, pred_y]):
                                up_val = np.array([[fighter_row[pred_x], fighter_row[pred_y]]])
                                prob_lr = logreg.predict_proba(up_val)[0, 1]
                                prob_knn = knn.predict_proba(up_val)[0, 1]

                                # Dataset‑wide win rates (based on ALL filtered historical fights)
                                full_hist = data[data['Win?'].isin(['Yes','No'])].sort_values('FightDate')
                                if len(full_hist) > 0:
                                    overall_wr = (full_hist['Win?'] == 'Yes').mean() * 100
                                    recent = full_hist.tail(recent_window)
                                    recent_wr = (recent['Win?'] == 'Yes').mean() * 100 if len(recent) > 0 else 0.0
                                    shrunk_lr = (prior_weight * (overall_wr/100) + prob_lr) / (prior_weight + 1)
                                    shrunk_knn = (prior_weight * (overall_wr/100) + prob_knn) / (prior_weight + 1)
                                else:
                                    overall_wr = recent_wr = 0.0
                                    shrunk_lr = shrunk_knn = None

                                col_p1, col_p2, col_p3 = st.columns(3)
                                with col_p1:
                                    st.metric("LR win prob", f"{prob_lr:.1%}")
                                    st.metric("LR shrunken", f"{shrunk_lr:.1%}" if shrunk_lr is not None else "N/A")
                                with col_p2:
                                    st.metric("KNN win prob", f"{prob_knn:.1%}")
                                    st.metric("KNN shrunken", f"{shrunk_knn:.1%}" if shrunk_knn is not None else "N/A")
                                with col_p3:
                                    st.metric("Overall Win% (dataset)", f"{overall_wr:.1f}%")
                                    st.metric("Recent Win% (last {recent_window})", f"{recent_wr:.1f}%")
                            else:
                                st.warning("Selected fighter does not have both predictor values.")
                else:
                    st.write("No upcoming fights available.")
else:
    st.warning("Not enough numerical features for win/loss prediction.")

# ---------- 3D Win/Loss Prediction ----------
st.header("3D Win/Loss Prediction")
st.markdown("Select three numerical variables. Logistic Regression and KNN models are fitted. Log‑loss, Brier score, win probability, and dataset‑wide win rates are shown.")

three_d_features = [c for c in numerical_features if c in data.columns and data[c].nunique(dropna=True) >= 2]
if len(three_d_features) >= 3:
    col1, col2, col3 = st.columns(3)
    with col1:
        x3d = st.selectbox("X", three_d_features, key="x3d")
    with col2:
        y3d = st.selectbox("Y", three_d_features, key="y3d")
    with col3:
        z3d = st.selectbox("Z", three_d_features, key="z3d")

    if x3d and y3d and z3d:
        hist3d = data[data['Win?'].isin(['Yes','No'])].copy()
        hist3d = hist3d[[x3d, y3d, z3d, 'Win?']].dropna()
        if len(hist3d) < 10 or hist3d['Win?'].nunique() < 2:
            st.warning("Not enough historical data.")
        else:
            hist3d['target'] = (hist3d['Win?'] == 'Yes').astype(int)
            X_hist3d = hist3d[[x3d, y3d, z3d]].values
            y_hist3d = hist3d['target'].values

            logreg3d = LogisticRegression(max_iter=1000)
            logreg3d.fit(X_hist3d, y_hist3d)
            y_prob_lr3d = logreg3d.predict_proba(X_hist3d)[:, 1]
            ll_lr3d = log_loss(y_hist3d, y_prob_lr3d)
            bs_lr3d = brier_score_loss(y_hist3d, y_prob_lr3d)

            k3d = st.slider("KNN neighbors", min_value=1, max_value=20, value=5, key="knn_3d")
            knn3d = KNeighborsClassifier(n_neighbors=k3d)
            knn3d.fit(X_hist3d, y_hist3d)
            y_prob_knn3d = knn3d.predict_proba(X_hist3d)[:, 1]
            ll_knn3d = log_loss(y_hist3d, y_prob_knn3d)
            bs_knn3d = brier_score_loss(y_hist3d, y_prob_knn3d)

            plot_data3d = data[[x3d, y3d, z3d, 'DetailedResult', 'Fight']].dropna()
            fig3d = px.scatter_3d(
                plot_data3d,
                x=x3d, y=y3d, z=z3d,
                color='DetailedResult',
                color_discrete_map=color_map,
                hover_data=['Fight'],
                title="3D Scatter"
            )
            st.plotly_chart(fig3d, use_container_width=True)

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("LR Log‑loss", f"{ll_lr3d:.3f}")
                st.metric("LR Brier", f"{bs_lr3d:.3f}")
            with col_m2:
                st.metric("KNN Log‑loss", f"{ll_knn3d:.3f}")
                st.metric("KNN Brier", f"{bs_knn3d:.3f}")

            st.subheader("Win Probability Estimate")
            all_upcoming_3d = all_fights_display[all_fights_display['Win?'].isna() | (all_fights_display['Win?'] == '')]
            if not all_upcoming_3d.empty:
                up_ids_3d = all_upcoming_3d['FightID'].unique()
                chosen_up_3d = st.selectbox("Select upcoming fight", sorted(up_ids_3d), key="prob_3d_up")
                if chosen_up_3d:
                    up_rows_3d = all_upcoming_3d[all_upcoming_3d['FightID'] == chosen_up_3d]
                    if len(up_rows_3d) == 2:
                        fighter_row = up_rows_3d.iloc[0]
                        fighter_name = fighter_row['Fighter']
                        feats = [x3d, y3d, z3d]
                        if all(pd.notna(fighter_row[f]) for f in feats):
                            up_val3d = np.array([fighter_row[feats].values])
                            prob_lr = logreg3d.predict_proba(up_val3d)[0, 1]
                            prob_knn = knn3d.predict_proba(up_val3d)[0, 1]

                            # Dataset‑wide win rates
                            full_hist = data[data['Win?'].isin(['Yes','No'])].sort_values('FightDate')
                            if len(full_hist) > 0:
                                overall_wr = (full_hist['Win?'] == 'Yes').mean() * 100
                                recent = full_hist.tail(recent_window)
                                recent_wr = (recent['Win?'] == 'Yes').mean() * 100 if len(recent) > 0 else 0.0
                                shrunk_lr = (prior_weight * (overall_wr/100) + prob_lr) / (prior_weight + 1)
                                shrunk_knn = (prior_weight * (overall_wr/100) + prob_knn) / (prior_weight + 1)
                            else:
                                overall_wr = recent_wr = 0.0
                                shrunk_lr = shrunk_knn = None

                            col_p1, col_p2, col_p3 = st.columns(3)
                            with col_p1:
                                st.metric("LR win prob", f"{prob_lr:.1%}")
                                st.metric("LR shrunken", f"{shrunk_lr:.1%}" if shrunk_lr is not None else "N/A")
                            with col_p2:
                                st.metric("KNN win prob", f"{prob_knn:.1%}")
                                st.metric("KNN shrunken", f"{shrunk_knn:.1%}" if shrunk_knn is not None else "N/A")
                            with col_p3:
                                st.metric("Overall Win% (dataset)", f"{overall_wr:.1f}%")
                                st.metric("Recent Win% (last {recent_window})", f"{recent_wr:.1f}%")
                        else:
                            st.warning("Selected fighter does not have all predictor values.")
            else:
                st.write("No upcoming fights available.")
else:
    st.warning("Not enough numerical features for a 3D plot (need at least 3).")

# =========================================================================
# ADVANCED ANALYSIS (filtered data)
# =========================================================================
st.header("Advanced Analysis")

# ---------- 1. Numerical Feature Importance ----------
st.subheader("Feature Importance – Fighter Numerical Stats")

importance_features = [c for c in numerical_features
                       if not c.startswith('Opponent_')
                       and not c.endswith('_Diff')
                       and not re.match(r'Prev\d+_', c)]

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

if not mi_df.empty:
    fig_mi = px.bar(mi_df, x='Mutual Information', y='Feature', orientation='h',
                    title="Top 20 Fighter Stats by Mutual Information with Win/Loss")
    st.plotly_chart(fig_mi, use_container_width=True)
else:
    st.warning("Not enough historical data for feature importance.")

# ---------- Best Variable Combinations (Log‑loss & Brier, excluding FighterOddsNum) ----------
st.subheader("Best Variable Combinations for Win/Loss")
st.markdown("Find the best 2‑ and 3‑variable combinations by log‑loss and Brier score. `FighterOddsNum` is excluded.")

combo_candidates = [c for c in numerical_features if c != 'FighterOddsNum' and c in data.columns and data[c].nunique(dropna=True) >= 2]

data_fingerprint = hash(str(data.shape))

if "combo_results" not in st.session_state:
    st.session_state.combo_results = None
if "last_combo_hash" not in st.session_state:
    st.session_state.last_combo_hash = data_fingerprint

if st.session_state.last_combo_hash != data_fingerprint:
    st.session_state.combo_results = None
    st.session_state.last_combo_hash = data_fingerprint

if not mi_df.empty:
    top_features = mi_df['Feature'].tolist()
else:
    top_features = combo_candidates

num_top = st.slider("Number of top features to test", min_value=5, max_value=min(30, len(top_features)), value=10)
candidates = top_features[:num_top]
candidates = [c for c in candidates if c != 'FighterOddsNum']

if len(candidates) >= 2:
    compute_clicked = st.button("Compute best combinations", key="compute_combo")

    if compute_clicked or st.session_state.combo_results is None:
        with st.spinner(f"Testing combinations among top {len(candidates)} features…"):
            hist = data[data['Win?'].isin(['Yes','No'])].copy()
            hist['WinNum'] = (hist['Win?'] == 'Yes').astype(int)

            # --- Two variables ---
            combo_two = []
            for x_col in candidates:
                for y_col in candidates:
                    if x_col == y_col:
                        continue
                    sub = hist[[x_col, y_col, 'WinNum']].dropna()
                    if len(sub) < 10 or sub['WinNum'].nunique() < 2:
                        continue
                    X = sub[[x_col, y_col]].values
                    y = sub['WinNum'].values
                    try:
                        lr = LogisticRegression(max_iter=1000).fit(X, y)
                        y_prob = lr.predict_proba(X)[:, 1]
                        ll = log_loss(y, y_prob)
                        bs = brier_score_loss(y, y_prob)
                        combo_two.append({'Variables': f"{x_col}, {y_col}", 'LogLoss': ll, 'Brier': bs})
                    except:
                        pass
            df2 = pd.DataFrame(combo_two).sort_values('LogLoss').head(20)

            # --- Three variables ---
            combo_three = []
            for combo in itertools.combinations(candidates, 3):
                sub = hist[list(combo) + ['WinNum']].dropna()
                if len(sub) < 10 or sub['WinNum'].nunique() < 2:
                    continue
                X = sub[list(combo)].values
                y = sub['WinNum'].values
                try:
                    lr = LogisticRegression(max_iter=1000).fit(X, y)
                    y_prob = lr.predict_proba(X)[:, 1]
                    ll = log_loss(y, y_prob)
                    bs = brier_score_loss(y, y_prob)
                    combo_three.append({'Variables': ', '.join(combo), 'LogLoss': ll, 'Brier': bs})
                except:
                    pass
            df3 = pd.DataFrame(combo_three).sort_values('LogLoss').head(20)

            st.session_state.combo_results = {'two': df2, 'three': df3}

    if st.session_state.combo_results is not None:
        st.write("**Top 20 Two‑Variable Combinations (Log‑loss)**")
        st.dataframe(st.session_state.combo_results['two'], use_container_width=True)
        st.write("**Top 20 Three‑Variable Combinations (Log‑loss)**")
        st.dataframe(st.session_state.combo_results['three'], use_container_width=True)
    else:
        st.info("Click the button above to compute the best combinations.")
else:
    st.warning("Not enough features to test (after excluding FighterOddsNum).")

# ---------- Categorical Feature Importance ----------
st.subheader("Categorical Feature Importance with Win/Loss")

potential_cat_cols = ['WC','Stance','Country','EventCountry','Title','ScheduledRounds','HometownFighter','Opponent_Hometown']
categorical_cols = [c for c in potential_cat_cols if c in data.columns and data[c].nunique(dropna=True) > 1]

@st.cache_data
def categorical_importance(_data, cat_cols):
    valid = _data[_data['Win?'].isin(['Yes','No'])].copy()
    valid['Target'] = (valid['Win?'] == 'Yes').astype(int)
    scores = {}
    for col in cat_cols:
        sub = valid[[col, 'Target']].dropna()
        if sub[col].nunique() < 2:
            continue
        codes, _ = pd.factorize(sub[col])
        scores[col] = mutual_info_score(codes, sub['Target'])
    if scores:
        return pd.DataFrame({'Feature': list(scores.keys()), 'Mutual Information': list(scores.values())}).sort_values('Mutual Information', ascending=False).head(20)
    return pd.DataFrame()

cat_mi_df = categorical_importance(data, categorical_cols)

if not cat_mi_df.empty:
    fig_cat = px.bar(cat_mi_df, x='Mutual Information', y='Feature', orientation='h',
                     title="Top Categorical Features by Mutual Information with Win/Loss",
                     color_discrete_sequence=['#636efa'])
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.write("No categorical columns with meaningful variation.")

# =========================================================================
# SPIDER CHART – INDEPENDENT CATEGORICAL FILTERS + LR/KNN + SIMILARITY + BAYESIAN WIN RATES
# =========================================================================
st.header("Fight Similarity & Comparison (Independent Filters)")

# ---- Spider‑specific categorical filters (fighter‑side only) ----
st.subheader("Spider Chart Filters (applied only here)")

col_sp1, col_sp2 = st.columns(2)

with col_sp1:
    spider_wc = st.multiselect("Weight Class", sorted(all_fights_display['WC'].dropna().unique()), key="spider_wc")
    spider_stance = st.multiselect("Stance", sorted(all_fights_display['Stance'].dropna().unique()), key="spider_stance")
    spider_country = st.multiselect("Country", sorted(all_fights_display['Country'].dropna().unique()), key="spider_country")
    spider_sched_rounds = st.multiselect("Scheduled Rounds", sorted(all_fights_display['ScheduledRounds'].dropna().unique()), key="spider_sched")
    spider_event_country = st.multiselect("Event Country", sorted(all_fights_display['EventCountry'].dropna().unique()), key="spider_eventc")

with col_sp2:
    spider_title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"], key="spider_title")
    spider_hometown = st.selectbox("Hometown", ["All", "Yes", "No"], key="spider_home")
    spider_opp_hometown = st.selectbox("Opp Hometown", ["All", "Yes", "No"], key="spider_opphome")
    spider_new_wc = st.checkbox("New Weight Class", key="spider_new_wc")
    spider_skip_nc = st.checkbox("Skip NC outcomes", key="spider_skip_nc")
    spider_prev_title = st.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"], key="spider_prev_title")
    spider_opp_prev_title = st.selectbox("Opp Prev Fight Was Title?", ["All", "Yes", "No"], key="spider_opp_prev_title")

# Previous outcome columns (spider version, depends on spider_skip_nc)
if spider_skip_nc:
    spider_prev1_col = 'Prev1_Outcome_skipNC'; spider_prev2_col = 'Prev2_Outcome_skipNC'; spider_prev3_col = 'Prev3_Outcome_skipNC'
    spider_career1_col = 'Career1_Outcome_skipNC'; spider_career2_col = 'Career2_Outcome_skipNC'; spider_career3_col = 'Career3_Outcome_skipNC'
    spider_opp_career1_col = 'Opponent_Career1_Outcome_skipNC'; spider_opp_career2_col = 'Opponent_Career2_Outcome_skipNC'; spider_opp_career3_col = 'Opponent_Career3_Outcome_skipNC'
else:
    spider_prev1_col = 'Prev1_Outcome_raw'; spider_prev2_col = 'Prev2_Outcome_raw'; spider_prev3_col = 'Prev3_Outcome_raw'
    spider_career1_col = 'Career1_Outcome_raw'; spider_career2_col = 'Career2_Outcome_raw'; spider_career3_col = 'Career3_Outcome_raw'
    spider_opp_career1_col = 'Opponent_Career1_Outcome_raw'; spider_opp_career2_col = 'Opponent_Career2_Outcome_raw'; spider_opp_career3_col = 'Opponent_Career3_Outcome_raw'

all_outcomes_raw_spider = sorted(all_fights[spider_prev1_col].dropna().unique())
all_outcomes_career_spider = sorted(all_fights[spider_career1_col].dropna().unique())

with st.expander("Previous Outcomes (Spider)"):
    spider_prev1 = st.multiselect("Prev Fight 1", all_outcomes_raw_spider, key="spider_prev1")
    spider_prev2 = st.multiselect("Prev Fight 2", all_outcomes_raw_spider, key="spider_prev2")
    spider_prev3 = st.multiselect("Prev Fight 3", all_outcomes_raw_spider, key="spider_prev3")
    spider_opp_prev1 = st.multiselect("Opp Prev 1", all_outcomes_raw_spider, key="spider_opp_prev1")
    spider_opp_prev2 = st.multiselect("Opp Prev 2", all_outcomes_raw_spider, key="spider_opp_prev2")
    spider_opp_prev3 = st.multiselect("Opp Prev 3", all_outcomes_raw_spider, key="spider_opp_prev3")
    spider_career1 = st.multiselect("Career F1", all_outcomes_career_spider, key="spider_career1")
    spider_career2 = st.multiselect("Career F2", all_outcomes_career_spider, key="spider_career2")
    spider_career3 = st.multiselect("Career F3", all_outcomes_career_spider, key="spider_career3")
    spider_opp_career1 = st.multiselect("Opp Career F1", all_outcomes_career_spider, key="spider_opp_career1")
    spider_opp_career2 = st.multiselect("Opp Career F2", all_outcomes_career_spider, key="spider_opp_career2")
    spider_opp_career3 = st.multiselect("Opp Career F3", all_outcomes_career_spider, key="spider_opp_career3")

# ---- Apply spider filters to a copy of all_fights_display ----
spider_data = all_fights_display.copy()

# Categorical filters (all fighter‑side, safe)
if spider_wc: spider_data = spider_data[spider_data['WC'].isin(spider_wc)]
if spider_stance: spider_data = spider_data[spider_data['Stance'].isin(spider_stance)]
if spider_country: spider_data = spider_data[spider_data['Country'].isin(spider_country)]
if spider_sched_rounds: spider_data = spider_data[spider_data['ScheduledRounds'].isin(spider_sched_rounds)]
if spider_title_fight != "All": spider_data = spider_data[spider_data['Title'] == spider_title_fight]
if spider_hometown != "All": spider_data = spider_data[spider_data['HometownFighter'] == spider_hometown]
if spider_opp_hometown != "All": spider_data = spider_data[spider_data['Opponent_Hometown'] == spider_opp_hometown]
if spider_event_country: spider_data = spider_data[spider_data['EventCountry'].isin(spider_event_country)]
if spider_new_wc: spider_data = spider_data[spider_data['IsNewWeightClass'] == True]

# Title/previous outcomes
if spider_prev_title != "All":
    spider_data = spider_data[spider_data['Prev1_Title'] == spider_prev_title]
if spider_opp_prev_title != "All":
    spider_data = spider_data[spider_data['Opponent_Prev1_Title'] == spider_opp_prev_title]
if spider_prev1: spider_data = spider_data[spider_data[spider_prev1_col].isin(spider_prev1)]
if spider_prev2: spider_data = spider_data[spider_data[spider_prev2_col].isin(spider_prev2)]
if spider_prev3: spider_data = spider_data[spider_data[spider_prev3_col].isin(spider_prev3)]
if spider_career1: spider_data = spider_data[spider_data[spider_career1_col].isin(spider_career1)]
if spider_career2: spider_data = spider_data[spider_data[spider_career2_col].isin(spider_career2)]
if spider_career3: spider_data = spider_data[spider_data[spider_career3_col].isin(spider_career3)]
if spider_opp_career1: spider_data = spider_data[spider_data[spider_opp_career1_col].isin(spider_opp_career1)]
if spider_opp_career2: spider_data = spider_data[spider_data[spider_opp_career2_col].isin(spider_opp_career2)]
if spider_opp_career3: spider_data = spider_data[spider_data[spider_opp_career3_col].isin(spider_opp_career3)]

for opp_shift, opp_widget in [(1, spider_opp_prev1), (2, spider_opp_prev2), (3, spider_opp_prev3)]:
    raw_col = f'Opponent_Prev{opp_shift}_Outcome_raw'
    if raw_col in spider_data.columns:
        use_col = f'Opponent_Prev{opp_shift}_Outcome_skipNC' if spider_skip_nc else raw_col
        if use_col in spider_data.columns and opp_widget:
            spider_data = spider_data[spider_data[use_col].isin(opp_widget)]

# ---- Extract upcoming fights from spider_data ----
spider_upcoming = spider_data[spider_data['Win?'].isna() | (spider_data['Win?'] == '')]

if spider_upcoming.empty:
    st.write("No upcoming fights after spider filters.")
else:
    fight_counts = spider_upcoming.groupby('FightID').size()
    complete_ids = fight_counts[fight_counts == 2].index
    spider_upcoming = spider_upcoming[spider_upcoming['FightID'].isin(complete_ids)]

    if spider_upcoming.empty:
        st.warning("No upcoming fight has both fighters after spider filters. Adjust your spider filters to see comparisons.")
    else:
        # Full historical data (all fights passing spider filters)
        spider_hist_full = spider_data[spider_data['Win?'].isin(['Yes','No'])].sort_values('FightDate')

        # Variable selector
        numeric_cols = [c for c in spider_upcoming.columns if pd.api.types.is_numeric_dtype(spider_upcoming[c])]
        clean_cols = [c for c in numeric_cols if not re.match(r'Prev\d+_', c) and not c.startswith('Opponent_Prev')]
        wanted_keys = [
            'Age', 'Height', 'Reach',
            'DaysSincePrev', 'Avg3DaysGap',
            'FightNumber', 'Opponent_FightNumber',
            'FighterOddsNum', 'PrevFighterOddsNum',
            'CareerWinPct', 'CareerAvg_', 'Opponent_CareerAvg_',
            '_Diff'
        ]
        spider_vars = sorted([c for c in clean_cols if any(c.startswith(k) or k in c for k in wanted_keys)])

        if not spider_vars:
            st.warning("No numeric variables found.")
        else:
            selected_vars = st.multiselect("Select variables for models", spider_vars,
                                           default=spider_vars[:5], max_selections=8, key="spider_vars")

        if selected_vars:
            spider_hist_clean = spider_hist_full.dropna(subset=selected_vars)
            if len(spider_hist_clean) < 10 or spider_hist_clean['Win?'].nunique() < 2:
                st.warning("Not enough historical data to train models.")
            else:
                spider_hist_clean['target'] = (spider_hist_clean['Win?'] == 'Yes').astype(int)
                X_spider = spider_hist_clean[selected_vars].values
                y_spider = spider_hist_clean['target'].values

                # Logistic Regression
                lr_spider = LogisticRegression(max_iter=1000)
                lr_spider.fit(X_spider, y_spider)
                y_prob_lr = lr_spider.predict_proba(X_spider)[:, 1]
                ll_lr_spider = log_loss(y_spider, y_prob_lr)
                bs_lr_spider = brier_score_loss(y_spider, y_prob_lr)

                # KNN
                k_spider = st.slider("KNN neighbors", min_value=1, max_value=20, value=5, key="knn_spider")
                knn_spider = KNeighborsClassifier(n_neighbors=k_spider)
                knn_spider.fit(X_spider, y_spider)
                y_prob_knn_spider = knn_spider.predict_proba(X_spider)[:, 1]
                ll_knn_spider = log_loss(y_spider, y_prob_knn_spider)
                bs_knn_spider = brier_score_loss(y_spider, y_prob_knn_spider)

                col_sm1, col_sm2 = st.columns(2)
                with col_sm1:
                    st.metric("LogReg Log‑loss", f"{ll_lr_spider:.3f}")
                    st.metric("LogReg Brier", f"{bs_lr_spider:.3f}")
                with col_sm2:
                    st.metric("KNN Log‑loss", f"{ll_knn_spider:.3f}")
                    st.metric("KNN Brier", f"{bs_knn_spider:.3f}")

                # Pick upcoming fight
                up_ids = sorted(spider_upcoming['FightID'].unique())
                chosen_fight = st.selectbox("Choose an upcoming fight", up_ids, key="spider_fight")

                if chosen_fight:
                    fight_rows = spider_upcoming[spider_upcoming['FightID'] == chosen_fight]
                    f1 = fight_rows.iloc[0]
                    f2 = fight_rows.iloc[1]

                    # Radar of differentials
                    radar_vals = []
                    for var in selected_vars:
                        if var.endswith('_Diff') or var in {'AgeDiff','HeightDiff','ReachDiff'}:
                            val = f1[var] if pd.notna(f1[var]) else 0
                        else:
                            v1 = f1[var] if pd.notna(f1[var]) else 0
                            v2 = f2[var] if pd.notna(f2[var]) else 0
                            val = v1 - v2
                        radar_vals.append(val)

                    fig = go.Figure(go.Scatterpolar(r=radar_vals, theta=selected_vars, fill='toself',
                                                    name=f"{f1['Fighter']} advantage"))
                    fig.update_layout(polar=dict(radialaxis=dict(visible=True)),
                                      title=f"Advantage: {f1['Fighter']} vs {f2['Fighter']}")
                    st.plotly_chart(fig, use_container_width=True)

                    # Win probabilities
                    up_vec = f1[selected_vars].values.reshape(1, -1)
                    prob_lr_f1 = lr_spider.predict_proba(up_vec)[0, 1]
                    prob_knn_f1 = knn_spider.predict_proba(up_vec)[0, 1]

                    # Dataset‑wide win rates (based on ALL spider_hist_full, not just clean)
                    overall_wr_spider = (spider_hist_full['Win?'] == 'Yes').mean() * 100 if len(spider_hist_full) > 0 else 0.0
                    recent_spider = spider_hist_full.tail(recent_window)
                    recent_wr_spider = (recent_spider['Win?'] == 'Yes').mean() * 100 if len(recent_spider) > 0 else 0.0
                    shrunk_lr_spider = (prior_weight * (overall_wr_spider / 100) + prob_lr_f1) / (prior_weight + 1)
                    shrunk_knn_spider = (prior_weight * (overall_wr_spider / 100) + prob_knn_f1) / (prior_weight + 1)

                    col_sp1, col_sp2, col_sp3 = st.columns(3)
                    with col_sp1:
                        st.metric("LogReg win prob", f"{prob_lr_f1:.1%}")
                        st.metric("LogReg shrunken", f"{shrunk_lr_spider:.1%}")
                    with col_sp2:
                        st.metric("KNN win prob", f"{prob_knn_f1:.1%}")
                        st.metric("KNN shrunken", f"{shrunk_knn_spider:.1%}")
                    with col_sp3:
                        st.metric("Overall Win% (spider)", f"{overall_wr_spider:.1f}%")
                        st.metric(f"Recent Win% (last {recent_window})", f"{recent_wr_spider:.1f}%")

                    # ---- Similarity Score (restricted to the most recent N fights) ----
                    st.subheader(f"Most Similar Historical Fights (from last {recent_window} fights)")
                    scaler = StandardScaler()
                    X_spider_scaled = scaler.fit_transform(X_spider)
                    up_scaled = scaler.transform(up_vec)
                    dists = cdist(up_scaled, X_spider_scaled, 'euclidean').flatten()
                    sim_scores = 100 * (1 - dists / (dists.max() or 1))

                    sim_df = spider_hist_clean[['FightDate', 'Fighter', 'Opponent', 'Win?']].copy()
                    sim_df['Similarity'] = sim_scores.round(1)

                    # Keep only the most recent N fights, then top 20 by similarity
                    sim_df = sim_df.sort_values('FightDate', ascending=False).head(recent_window)
                    top_sim = sim_df.sort_values('Similarity', ascending=False).head(20)

                    st.dataframe(top_sim, use_container_width=True)
