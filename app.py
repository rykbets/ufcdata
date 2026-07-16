import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import os
import gdown
import itertools
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mutual_info_score
from sklearn.linear_model import LinearRegression, LogisticRegression
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

# ---------- Win/Loss Prediction (AUC) ----------
st.header("Win/Loss Prediction")
st.markdown("Select two predictor variables. A logistic regression is fitted to separate wins from losses (AUC shown).")

# Use the same numerical_features list
available_pred = [c for c in numerical_features if c in data.columns and data[c].nunique(dropna=True) >= 2]

if len(available_pred) >= 2:
    col1, col2 = st.columns(2)
    with col1:
        pred_x = st.selectbox("Predictor X", available_pred, key="pred_x")
    with col2:
        pred_y = st.selectbox("Predictor Y", available_pred, key="pred_y")

    if pred_x and pred_y:
        # Scatter data (filtered fights + upcoming)
        plot_data = data[[pred_x, pred_y, 'DetailedResult', 'Fight', 'Win?']].copy()

        # Fit logistic regression on historical data only
        hist = data[data['Win?'].isin(['Yes','No'])].copy()
        hist = hist[[pred_x, pred_y, 'Win?']].dropna()
        if len(hist) < 10 or hist['Win?'].nunique() < 2:
            st.warning("Not enough historical data for logistic regression.")
        else:
            hist['target'] = (hist['Win?'] == 'Yes').astype(int)
            X_hist = hist[[pred_x, pred_y]].values
            y_hist = hist['target'].values

            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score

            logreg = LogisticRegression(max_iter=1000)
            logreg.fit(X_hist, y_hist)

            # AUC
            y_prob = logreg.predict_proba(X_hist)[:, 1]
            auc = roc_auc_score(y_hist, y_prob)

            # Create a grid for decision boundary
            x_min, x_max = X_hist[:, 0].min() - 0.5, X_hist[:, 0].max() + 0.5
            y_min, y_max = X_hist[:, 1].min() - 0.5, X_hist[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                 np.linspace(y_min, y_max, 100))
            Z = logreg.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            Z = Z.reshape(xx.shape)

            # Plot scatter with decision contour
            fig = px.scatter(
                plot_data, x=pred_x, y=pred_y,
                color='DetailedResult',
                color_discrete_map=color_map,
                hover_data=['Fight'],
                title=f"Logistic Regression: {pred_x} & {pred_y} (AUC = {auc:.3f})"
            )
            fig.add_trace(go.Contour(
                x=np.linspace(x_min, x_max, 100),
                y=np.linspace(y_min, y_max, 100),
                z=Z,
                contours_coloring='lines',
                line_width=1,
                showscale=False,
                name='Decision boundary'
            ))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Logistic regression AUC: {auc:.3f}  (predicting Win from {pred_x} and {pred_y})")

            # ---------- Win Probability for an upcoming fight ----------
            st.subheader("Win Probability Estimate")
            all_upcoming_reg = all_fights_display[all_fights_display['Win?'].isna() | (all_fights_display['Win?'] == '')]
            if not all_upcoming_reg.empty:
                up_ids = all_upcoming_reg['FightID'].unique()
                chosen_up = st.selectbox("Select upcoming fight to predict", sorted(up_ids), key="prob_up")
                if chosen_up:
                    up_rows = all_upcoming_reg[all_upcoming_reg['FightID'] == chosen_up]
                    if len(up_rows) == 2:
                        fighter_row = up_rows.iloc[0]
                        if all(pd.notna(fighter_row[f]) for f in [pred_x, pred_y]):
                            up_val = np.array([[fighter_row[pred_x], fighter_row[pred_y]]])
                            prob = logreg.predict_proba(up_val)[0, 1]
                            st.metric(
                                label=f"Win probability for {fighter_row['Fighter']}",
                                value=f"{prob:.1%}"
                            )
                        else:
                            st.warning("Selected fighter does not have both predictor values.")
                else:
                    st.info("Choose an upcoming fight to see win probability.")
            else:
                st.write("No upcoming fights available.")
else:
    st.warning("Not enough numerical features for win/loss prediction.")

# ---------- 3D Scatterplot with Regression Plane and Win Probability ----------
st.header("3D Variable Relationships")
st.markdown("Select three numerical variables. A regression plane is fitted to predict Z from X and Y (R² shown). Also estimate win probability using all three variables.")

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
        plot_data = data[[x3d, y3d, z3d, 'DetailedResult', 'Fight']].dropna()
        if len(plot_data) < 10:
            st.warning("Not enough data for 3D plot.")
        else:
            # Fit plane: Z ~ X + Y
            X_plane = plot_data[[x3d, y3d]].values
            Z_plane = plot_data[z3d].values
            plane_model = LinearRegression()
            plane_model.fit(X_plane, Z_plane)
            r2_plane = plane_model.score(X_plane, Z_plane)

            # Create a grid for the plane surface
            x_range = np.linspace(X_plane[:, 0].min(), X_plane[:, 0].max(), 20)
            y_range = np.linspace(X_plane[:, 1].min(), X_plane[:, 1].max(), 20)
            X_grid, Y_grid = np.meshgrid(x_range, y_range)
            Z_grid = plane_model.predict(np.c_[X_grid.ravel(), Y_grid.ravel()]).reshape(X_grid.shape)

            fig3d = px.scatter_3d(
                plot_data,
                x=x3d, y=y3d, z=z3d,
                color='DetailedResult',
                color_discrete_map=color_map,
                hover_data=['Fight'],
                title=f"3D Scatter: {x3d} vs {y3d} vs {z3d} (Plane R² = {r2_plane:.3f})"
            )

            # Add the regression plane as a surface trace
            fig3d.add_trace(go.Surface(
                x=x_range, y=y_range, z=Z_grid,
                opacity=0.5,
                colorscale='Greys',
                showscale=False,
                name='Regression plane'
            ))

            st.plotly_chart(fig3d, use_container_width=True)
            st.caption(f"Multiple linear regression: {z3d} ~ {x3d} + {y3d}   |   R² = {r2_plane:.3f}")

            # Win probability using all three variables (X,Y,Z) via logistic regression
            st.subheader("Win Probability using X, Y, Z")
            all_upcoming_3d = all_fights_display[all_fights_display['Win?'].isna() | (all_fights_display['Win?'] == '')]
            if not all_upcoming_3d.empty:
                up_ids_3d = all_upcoming_3d['FightID'].unique()
                chosen_up_3d = st.selectbox("Select upcoming fight", sorted(up_ids_3d), key="prob_3d_up")
                if chosen_up_3d:
                    up_rows_3d = all_upcoming_3d[all_upcoming_3d['FightID'] == chosen_up_3d]
                    if len(up_rows_3d) == 2:
                        fighter_row = up_rows_3d.iloc[0]
                        feats_3d = [x3d, y3d, z3d]
                        if all(pd.notna(fighter_row[f]) for f in feats_3d):
                            hist3d = data[data['Win?'].isin(['Yes','No'])].copy()
                            hist3d = hist3d[feats_3d + ['Win?']].dropna()
                            if len(hist3d) < 10:
                                st.warning("Not enough historical data for logistic regression.")
                            else:
                                hist3d['target'] = (hist3d['Win?'] == 'Yes').astype(int)
                                X_hist3d = hist3d[feats_3d].values
                                y_hist3d = hist3d['target'].values
                                if len(np.unique(y_hist3d)) >= 2:
                                    logreg3d = LogisticRegression()
                                    logreg3d.fit(X_hist3d, y_hist3d)
                                    up_val3d = np.array([fighter_row[feats_3d].values])
                                    prob3d = logreg3d.predict_proba(up_val3d)[0, 1]
                                    st.metric(
                                        label=f"Win probability for {fighter_row['Fighter']} (using X,Y,Z)",
                                        value=f"{prob3d:.1%}"
                                    )
                                else:
                                    st.warning("Target variable has no variation.")
                        else:
                            st.warning("Selected fighter does not have all three predictor values.")
            else:
                st.write("No upcoming fights available.")
else:
    st.warning("Not enough numerical features for a 3D plot (need at least 3).")

# =========================================================================
# ADVANCED ANALYSIS (filtered data) – cached importance
# =========================================================================
st.header("Advanced Analysis")

# ---------- 1. Numerical Feature Importance (fighter stats only) ----------
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

# ---------- 2. Categorical Feature Importance (Mutual Information) ----------
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
    st.write("No categorical columns with meaningful variation or no historical data to compute MI.")

# ---------- Best Variable Combinations for Win/Loss (AUC) – Speed-Optimised ----------
st.subheader("Best Variable Combinations for Win/Loss")
st.markdown("Limit the search to the top‑N most important numerical features to speed things up.")

# Fingerprint of current filtered data to detect real filter changes
import hashlib
data_fingerprint = hashlib.md5(pd.util.hash_pandas_object(data).values).hexdigest()

if "auc_results" not in st.session_state:
    st.session_state.auc_results = None
if "last_data_hash" not in st.session_state:
    st.session_state.last_data_hash = data_fingerprint

# Clear results only when the filtered data actually changes
if st.session_state.last_data_hash != data_fingerprint:
    st.session_state.auc_results = None
    st.session_state.last_data_hash = data_fingerprint

# Re‑use the feature importance list (or compute it on the fly)
# importance_features is already defined earlier and contains only fighter-side numerical stats.
# We'll take the top N features by mutual information.
if 'importance_features' in dir() and importance_features:
    mi_df_all = numerical_importance(data, importance_features)
    top_features = mi_df_all['Feature'].tolist()
else:
    # Fallback – use all numerical features (already computed earlier)
    top_features = [c for c in numerical_features if c in data.columns and data[c].nunique(dropna=True) >= 2]

# Let the user choose how many top features to test
num_top = st.slider("Number of top features to test", min_value=5, max_value=min(30, len(top_features)), value=10)
candidates = top_features[:num_top]

if len(candidates) >= 2:
    compute_clicked = st.button("Compute best AUC combinations (fast)", key="compute_auc_fast")

    if compute_clicked or st.session_state.auc_results is None:
        with st.spinner(f"Computing AUC combinations among top {num_top} features… (should be fast)"):
            hist = data[data['Win?'].isin(['Yes','No'])].copy()
            hist['WinNum'] = (hist['Win?'] == 'Yes').astype(int)

            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score

            # --- Two variables ---
            auc_two = []
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
                        model = LogisticRegression(max_iter=1000).fit(X, y)
                        y_prob = model.predict_proba(X)[:, 1]
                        auc = roc_auc_score(y, y_prob)
                        auc_two.append({'Variables': f"{x_col}, {y_col}", 'AUC': auc})
                    except:
                        pass
            df_auc2 = pd.DataFrame(auc_two).sort_values('AUC', ascending=False).head(20)

            # --- Three variables ---
            import itertools
            auc_three = []
            for combo in itertools.combinations(candidates, 3):
                sub = hist[list(combo) + ['WinNum']].dropna()
                if len(sub) < 10 or sub['WinNum'].nunique() < 2:
                    continue
                X = sub[list(combo)].values
                y = sub['WinNum'].values
                try:
                    model = LogisticRegression(max_iter=1000).fit(X, y)
                    y_prob = model.predict_proba(X)[:, 1]
                    auc = roc_auc_score(y, y_prob)
                    auc_three.append({'Variables': ', '.join(combo), 'AUC': auc})
                except:
                    pass
            df_auc3 = pd.DataFrame(auc_three).sort_values('AUC', ascending=False).head(20)

            st.session_state.auc_results = {
                'two_vars': df_auc2,
                'three_vars': df_auc3
            }

    if st.session_state.auc_results is not None:
        st.write("**Top 20 Two‑Variable Win/Loss Predictors (AUC)**")
        st.dataframe(st.session_state.auc_results['two_vars'], use_container_width=True)
        st.write("**Top 20 Three‑Variable Win/Loss Predictors (AUC)**")
        st.dataframe(st.session_state.auc_results['three_vars'], use_container_width=True)
    else:
        st.info("Click the button above to compute the best variable combinations.")
else:
    st.warning("Not enough numerical features for AUC analysis.")
# =========================================================================
# SPIDER CHART (DIFFERENTIALS) + SIMILARITY (FILTERED DATA)
# =========================================================================
st.header("Fight Similarity & Comparison (Filtered)")

upcoming_filtered = data[data['Win?'].isna() | (data['Win?'] == '')]

if upcoming_filtered.empty:
    st.write("No upcoming fights in the filtered dataset.")
else:
    numeric_cols = [c for c in upcoming_filtered.columns if pd.api.types.is_numeric_dtype(upcoming_filtered[c])]

    clean_cols = []
    for col in numeric_cols:
        if re.match(r'Prev\d+_', col):
            continue
        if col.startswith('Opponent_Prev'):
            continue
        clean_cols.append(col)

    wanted_keys = [
        'Age', 'Height', 'Reach',
        'DaysSincePrev', 'Avg3DaysGap',
        'FightNumber', 'Opponent_FightNumber',
        'FighterOddsNum', 'PrevFighterOddsNum',
        'CareerWinPct', 'CareerAvg_', 'Opponent_CareerAvg_',
        '_Diff'
    ]
    spider_vars = sorted([
        c for c in clean_cols
        if any(c.startswith(k) or k in c for k in wanted_keys)
    ])

    if not spider_vars:
        st.warning("No numeric variables found in upcoming data after filtering.")
    else:
        selected_vars = st.multiselect(
            "Select up to 8 variables",
            spider_vars,
            default=spider_vars[:5],
            max_selections=8,
            key="spider_select_vars"
        )

    if selected_vars:
        up_ids = upcoming_filtered['FightID'].unique()
        chosen_fight = st.selectbox(
            "Choose an upcoming fight",
            sorted(up_ids),
            key="spider_fight"
        )

        if chosen_fight:
            fight_rows = upcoming_filtered[upcoming_filtered['FightID'] == chosen_fight]
            if len(fight_rows) != 2:
                st.error("Could not load both fighters.")
            else:
                f1 = fight_rows.iloc[0]
                f2 = fight_rows.iloc[1]

                radar_values = []
                for var in selected_vars:
                    if var.endswith('_Diff') or var in {'AgeDiff', 'HeightDiff', 'ReachDiff'}:
                        val = f1[var] if pd.notna(f1[var]) else 0
                    else:
                        v1 = f1[var] if pd.notna(f1[var]) else 0
                        v2 = f2[var] if pd.notna(f2[var]) else 0
                        val = v1 - v2
                    radar_values.append(val)

                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=radar_values,
                    theta=selected_vars,
                    fill='toself',
                    name=f"{f1['Fighter']} advantage"
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title=f"Advantage: {f1['Fighter']} vs {f2['Fighter']}"
                )
                st.plotly_chart(fig_radar, use_container_width=True)

                # Similarity scorer (uses filtered historical data)
                hist_filtered = data[data['Win?'].isin(['Yes', 'No'])].dropna(subset=selected_vars)
                if not hist_filtered.empty:
                    up_row = f1
                    X_hist = hist_filtered[selected_vars].values
                    scaler = StandardScaler()
                    X_hist_scaled = scaler.fit_transform(X_hist)
                    up_vec = up_row[selected_vars].values.reshape(1, -1)
                    up_scaled = scaler.transform(up_vec)

                    dists = cdist(up_scaled, X_hist_scaled, metric='euclidean').flatten()
                    max_dist = dists.max() if dists.max() > 0 else 1
                    similarity = 100 * (1 - dists / max_dist)

                    res_df = hist_filtered[['FightDate', 'Fighter', 'Opponent', 'WC', 'Win?', 'Method']].copy()
                    res_df['Similarity'] = similarity.round(1)
                    res_df = res_df.sort_values('Similarity', ascending=False).head(20)

                    st.write(f"**Most similar historical fights to {up_row['Fighter']}**")
                    st.dataframe(res_df, use_container_width=True)

                    # Win % among highly similar fights
                    high_sim = res_df[res_df['Similarity'] >= 90]
                    if not high_sim.empty:
                        win_rate_90 = (high_sim['Win?'] == 'Yes').mean() * 100
                        st.metric(
                            label=f"Win rate in ≥90% similar fights ({len(high_sim)} matches)",
                            value=f"{win_rate_90:.1f}%"
                        )
                    else:
                        st.caption("No historical fight with ≥90% similarity.")
                else:
                    st.warning("No complete historical fights for similarity in the filtered data.")
