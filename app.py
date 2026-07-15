import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import os
import gdown
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import mutual_info_score
import plotly.express as px
import plotly.graph_objects as go   # <-- add this

st.set_page_config(page_title="UFC Pre‑Fight Dashboard", layout="wide")

# ============================================================
# 🔑 YOUR GOOGLE DRIVE FILE IDS (make sure files are publicly shared)
# ============================================================
MAIN_FILE_ID      = "1eWDGGS8qQdLWvS_dgJ-HqObsr4ie9RcD"          # ufc_all_data.csv
UPCOMING_FILE_ID  = "1mUyNR2WLHQjC8IuvG7RA6LoXPjJq3aZ1"      # upcoming_fights.csv (or "" if none)
# ============================================================

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

    # Differences with opponent (also keep opponent's raw attributes)
    pairs = fight_totals.merge(
        fight_totals[['FightID','Fighter','Age','Height','Reach']],
        left_on=['FightID','Opponent'], right_on=['FightID','Fighter'],
        suffixes=('','_opp'), how='left'
    )
    pairs.drop(columns=['Fighter_opp'], inplace=True)
    pairs['AgeDiff'] = pairs['Age'] - pairs['Age_opp']
    pairs['HeightDiff'] = pairs['Height'] - pairs['Height_opp']
    pairs['ReachDiff'] = pairs['Reach'] - pairs['Reach_opp']
    fight_totals = pairs.copy()   # now contains Age_opp, Height_opp, Reach_opp

    # Career pre‑fight averages (striking/grappling)
    career_stat_cols = ['SS','SSA','TS','TSA','TD','TDA','Subs','Reversals','KD','DSL']
    if 'Ctrl' in fight_totals.columns:
        career_stat_cols.append('Ctrl')

    for col in career_stat_cols:
        if col in fight_totals.columns:
            cum_col = fight_totals.groupby('Fighter')[col].cumsum()
            fight_totals[f'cum_{col}'] = cum_col
            fight_totals[f'prev_cum_{col}'] = fight_totals.groupby('Fighter')[f'cum_{col}'].shift(1).fillna(0)

    fight_totals['prev_fights_count'] = fight_totals.groupby('Fighter').cumcount()
    for col in career_stat_cols:
        if col in fight_totals.columns:
            fight_totals[f'CareerAvg_{col}'] = (fight_totals[f'prev_cum_{col}'] / fight_totals['prev_fights_count'].replace(0, np.nan))

    fight_totals['CareerAvg_SS_Acc'] = (
        (fight_totals['CareerAvg_SS'] / fight_totals['CareerAvg_SSA'].replace(0, np.nan)) * 100
    ).round(1)

    # Career win % before this fight
    fight_totals['cum_wins'] = fight_totals.groupby('Fighter')['Win?'].apply(
        lambda x: (x == 'Yes').cumsum()
    ).reset_index(level=0, drop=True)
    fight_totals['prev_wins'] = fight_totals.groupby('Fighter')['cum_wins'].shift(1).fillna(0)
    fight_totals['CareerWinPct'] = (fight_totals['prev_wins'] / fight_totals['prev_fights_count'].replace(0, np.nan)) * 100

    fight_totals = fight_totals.copy()

    # Previous fight stats (shifts) – also shift Title for opponent
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

    # Skip NC outcomes (unchanged)
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

    # Career milestone outcomes (fighter) – unchanged
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

    # Opponent previous outcomes & titles (raw) – unchanged
    for shift in [1,2,3]:
        col = f'Prev{shift}_Outcome_raw'
        title_col = f'Prev{shift}_Title'
        opp_df = fight_totals[['FightID','Fighter',col]].dropna(subset=[col])
        opp_df = opp_df.rename(columns={'Fighter':'Opponent', col:f'Opponent_Prev{shift}_Outcome_raw'})
        fight_totals = fight_totals.merge(opp_df, on=['FightID','Opponent'], how='left')
        opp_title_df = fight_totals[['FightID','Fighter',title_col]].dropna(subset=[title_col])
        opp_title_df = opp_title_df.rename(columns={'Fighter':'Opponent', title_col:f'Opponent_Prev{shift}_Title'})
        fight_totals = fight_totals.merge(opp_title_df, on=['FightID','Opponent'], how='left')

    # Opponent career milestones – unchanged
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

    # New weight class indicator – unchanged
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

# --- NEW: Opponent physical attribute sliders ---
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

# Title fight in previous fight
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

# --- Apply opponent physical filters if columns exist ---
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

# ---------- Scatter Plot ----------
st.header("Scatter Plot")
career_stat_cols_plot = ['SS','SSA','TS','TSA','TD','TDA','Subs','Reversals','KD','DSL']
if 'Ctrl' in data.columns: career_stat_cols_plot.append('Ctrl')
career_avg_columns = [f'CareerAvg_{c}' for c in career_stat_cols_plot] + ['CareerAvg_SS_Acc', 'CareerWinPct']
numeric_cols = ['Age','Height','Reach','AgeDiff','HeightDiff','ReachDiff','DaysSincePrev','Avg3DaysGap',
                'FightNumber','Opponent_FightNumber','FighterOddsNum','PrevFighterOddsNum'] + career_avg_columns
numeric_cols = [c for c in numeric_cols if c in data.columns]

x_col = st.selectbox("X axis", sorted(numeric_cols), index=sorted(numeric_cols).index('CareerAvg_SS') if 'CareerAvg_SS' in numeric_cols else 0)
y_col = st.selectbox("Y axis", sorted(numeric_cols), index=sorted(numeric_cols).index('CareerAvg_KD') if 'CareerAvg_KD' in numeric_cols else 0)

# --- Add a result category column to color upcoming fights differently ---
def result_category(row):
    if pd.isna(row['Win?']) or str(row['Win?']).strip() == '':
        return 'Upcoming'
    if row['Win?'] == 'Yes': return 'Win'
    if row['Win?'] == 'No': return 'Loss'
    if row['Win?'] == 'Draw': return 'Draw'
    if row['Win?'] == 'No Contest': return 'No Contest'
    return 'Other'
data['Result'] = data.apply(result_category, axis=1)

color_discrete_map = {
    'Win': 'green',
    'Loss': 'red',
    'Draw': 'gray',
    'No Contest': 'purple',
    'Upcoming': 'blue'
}

fig = px.scatter(
    data, x=x_col, y=y_col, color='Result',
    color_discrete_map=color_discrete_map,
    hover_data=['Fighter', 'Opponent', 'WC'],
    title=f'{y_col} vs {x_col}'
)
st.plotly_chart(fig, use_container_width=True, key=f"scatter_{x_col}_{y_col}")

# ---------- Decision Tree (max depth 5) ----------
st.header("Customizable Decision Tree")
st.markdown("Use the filtered data (excluding upcoming fights) to find the most informative splits.")

# ---------- Build opponent‑side statistics on the full display set ----------
encoded_data = all_fights_display.copy()

opp_career = encoded_data[['FightID','Fighter','CareerWinPct']].rename(
    columns={'Fighter':'Opponent', 'CareerWinPct':'Opponent_CareerWinPct'})
encoded_data = encoded_data.merge(opp_career, on=['FightID','Opponent'], how='left')

opp_days = encoded_data[['FightID','Fighter','DaysSincePrev','Avg3DaysGap']].rename(
    columns={'Fighter':'Opponent', 'DaysSincePrev':'Opponent_DaysSincePrev', 'Avg3DaysGap':'Opponent_Avg3DaysGap'})
encoded_data = encoded_data.merge(opp_days, on=['FightID','Opponent'], how='left')

# ---------- Core numeric features ----------
core_features = [
    'Age', 'Height', 'Reach',
    'Age_opp', 'Height_opp', 'Reach_opp',
    'AgeDiff', 'HeightDiff', 'ReachDiff',
    'DaysSincePrev', 'Avg3DaysGap',
    'Opponent_DaysSincePrev', 'Opponent_Avg3DaysGap',
    'FightNumber', 'Opponent_FightNumber',
    'FighterOddsNum', 'PrevFighterOddsNum',
    'CareerWinPct', 'Opponent_CareerWinPct'
]

career_avg_cols = [col for col in encoded_data.columns if col.startswith('CareerAvg_')]

feature_cols = [c for c in core_features + career_avg_cols 
                if c in encoded_data.columns and encoded_data[c].nunique(dropna=True) >= 2]

# ---------- Binary outcome features (fighter + opponent, 3 shifts) ----------
outcome_cols = {
    'Prev1': prev1_col,
    'Prev2': prev2_col,
    'Prev3': prev3_col,
    'OppPrev1': 'Opponent_Prev1_Outcome_raw',
    'OppPrev2': 'Opponent_Prev2_Outcome_raw',
    'OppPrev3': 'Opponent_Prev3_Outcome_raw'
}

for prefix, col in outcome_cols.items():
    if col not in encoded_data.columns:
        continue
    encoded_data[f'{prefix}_is_Win']   = encoded_data[col].str.startswith('Win').astype(int)
    encoded_data[f'{prefix}_is_Loss']  = encoded_data[col].str.startswith('Loss').astype(int)
    encoded_data[f'{prefix}_is_Draw']  = encoded_data[col].str.contains('Draw', na=False).astype(int)
    encoded_data[f'{prefix}_is_NC']    = encoded_data[col].str.contains('No Contest', na=False).astype(int)
    encoded_data[f'{prefix}_method_KO']  = encoded_data[col].str.contains('KO', na=False).astype(int)
    encoded_data[f'{prefix}_method_Sub'] = encoded_data[col].str.contains('Sub', na=False).astype(int)
    encoded_data[f'{prefix}_method_Dec'] = encoded_data[col].str.contains('Decision', na=False).astype(int)
    encoded_data[f'{prefix}_method_DQ']  = encoded_data[col].str.contains('DQ', na=False).astype(int)

    for feat in [f'{prefix}_is_Win', f'{prefix}_is_Loss', f'{prefix}_is_Draw', f'{prefix}_is_NC',
                 f'{prefix}_method_KO', f'{prefix}_method_Sub', f'{prefix}_method_Dec', f'{prefix}_method_DQ']:
        if feat in encoded_data.columns and encoded_data[feat].nunique(dropna=True) >= 2:
            feature_cols.append(feat)

# ---------- ALL Title/Hometown binary filters ----------
binary_cols = [
    'Prev1_Title', 'Prev2_Title', 'Prev3_Title',
    'Opponent_Prev1_Title', 'Opponent_Prev2_Title', 'Opponent_Prev3_Title',
    'HometownFighter', 'Opponent_Hometown'
]
for col in binary_cols:
    if col in encoded_data.columns:
        clean_col = col + '_clean'
        encoded_data[clean_col] = encoded_data[col].astype(str).str.strip().str.lower().map({'yes': 1}).fillna(0).astype(int)
        feature_cols.append(clean_col)

feature_cols = sorted(list(set(feature_cols)))

# ---------- Tree data (Win/Loss only) ----------
tree_data = encoded_data[encoded_data['Win?'].isin(['Yes','No'])].copy()
tree_data['Target'] = (tree_data['Win?'] == 'Yes').astype(int)

# ---------- Helper functions ----------
def find_best_split(subset, feature_pool):
    best_feat, best_thresh, best_gain = None, None, -1
    y = subset['Target'].values
    parent_entropy = -(y.mean() * np.log2(y.mean() + 1e-10) + (1 - y.mean()) * np.log2(1 - y.mean() + 1e-10))
    n = len(y)
    for feat in feature_pool:
        X = subset[feat].values
        if np.isnan(X).all():
            continue
        sorted_idx = np.argsort(X)
        X_sorted = X[sorted_idx]
        y_sorted = y[sorted_idx]
        for i in range(1, n - 1):
            if X_sorted[i] == X_sorted[i - 1]:
                continue
            left_weight = i / n
            right_weight = 1 - left_weight
            left_entropy = -(y_sorted[:i].mean() * np.log2(y_sorted[:i].mean() + 1e-10) +
                             (1 - y_sorted[:i].mean()) * np.log2(1 - y_sorted[:i].mean() + 1e-10))
            right_entropy = -(y_sorted[i:].mean() * np.log2(y_sorted[i:].mean() + 1e-10) +
                              (1 - y_sorted[i:].mean()) * np.log2(1 - y_sorted[i:].mean() + 1e-10))
            gain = parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)
            if gain > best_gain:
                best_gain = gain
                best_feat = feat
                best_thresh = (X_sorted[i - 1] + X_sorted[i]) / 2
    if best_feat and set(subset[best_feat].dropna().unique()).issubset({0, 1}):
        best_thresh = 0.5
    return best_feat, best_thresh, best_gain

def suggest_features(subset, feature_pool, top_k=3):
    pool = [f for f in feature_pool if f in subset.columns]
    if not pool:
        return []
    X_sub = subset[pool].dropna()
    y_sub = subset.loc[X_sub.index, 'Target'].values
    if len(X_sub) == 0:
        return []
    mi_scores = mutual_info_classif(X_sub, y_sub, discrete_features=False)
    sorted_idx = np.argsort(mi_scores)[::-1]
    suggested = []
    for idx in sorted_idx:
        if mi_scores[idx] > 0:
            suggested.append(pool[idx])
            if len(suggested) >= top_k:
                break
    return suggested

# ---------- Initialize session state ----------
if 'tree_nodes' not in st.session_state:
    st.session_state.tree_nodes = {}
    st.session_state.next_node_id = 1
    st.session_state.root_built = False

with st.form(key="tree_form"):
    leaf_size = st.number_input("Minimum samples per leaf", min_value=1, value=20)
    build_clicked = st.form_submit_button("Build Tree")

if build_clicked:
    st.session_state.tree_nodes = {}
    st.session_state.next_node_id = 1
    st.session_state.root_built = True
    st.session_state.tree_nodes[0] = {
        'data': tree_data.copy(),
        'feature': None,
        'threshold': None,
        'children': [],
        'depth': 0
    }

# ---------- Feature list expander ----------
with st.expander("🔍 Feature list (click to see all)"):
    binary_clean = [f for f in feature_cols if f.endswith('_clean')]
    other = [f for f in feature_cols if not f.endswith('_clean')]
    st.markdown("**Binary yes/no filters:** " + ", ".join(f"`{f}`" for f in binary_clean) if binary_clean else "None")
    st.markdown("**Other features:** " + ", ".join(other))

# ---------- Simple Black‑&‑White Tree Diagram ----------
def draw_tree():
    if not st.session_state.tree_nodes:
        return

    def count_leaves(node_id):
        node = st.session_state.tree_nodes.get(node_id)
        if not node or not node['children']:
            return 1
        return sum(count_leaves(child) for child in node['children'])

    def layout_tree(node_id, x, y, dx):
        node = st.session_state.tree_nodes.get(node_id)
        if not node:
            return
        win_rate = node['data']['Target'].mean() * 100
        n = len(node['data'])
        text = f"Node {node_id} (n={n}, win={win_rate:.1f}%)"
        if node['feature'] is not None:
            feat = node['feature']
            thresh = node['threshold']
            if set(node['data'][feat].dropna().unique()).issubset({0, 1}):
                text += f"<br>   {feat} ≤ 0.5 → No<br>   {feat} > 0.5 → Yes"
            else:
                text += f"<br>   {feat} ≤ {thresh:.2f}"
        annotations.append(go.layout.Annotation(
            x=x, y=y, text=text, showarrow=False,
            font=dict(color="white", size=11), bgcolor="rgba(0,0,0,0)",
            align="center"
        ))

        if not node['children'] or len(node['children']) < 2:
            return

        left_width = count_leaves(node['children'][0])
        right_width = count_leaves(node['children'][1])
        total_width = left_width + right_width
        if total_width == 0:
            return
        left_dx = dx * (left_width / total_width)
        right_dx = dx * (right_width / total_width)

        left_x = x - (dx / 2) + (left_dx / 2)
        right_x = x + (dx / 2) - (right_dx / 2)
        child_y = y - 1.2

        shapes.append(go.layout.Shape(
            type="line", x0=x, y0=y - 0.3, x1=left_x, y1=child_y + 0.3,
            line=dict(color="gray", width=1)
        ))
        shapes.append(go.layout.Shape(
            type="line", x0=x, y0=y - 0.3, x1=right_x, y1=child_y + 0.3,
            line=dict(color="gray", width=1)
        ))

        layout_tree(node['children'][0], left_x, child_y, left_dx)
        layout_tree(node['children'][1], right_x, child_y, right_dx)

    shapes = []
    annotations = []
    root = st.session_state.tree_nodes.get(0)
    if root:
        total_leaves = count_leaves(0) or 1
        layout_tree(0, 0, 0, total_leaves * 2.5)

    if annotations:
        fig = go.Figure()
        fig.update_layout(
            shapes=shapes,
            annotations=annotations,
            xaxis=dict(visible=False, range=[-total_leaves*2, total_leaves*2]),
            yaxis=dict(visible=False, range=[-10, 2]),
            plot_bgcolor='black',
            paper_bgcolor='black',
            height=600,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

# ---------- Main display (depth increased to 5) ----------
if st.session_state.root_built:
    st.subheader("Tree Diagram")
    draw_tree()

    node_ids = sorted(st.session_state.tree_nodes.keys())
    selected_node = st.selectbox("Select node to view / split", node_ids, format_func=lambda id: f"Node {id}")
    if selected_node is not None:
        node = st.session_state.tree_nodes[selected_node]
        data = node['data']
        depth = node['depth']
        win_rate = data['Target'].mean() * 100
        n = len(data)
        st.write(f"**Node {selected_node}** (depth {depth}, n={n}, win={win_rate:.1f}%)")

        available_features = [f for f in feature_cols if f in data.columns]
        suggestions = suggest_features(data, available_features, top_k=3)
        st.write("**Suggested features:**", ", ".join(suggestions) if suggestions else "None")

        if node['feature'] is None and depth < 5:          # ⬅ increased to 5
            col1, col2 = st.columns([3, 1])
            with col1:
                selected_feature = st.selectbox("Select feature to split", available_features, key=f"feat_{selected_node}")
            with col2:
                split_clicked = st.button("Split", key=f"split_{selected_node}")

            if split_clicked:
                unique_vals = data[selected_feature].dropna().unique()
                if len(unique_vals) < 2:
                    st.warning(f"**{selected_feature}** is constant in this node.")
                    st.write("Distribution:", data[selected_feature].value_counts(dropna=False).to_dict())
                else:
                    best_feat, best_thresh, gain = find_best_split(data, [selected_feature])
                    if best_feat is None or gain <= 0:
                        st.warning(f"No significant split found for '{selected_feature}'. Try another feature.")
                    else:
                        node['feature'] = best_feat
                        node['threshold'] = best_thresh
                        left_mask = data[best_feat] <= best_thresh
                        right_mask = data[best_feat] > best_thresh
                        left_data = data[left_mask].copy()
                        right_data = data[right_mask].copy()
                        left_id = st.session_state.next_node_id
                        right_id = st.session_state.next_node_id + 1
                        st.session_state.next_node_id += 2
                        st.session_state.tree_nodes[left_id] = {
                            'data': left_data, 'feature': None, 'threshold': None, 'children': [], 'depth': depth + 1
                        }
                        st.session_state.tree_nodes[right_id] = {
                            'data': right_data, 'feature': None, 'threshold': None, 'children': [], 'depth': depth + 1
                        }
                        node['children'] = [left_id, right_id]
                        st.rerun()

        elif node['feature'] is not None:
            st.write(f"Currently split on **{node['feature']}**")
            if st.button("Resplit", key=f"resplit_{selected_node}"):
                for child_id in node['children']:
                    if child_id in st.session_state.tree_nodes:
                        del st.session_state.tree_nodes[child_id]
                node['feature'] = None
                node['threshold'] = None
                node['children'] = []
                st.rerun()
