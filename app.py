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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, mutual_info_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cdist

st.set_page_config(page_title="UFC Pre‑Fight Dashboard", layout="wide")

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

    # ---------- DEFENSIVE STATS (opponent's offensive numbers in this fight) ----------
    opp_fight_stats = fight_totals[['FightID','Fighter','SS','SSA','TS','TSA','TD','TDA',
                                    'Subs','Reversals','KD','DSL','DSA','CSL','CSA','GSL','GSA','Ctrl']].copy()
    opp_fight_stats.rename(columns={'Fighter':'Opponent'}, inplace=True)
    for col in ['SS','SSA','TS','TSA','TD','TDA','Subs','Reversals','KD','DSL','DSA',
                'CSL','CSA','GSL','GSA','Ctrl']:
        opp_fight_stats.rename(columns={col: f'Def_{col}'}, inplace=True)
    fight_totals = fight_totals.merge(opp_fight_stats, on=['FightID','Opponent'], how='left')

    # Odds parsing
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

    # Physical differences
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

    # ---------- Career averages ----------
    career_stat_cols = ['SS','SSA','TS','TSA','TD','TDA','Subs','Reversals','KD','DSL','DSA']
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

    avg_cols_off = [f'CareerAvg_{c}' for c in career_stat_cols] + ['CareerWinPct']
    for col in avg_cols_off:
        if col in fight_totals.columns:
            fight_totals[col] = fight_totals.groupby('Fighter')[col].ffill().bfill()

    # ---------- DERIVED CAREER RATIOS ----------
    fight_totals['CareerAvg_TS_Acc'] = (fight_totals['CareerAvg_TS'] / fight_totals['CareerAvg_TSA'].replace(0, np.nan)) * 100
    fight_totals['CareerAvg_TD_Acc'] = (fight_totals['CareerAvg_TD'] / fight_totals['CareerAvg_TDA'].replace(0, np.nan)) * 100
    fight_totals['CareerAvg_DS_Acc'] = (fight_totals['CareerAvg_DSL'] / fight_totals['CareerAvg_DSA'].replace(0, np.nan)) * 100
    fight_totals['CareerAvg_DSL_per_KD'] = fight_totals['CareerAvg_DSL'] / fight_totals['CareerAvg_KD'].replace(0, np.nan)
    fight_totals['CareerAvg_Ctrl_per_TD'] = fight_totals['CareerAvg_Ctrl'] / fight_totals['CareerAvg_TD'].replace(0, np.nan) if 'CareerAvg_Ctrl' in fight_totals.columns else np.nan

    # ---------- Defensive career averages ----------
    def_cols = ['Def_SS','Def_SSA','Def_TS','Def_TSA','Def_TD','Def_TDA',
                'Def_Subs','Def_Reversals','Def_KD','Def_DSL','Def_DSA','Def_Ctrl']
    stats_def = fight_totals[has_stats].copy()
    stats_def.sort_values(['Fighter','FightDate'], inplace=True)
    for col in [c for c in def_cols if c in stats_def.columns]:
        stats_def[f'cum_{col}'] = stats_def.groupby('Fighter')[col].cumsum()
        stats_def[f'prev_cum_{col}'] = stats_def.groupby('Fighter')[f'cum_{col}'].shift(1).fillna(0)
    stats_def['prev_fights_count'] = stats_def.groupby('Fighter').cumcount()
    def_avg_cols = []
    for col in [c for c in def_cols if c in stats_def.columns]:
        stats_def[f'CareerAvg_{col}'] = (stats_def[f'prev_cum_{col}'] / stats_def['prev_fights_count'].replace(0, np.nan))
        def_avg_cols.append(f'CareerAvg_{col}')
    fight_totals = fight_totals.merge(stats_def[['FightID','Fighter'] + def_avg_cols], on=['FightID','Fighter'], how='left')
    for col in def_avg_cols:
        if col in fight_totals.columns:
            fight_totals[col] = fight_totals.groupby('Fighter')[col].ffill().bfill()

    # Defensive career ratios
    if 'CareerAvg_Def_TS' in fight_totals.columns:
        fight_totals['CareerAvg_Def_TS_Acc'] = (fight_totals['CareerAvg_Def_TS'] / fight_totals['CareerAvg_Def_TSA'].replace(0, np.nan)) * 100
        fight_totals['CareerAvg_Def_TD_Acc'] = (fight_totals['CareerAvg_Def_TD'] / fight_totals['CareerAvg_Def_TDA'].replace(0, np.nan)) * 100
        fight_totals['CareerAvg_Def_DS_Acc'] = (fight_totals['CareerAvg_Def_DSL'] / fight_totals['CareerAvg_Def_DSA'].replace(0, np.nan)) * 100
        fight_totals['CareerAvg_Def_DSL_per_KD'] = fight_totals['CareerAvg_Def_DSL'] / fight_totals['CareerAvg_Def_KD'].replace(0, np.nan)
        fight_totals['CareerAvg_Def_Ctrl_per_TD'] = fight_totals['CareerAvg_Def_Ctrl'] / fight_totals['CareerAvg_Def_TD'].replace(0, np.nan)

    # ---------- Opponent career averages ----------
    avg_cols_ext = (
        [f'CareerAvg_{c}' for c in career_stat_cols] +
        ['CareerWinPct'] +
        def_avg_cols
    )
    opp_career = fight_totals[['FightID','Fighter'] + avg_cols_ext].copy()
    opp_career.rename(columns={'Fighter':'Opponent',
                               **{c: f'Opponent_{c}' for c in avg_cols_ext}},
                      inplace=True)
    fight_totals = fight_totals.merge(opp_career, on=['FightID','Opponent'], how='left')

    # Opponent defensive ratios
    if 'Opponent_CareerAvg_Def_TS' in fight_totals.columns:
        fight_totals['Opponent_CareerAvg_Def_TS_Acc'] = (
            fight_totals['Opponent_CareerAvg_Def_TS'] /
            fight_totals['Opponent_CareerAvg_Def_TSA'].replace(0, np.nan) * 100
        )
        fight_totals['Opponent_CareerAvg_Def_TD_Acc'] = (
            fight_totals['Opponent_CareerAvg_Def_TD'] /
            fight_totals['Opponent_CareerAvg_Def_TDA'].replace(0, np.nan) * 100
        )
        fight_totals['Opponent_CareerAvg_Def_DS_Acc'] = (
            fight_totals['Opponent_CareerAvg_Def_DSL'] /
            fight_totals['Opponent_CareerAvg_Def_DSA'].replace(0, np.nan) * 100
        )
        fight_totals['Opponent_CareerAvg_Def_DSL_per_KD'] = (
            fight_totals['Opponent_CareerAvg_Def_DSL'] /
            fight_totals['Opponent_CareerAvg_Def_KD'].replace(0, np.nan)
        )
        fight_totals['Opponent_CareerAvg_Def_Ctrl_per_TD'] = (
            fight_totals['Opponent_CareerAvg_Def_Ctrl'] /
            fight_totals['Opponent_CareerAvg_Def_TD'].replace(0, np.nan)
        )

    # Opponent days & gaps
    opp_days = fight_totals[['FightID','Fighter','DaysSincePrev','Avg3DaysGap']].copy()
    opp_days.rename(columns={
        'Fighter':'Opponent',
        'DaysSincePrev':'Opponent_DaysSincePrev',
        'Avg3DaysGap':'Opponent_Avg3DaysGap'
    }, inplace=True)
    fight_totals = fight_totals.merge(opp_days, on=['FightID','Opponent'], how='left')

    # ---------- Prev7Wins / Prev7Losses ----------
    def prev7_record(group):
        group = group.sort_values('FightDate')
        wins = []
        losses = []
        for i in group.index:
            prev = group.loc[:i-1]
            prev_valid = prev[prev['Win?'].isin(['Yes', 'No'])]
            last7 = prev_valid.tail(7)
            w = (last7['Win?'] == 'Yes').sum()
            l = (last7['Win?'] == 'No').sum()
            wins.append(w)
            losses.append(l)
        return pd.DataFrame({'Prev7Wins': wins, 'Prev7Losses': losses}, index=group.index)

    rec_df = fight_totals.groupby('Fighter', group_keys=False).apply(prev7_record)
    fight_totals = fight_totals.join(rec_df)

    opp_rec = fight_totals[['FightID','Fighter','Prev7Wins','Prev7Losses']].copy()
    opp_rec.rename(columns={'Fighter':'Opponent',
                            'Prev7Wins':'Opponent_Prev7Wins',
                            'Prev7Losses':'Opponent_Prev7Losses'}, inplace=True)
    fight_totals = fight_totals.merge(opp_rec, on=['FightID','Opponent'], how='left')

    # ---------- COLLEY MATRIX RATINGS (fully vectorized, no loops) ----------
    # Only historical fights (Win? not NaN)
    hist = fight_totals[fight_totals['Win?'].notna()][['Fighter','Opponent','Win?']].drop_duplicates()

    # Count games per fighter (appearances in either column)
    all_fighters = pd.concat([hist['Fighter'], hist['Opponent']])
    n_games = all_fighters.value_counts()

    # Wins and losses for the fighter column (when Fighter is the row's Fighter)
    wins = hist[hist['Win?'] == 'Yes'].groupby('Fighter').size()
    losses = hist[hist['Win?'] == 'No'].groupby('Fighter').size()

    # Ensure all fighters have 0 for missing wins/losses
    wins = wins.reindex(n_games.index, fill_value=0)
    losses = losses.reindex(n_games.index, fill_value=0)

    # Build opponent count matrix (how many times each pair fought)
    # Create an undirected edge list with counts
    edges = pd.concat([
        hist[['Fighter','Opponent']].rename(columns={'Fighter':'f1','Opponent':'f2'}),
        hist[['Opponent','Fighter']].rename(columns={'Opponent':'f1','Fighter':'f2'})
    ])
    opp_matrix = edges.groupby(['f1','f2']).size().unstack(fill_value=0)
    # Ensure we have all fighters in both rows and columns
    opp_matrix = opp_matrix.reindex(index=n_games.index, columns=n_games.index, fill_value=0)

    fighters_list = n_games.index.tolist()
    N = len(fighters_list)
    if N == 0:
        fight_totals['FighterColleyRating'] = 0.5
        fight_totals['OpponentColleyRating'] = 0.5
        fight_totals['ColleyRating_Diff'] = 0.0
    else:
        f_to_idx = {f: i for i, f in enumerate(fighters_list)}
        C = np.zeros((N, N))
        b = np.zeros(N)

        # Fill the Colley matrix
        for f in fighters_list:
            i = f_to_idx[f]
            C[i, i] = 2 + n_games[f]
            b[i] = 1 + (wins[f] - losses[f]) / 2

        for f1 in fighters_list:
            i = f_to_idx[f1]
            for f2, cnt in opp_matrix.loc[f1].items():
                if cnt > 0 and f2 in f_to_idx:
                    j = f_to_idx[f2]
                    C[i, j] = -cnt

        # Solve the linear system
        try:
            final_ratings = np.linalg.solve(C + np.eye(N)*1e-6, b)
        except np.linalg.LinAlgError:
            final_ratings = np.linalg.lstsq(C, b, rcond=None)[0]

        rating_map = {f: r for f, r in zip(fighters_list, final_ratings)}
        default_rating = np.median(final_ratings) if len(final_ratings) > 0 else 0.5

        # Vectorized assignment to all rows
        fight_totals['FighterColleyRating'] = fight_totals['Fighter'].map(rating_map).fillna(default_rating)
        fight_totals['OpponentColleyRating'] = fight_totals['Opponent'].map(rating_map).fillna(default_rating)
        fight_totals['ColleyRating_Diff'] = fight_totals['FighterColleyRating'] - fight_totals['OpponentColleyRating']
    # DIFFERENTIALS (add Colley diff)
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
        ('CareerAvg_TS_Acc', 'Opponent_CareerAvg_TS_Acc'),
        ('CareerAvg_TD_Acc', 'Opponent_CareerAvg_TD_Acc'),
        ('CareerAvg_DS_Acc', 'Opponent_CareerAvg_DS_Acc'),
        ('CareerAvg_DSL_per_KD', 'Opponent_CareerAvg_DSL_per_KD'),
        ('CareerAvg_Ctrl_per_TD', 'Opponent_CareerAvg_Ctrl_per_TD'),
        ('CareerAvg_Def_TS_Acc', 'Opponent_CareerAvg_Def_TS_Acc'),
        ('CareerAvg_Def_TD_Acc', 'Opponent_CareerAvg_Def_TD_Acc'),
        ('CareerAvg_Def_DS_Acc', 'Opponent_CareerAvg_Def_DS_Acc'),
        ('CareerAvg_Def_DSL_per_KD', 'Opponent_CareerAvg_Def_DSL_per_KD'),
        ('CareerAvg_Def_Ctrl_per_TD', 'Opponent_CareerAvg_Def_Ctrl_per_TD'),
        ('Prev7Wins', 'Opponent_Prev7Wins'),
        ('Prev7Losses', 'Opponent_Prev7Losses'),
        # Colley diff already computed
    ]
    for f_col, o_col in diff_pairs:
        if f_col in fight_totals.columns and o_col in fight_totals.columns:
            fight_totals[f'{f_col}_Diff'] = fight_totals[f_col] - fight_totals[o_col]

    # Career SS accuracy
    if 'CareerAvg_SS' in fight_totals.columns and 'CareerAvg_SSA' in fight_totals.columns:
        fight_totals['CareerAvg_SS_Acc'] = (
            (fight_totals['CareerAvg_SS'] / fight_totals['CareerAvg_SSA'].replace(0, np.nan)) * 100
        ).round(1)

    fight_totals['prev_fights_count'] = fight_totals['FightNumber'] - 1

    # Previous fight stats (shifts)
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
        opp_df = fight_totals[['FightID','Fighter',col]].dropna(subset=[col])
        opp_df = opp_df.rename(columns={'Fighter':'Opponent', col:f'Opponent_Prev{shift}_Outcome_raw'})
        fight_totals = fight_totals.merge(opp_df, on=['FightID','Opponent'], how='left')

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

# Load all data
all_fights = load_full_data()
all_fights_display = all_fights[all_fights['FightDate'] >= '2015-01-01'].copy()

# Clean categorical columns
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
        cur_odds = (0, 0)
    prev_min = int(all_fights_display['PrevFighterOddsNum'].min()) if not all_fights_display['PrevFighterOddsNum'].isna().all() else 0
    prev_max = int(all_fights_display['PrevFighterOddsNum'].max()) if not all_fights_display['PrevFighterOddsNum'].isna().all() else 0
    if prev_min != prev_max:
        prev_odds = st.slider("Prev Fight Odds", prev_min, prev_max, (prev_min, prev_max), step=10)
    else:
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
        dsl_kd = (avg_dsl / avg_kd) if avg_kd and avg_kd > 0 else 0
        avg_ctrl = subset['CareerAvg_Ctrl'].mean() if 'CareerAvg_Ctrl' in subset else 0
        ctrtd = (avg_ctrl / avg_td) if avg_td and avg_td > 0 else 0
        age_diff_mean = subset['AgeDiff'].mean()
        height_diff_mean = subset['HeightDiff'].mean()
        reach_diff_mean = subset['ReachDiff'].mean()
        win_pct = subset['CareerWinPct'].mean()
        avg_prev_wins = subset['Prev7Wins'].mean() if 'Prev7Wins' in subset else 0
        avg_prev_losses = subset['Prev7Losses'].mean() if 'Prev7Losses' in subset else 0

        st.write(f"**Career Win %:** {win_pct:.1f}%")
        st.write(f"**Prev 7 Record:** {avg_prev_wins:.0f}‑{avg_prev_losses:.0f}")
        st.write(f"**Career Avg SS:** {avg_ss:.1f} / {avg_ssa:.1f} (Acc: {avg_ss_acc:.1f}%)")
        st.write(f"**Career Avg TD:** {avg_td:.1f} / {avg_tda:.1f}")
        st.write(f"**Career Avg Subs:** {avg_subs:.1f} | Rev: {avg_rev:.1f}")
        st.write(f"**Career Avg KD:** {avg_kd:.1f} | DSL/KD: {dsl_kd:.3f}")
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
                pw = int(row['Prev7Wins']) if pd.notna(row['Prev7Wins']) else 0
                pl = int(row['Prev7Losses']) if pd.notna(row['Prev7Losses']) else 0
                st.write(f"**Career Win %:** {row['CareerWinPct']:.1f}% | **Prev 7 Record:** {pw}‑{pl}")
                st.write(f"**Colley Rating:** {row['FighterColleyRating']:.4f}" if pd.notna(row['FighterColleyRating']) else "**Colley Rating:** N/A")
                st.write(f"**Odds (Fighter/Opp):** {row['FighterOddsBFO']} / {row['OpponentOddsBFO']}")

                st.write("**Career Averages (offence):**")
                avg_items = []
                for col_name in ['CareerAvg_SS','CareerAvg_SSA','CareerAvg_KD','CareerAvg_TD','CareerAvg_TDA',
                                 'CareerAvg_Subs','CareerAvg_Reversals','CareerAvg_Ctrl','CareerAvg_DSL']:
                    if col_name in row:
                        val = row[col_name]
                        avg_items.append(f"{col_name.replace('CareerAvg_','')}: {val:.1f}" if pd.notna(val) else f"{col_name.replace('CareerAvg_','')}: --")
                if 'CareerAvg_TS_Acc' in row and pd.notna(row['CareerAvg_TS_Acc']):
                    avg_items.append(f"TS Acc: {row['CareerAvg_TS_Acc']:.1f}%")
                if 'CareerAvg_TD_Acc' in row and pd.notna(row['CareerAvg_TD_Acc']):
                    avg_items.append(f"TD Acc: {row['CareerAvg_TD_Acc']:.1f}%")
                if 'CareerAvg_DSL_per_KD' in row and pd.notna(row['CareerAvg_DSL_per_KD']):
                    avg_items.append(f"DSL/KD: {row['CareerAvg_DSL_per_KD']:.2f}")
                if 'CareerAvg_Ctrl_per_TD' in row and pd.notna(row['CareerAvg_Ctrl_per_TD']):
                    avg_items.append(f"Ctrl/TD: {row['CareerAvg_Ctrl_per_TD']:.1f}s")
                st.write(" · ".join(avg_items) if avg_items else "No career data")

                st.write("**Defensive Averages (opponents' stats against):**")
                def_items = []
                for col_name in ['CareerAvg_Def_TS_Acc','CareerAvg_Def_TD_Acc','CareerAvg_Def_DS_Acc',
                                 'CareerAvg_Def_DSL_per_KD','CareerAvg_Def_Ctrl_per_TD']:
                    if col_name in row and pd.notna(row[col_name]):
                        def_items.append(f"{col_name.replace('CareerAvg_Def_','')}: {row[col_name]:.1f}")
                st.write(" · ".join(def_items) if def_items else "No defensive data")

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

                # Title history – only most recent
                st.write("**Title History:**")
                f_title = row['Prev1_Title'] if pd.notna(row['Prev1_Title']) else '--'
                o_title = row['Opponent_Prev1_Title'] if 'Opponent_Prev1_Title' in row and pd.notna(row['Opponent_Prev1_Title']) else '--'
                st.write(f"Fighter's last fight was a title fight? {f_title}  |  Opponent's last fight was a title fight? {o_title}")
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
                'FighterOddsBFO','OpponentOddsBFO','Prev7Wins','Prev7Losses','FighterColleyRating']
if 'CareerAvg_Ctrl' in data.columns: display_cols.append('CareerAvg_Ctrl')
display_cols = [c for c in display_cols if c in last20.columns]
st.dataframe(last20[display_cols])

# =========================================================================
# COMMON DEFINITIONS
# =========================================================================
core = ['Age', 'Height', 'Reach', 'Age_opp', 'Height_opp', 'Reach_opp',
        'AgeDiff', 'HeightDiff', 'ReachDiff', 'DaysSincePrev', 'Avg3DaysGap',
        'FightNumber', 'Opponent_FightNumber', 'FighterOddsNum', 'PrevFighterOddsNum',
        'CareerWinPct', 'Opponent_CareerWinPct',
        'Prev7Wins', 'Opponent_Prev7Wins', 'Prev7Losses', 'Opponent_Prev7Losses',
        'FighterColleyRating', 'OpponentColleyRating', 'ColleyRating_Diff']
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

# ---------- Bayesian Shrinkage Sliders ----------
prior_weight = st.sidebar.slider("Bayesian prior weight", 0.0, 20.0, 5.0, step=0.5, key="prior_weight_global")
recent_window = st.sidebar.slider("Recent fights window", 1, 100, 50, key="recent_win_global")

# =========================================================================
# 3D LR WIN/LOSS PREDICTION + COMBO BUILDER
# =========================================================================
st.header("3D LR Win/Loss Prediction & Best LR Combinations")

three_d_features = [c for c in numerical_features if c in data.columns and data[c].nunique(dropna=True) >= 2]
if len(three_d_features) >= 3:
    col1, col2, col3 = st.columns(3)
    with col1:
        x_lr = st.selectbox("X", three_d_features, key="lr_x")
    with col2:
        y_lr = st.selectbox("Y", three_d_features, key="lr_y")
    with col3:
        z_lr = st.selectbox("Z", three_d_features, key="lr_z")

    if x_lr and y_lr and z_lr:
        plot_data = data[[x_lr, y_lr, z_lr, 'DetailedResult', 'Fight']].copy()
        plot_data = plot_data.loc[:, ~plot_data.columns.duplicated()].dropna()
        if len(plot_data) < 10:
            st.warning("Not enough data for 3D plot.")
        else:
            fig = px.scatter_3d(
                plot_data,
                x=x_lr, y=y_lr, z=z_lr,
                color='DetailedResult',
                color_discrete_map=color_map,
                hover_data=['Fight'],
                title="3D Scatter – Logistic Regression"
            )
            st.plotly_chart(fig, use_container_width=True)

        hist_base = data[data['Win?'].isin(['Yes','No'])].copy()
        hist_base = hist_base.loc[:, ~hist_base.columns.duplicated()]
        hist_lr = hist_base[[x_lr, y_lr, z_lr, 'Win?']].dropna()

        if len(hist_lr) < 10 or hist_lr['Win?'].nunique() < 2:
            st.warning("Not enough historical data for LR model.")
        else:
            hist_lr['target'] = (hist_lr['Win?'] == 'Yes').astype(int)
            X_lr = hist_lr[[x_lr, y_lr, z_lr]].values
            y_lr_target = hist_lr['target'].values

            lr_model = LogisticRegression(max_iter=1000)
            lr_model.fit(X_lr, y_lr_target)
            y_prob_lr_in = lr_model.predict_proba(X_lr)[:, 1]
            ll_lr = log_loss(y_lr_target, y_prob_lr_in)
            bs_lr = brier_score_loss(y_lr_target, y_prob_lr_in)

            full_hist = data[data['Win?'].isin(['Yes','No'])].sort_values('FightDate')
            if len(full_hist) > 0:
                overall_wr = (full_hist['Win?'] == 'Yes').mean() * 100
                recent = full_hist.tail(recent_window)
                recent_wr = (recent['Win?'] == 'Yes').mean() * 100 if len(recent) > 0 else 0.0
                recent_count = len(recent)
            else:
                overall_wr = recent_wr = 0.0
                recent_count = 0

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("LR Log‑loss", f"{ll_lr:.3f}")
            with col_m2:
                st.metric("LR Brier", f"{bs_lr:.3f}")
            with col_m3:
                st.metric("Overall Win%", f"{overall_wr:.1f}%")
                st.metric(f"Recent Win% (last {recent_window})", f"{recent_wr:.1f}%")

            train_means = {}
            for col in (x_lr, y_lr, z_lr):
                if col in hist_base.columns:
                    train_means[col] = hist_base[col].mean()
                else:
                    train_means[col] = 0

            st.subheader("LR Win Probability Estimate")
            all_upcoming = all_fights_display[
                all_fights_display['Win?'].isna() | (all_fights_display['Win?'] == '')
            ]
            if not all_upcoming.empty:
                up_ids = all_upcoming['FightID'].unique()
                chosen_id = st.selectbox("Select upcoming fight", sorted(up_ids), key="lr_up")
                if chosen_id:
                    up_rows = all_upcoming[all_upcoming['FightID'] == chosen_id]
                    if len(up_rows) == 2:
                        fighter_row = up_rows.iloc[0]

                        def safe_val(col):
                            try:
                                val = fighter_row[col]
                                return val if pd.notna(val) else train_means[col]
                            except (KeyError, ValueError):
                                return train_means[col]

                        v1 = safe_val(x_lr)
                        v2 = safe_val(y_lr)
                        v3 = safe_val(z_lr)

                        up_val = np.array([[v1, v2, v3]])
                        prob_lr = lr_model.predict_proba(up_val)[0, 1]

                        if recent_count > 0:
                            shrunk_recent = (prior_weight * overall_wr + recent_count * recent_wr) / (prior_weight + recent_count)
                        else:
                            shrunk_recent = overall_wr
                        shrunk_prob = (prior_weight * (shrunk_recent / 100) + prob_lr) / (prior_weight + 1)

                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            st.metric("LR win prob", f"{prob_lr:.1%}")
                        with col_p2:
                            st.metric("LR shrunken", f"{shrunk_prob:.1%}")
            else:
                st.write("No upcoming fights available.")

    # --- LR 3‑Variable Combination Builder (Brier) ---
    st.subheader("LR 3‑Variable Combinations (Brier)")
    combo_candidates = [c for c in numerical_features if c != 'FighterOddsNum' and c in data.columns and data[c].nunique(dropna=True) >= 2]

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
    st.warning("Not enough numerical features for a 3D plot (need at least 3).")

# =========================================================================
# 3D KNN WIN/LOSS PREDICTION (WEIGHTED + PLATT) + COMBO BUILDER
# =========================================================================
st.header("3D Weighted KNN Win/Loss Prediction (Platt‑scaled) & Best KNN Combinations")

if len(three_d_features) >= 3:
    col1_knn, col2_knn, col3_knn = st.columns(3)
    with col1_knn:
        x_knn = st.selectbox("X", three_d_features, key="knn_x")
    with col2_knn:
        y_knn = st.selectbox("Y", three_d_features, key="knn_y")
    with col3_knn:
        z_knn = st.selectbox("Z", three_d_features, key="knn_z")

    if x_knn and y_knn and z_knn:
        plot_data_knn = data[[x_knn, y_knn, z_knn, 'DetailedResult', 'Fight']].copy()
        plot_data_knn = plot_data_knn.loc[:, ~plot_data_knn.columns.duplicated()].dropna()
        if len(plot_data_knn) < 10:
            st.warning("Not enough data for 3D plot.")
        else:
            fig_knn = px.scatter_3d(
                plot_data_knn,
                x=x_knn, y=y_knn, z=z_knn,
                color='DetailedResult',
                color_discrete_map=color_map,
                hover_data=['Fight'],
                title="3D Scatter – Weighted KNN"
            )
            st.plotly_chart(fig_knn, use_container_width=True)

        hist_knn = data[data['Win?'].isin(['Yes','No'])].copy()
        hist_knn = hist_knn.loc[:, ~hist_knn.columns.duplicated()]
        def get_first_col(df, col_name):
            if col_name not in df.columns:
                return np.full(len(df), np.nan)
            sub = df[col_name]
            if isinstance(sub, pd.DataFrame):
                return sub.iloc[:, 0].to_numpy(dtype=np.float64, na_value=np.nan)
            return pd.to_numeric(sub, errors='coerce').to_numpy(dtype=np.float64)

        c1 = get_first_col(hist_knn, x_knn)
        c2 = get_first_col(hist_knn, y_knn)
        c3 = get_first_col(hist_knn, z_knn)
        win_col = hist_knn['Win?']
        if isinstance(win_col, pd.DataFrame):
            win_vals = win_col.iloc[:, 0].values
        else:
            win_vals = win_col.values

        train_df = pd.DataFrame({'f1': c1, 'f2': c2, 'f3': c3, 'Win?': win_vals}).dropna()
        if len(train_df) < 10 or train_df['Win?'].nunique() < 2:
            st.warning("Not enough training data for KNN model.")
        else:
            X_train = train_df[['f1','f2','f3']].values.astype(np.float64)
            y_train = (train_df['Win?'] == 'Yes').astype(int).values

            k_knn = st.slider("KNN neighbors (model)", 1, 20, 5, key="knn_model_k")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_train)

            base_knn = KNeighborsClassifier(n_neighbors=k_knn, weights='distance')
            calibrated_knn = CalibratedClassifierCV(base_knn, method='sigmoid', cv=5)
            calibrated_knn.fit(X_scaled, y_train)

            y_prob_in = calibrated_knn.predict_proba(X_scaled)[:, 1]
            y_prob_in = np.clip(y_prob_in, 0.1, 0.9)
            ll_knn = log_loss(y_train, y_prob_in)
            bs_knn = brier_score_loss(y_train, y_prob_in)

            full_hist = data[data['Win?'].isin(['Yes','No'])].sort_values('FightDate')
            overall_wr = (full_hist['Win?'] == 'Yes').mean() * 100 if len(full_hist) > 0 else 0.0
            recent = full_hist.tail(recent_window)
            recent_wr = (recent['Win?'] == 'Yes').mean() * 100 if len(recent) > 0 else 0.0
            recent_count = len(recent)

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("KNN Log‑loss", f"{ll_knn:.3f}")
            with col_m2:
                st.metric("KNN Brier", f"{bs_knn:.3f}")
            with col_m3:
                st.metric("Overall Win%", f"{overall_wr:.1f}%")
                st.metric(f"Recent Win% (last {recent_window})", f"{recent_wr:.1f}%")

            st.subheader("KNN Win Probability Estimate")
            all_upcoming = all_fights_display[all_fights_display['Win?'].isna() | (all_fights_display['Win?'] == '')]
            if not all_upcoming.empty:
                up_ids = all_upcoming['FightID'].unique()
                chosen_id = st.selectbox("Select upcoming fight", sorted(up_ids), key="knn_up")
                if chosen_id:
                    up_rows = all_upcoming[all_upcoming['FightID'] == chosen_id]
                    if len(up_rows) == 2:
                        fighter_row = up_rows.iloc[0]
                        means = X_train.mean(axis=0)
                        vals = []
                        for i, col_name in enumerate([x_knn, y_knn, z_knn]):
                            raw = get_first_col(pd.DataFrame(fighter_row).T, col_name)[0]
                            try:
                                v = float(raw) if pd.notna(raw) else means[i]
                            except (ValueError, TypeError):
                                v = means[i]
                            vals.append(v)
                        up_arr = np.array([vals], dtype=np.float64)
                        up_scaled = scaler.transform(up_arr)
                        prob_knn = calibrated_knn.predict_proba(up_scaled)[0, 1]
                        prob_knn = np.clip(prob_knn, 0.1, 0.9)

                        if recent_count > 0:
                            shrunk_recent = (prior_weight * overall_wr + recent_count * recent_wr) / (prior_weight + recent_count)
                        else:
                            shrunk_recent = overall_wr
                        shrunk_prob = (prior_weight * (shrunk_recent / 100) + prob_knn) / (prior_weight + 1)

                        col_p1, col_p2 = st.columns(2)
                        with col_p1:
                            st.metric("KNN win prob", f"{prob_knn:.1%}")
                        with col_p2:
                            st.metric("KNN shrunken", f"{shrunk_prob:.1%}")
            else:
                st.write("No upcoming fights available.")

    # --- KNN 3‑Variable Combination Builder (IN‑SAMPLE) ---
    st.subheader("KNN 3‑Variable Combinations (Brier, In‑Sample)")
    combo_candidates_knn = [c for c in numerical_features if c != 'FighterOddsNum' and c in data.columns and data[c].nunique(dropna=True) >= 2]
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
    st.warning("Not enough numerical features for a 3D plot (need at least 3).")

# =========================================================================
# FEATURE IMPORTANCE CHARTS (Numerical & Categorical)
# =========================================================================
st.header("Top 20 Feature Importance (Current Filter Set)")
hist_imp = data[data['Win?'].isin(['Yes', 'No'])].copy()
if len(hist_imp) < 10:
    st.warning("Too few historical fights after filtering to compute importance.")
else:
    hist_imp['Target'] = (hist_imp['Win?'] == 'Yes').astype(int)

    # Numerical – include all fighter stats (no opponent prefix, no Diff)
    importance_features = [c for c in numerical_features
                           if not c.startswith('Opponent_')
                           and not c.endswith('_Diff')
                           and not re.match(r'Prev\d+_', c)
                           and c in hist_imp.columns]
    if importance_features:
        X_num = hist_imp[importance_features].dropna()
        if len(X_num) > 10 and X_num.shape[1] > 0:
            imputer = SimpleImputer(strategy='median')
            X_imp = imputer.fit_transform(X_num)
            y_num = hist_imp.loc[X_num.index, 'Target']
            mi = mutual_info_classif(X_imp, y_num, discrete_features=False)
            mi_df_num = pd.DataFrame({'Feature': importance_features, 'Mutual Information': mi}).sort_values('Mutual Information', ascending=False).head(20)
            fig_num = px.bar(mi_df_num, x='Mutual Information', y='Feature', orientation='h',
                             title="Top 20 Fighter Stats by Mutual Information with Win/Loss")
            st.plotly_chart(fig_num, use_container_width=True)
        else:
            st.warning("Not enough complete rows for numerical importance.")
    else:
        st.warning("No numerical features available after filtering.")

    # Categorical – fixed list
    st.subheader("Categorical Feature Importance with Win/Loss")
    potential_cat_cols = ['WC','Stance','Country','EventCountry','Title','ScheduledRounds','HometownFighter','Opponent_Hometown']
    categorical_cols = [c for c in potential_cat_cols if c in hist_imp.columns and hist_imp[c].nunique(dropna=True) > 1]
    if categorical_cols:
        scores = {}
        for col in categorical_cols:
            sub = hist_imp[[col, 'Target']].dropna()
            if sub[col].nunique() < 2:
                continue
            codes, _ = pd.factorize(sub[col])
            scores[col] = mutual_info_score(codes, sub['Target'])
        if scores:
            cat_mi_df = pd.DataFrame({'Feature': list(scores.keys()), 'Mutual Information': list(scores.values())}).sort_values('Mutual Information', ascending=False).head(20)
            fig_cat = px.bar(cat_mi_df, x='Mutual Information', y='Feature', orientation='h',
                             title="Top Categorical Features by Mutual Information with Win/Loss",
                             color_discrete_sequence=['#636efa'])
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.warning("No categorical column had enough variation.")
    else:
        st.warning("No categorical features available after filtering.")

# =========================================================================
# SPIDER CHART – FIGHTER‑SIDE FILTERS + LR + CALIBRATED KNN + SHRINKAGE + SIMILARITY
# =========================================================================
st.header("Fight Similarity & Comparison (Independent Filters)")
st.subheader("Spider Chart Filters (fighter data only)")

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
    spider_new_wc = st.checkbox("New Weight Class", key="spider_new_wc")
    spider_skip_nc = st.checkbox("Skip NC outcomes", key="spider_skip_nc")
    spider_prev_title = st.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"], key="spider_prev_title")

if spider_skip_nc:
    spider_prev1_col = 'Prev1_Outcome_skipNC'; spider_prev2_col = 'Prev2_Outcome_skipNC'; spider_prev3_col = 'Prev3_Outcome_skipNC'
    spider_career1_col = 'Career1_Outcome_skipNC'; spider_career2_col = 'Career2_Outcome_skipNC'; spider_career3_col = 'Career3_Outcome_skipNC'
else:
    spider_prev1_col = 'Prev1_Outcome_raw'; spider_prev2_col = 'Prev2_Outcome_raw'; spider_prev3_col = 'Prev3_Outcome_raw'
    spider_career1_col = 'Career1_Outcome_raw'; spider_career2_col = 'Career2_Outcome_raw'; spider_career3_col = 'Career3_Outcome_raw'

all_outcomes_raw_spider = sorted(all_fights[spider_prev1_col].dropna().unique())
all_outcomes_career_spider = sorted(all_fights[spider_career1_col].dropna().unique())

with st.expander("Previous Outcomes (Spider)"):
    spider_prev1 = st.multiselect("Prev Fight 1", all_outcomes_raw_spider, key="spider_prev1")
    spider_prev2 = st.multiselect("Prev Fight 2", all_outcomes_raw_spider, key="spider_prev2")
    spider_prev3 = st.multiselect("Prev Fight 3", all_outcomes_raw_spider, key="spider_prev3")
    spider_career1 = st.multiselect("Career F1", all_outcomes_career_spider, key="spider_career1")
    spider_career2 = st.multiselect("Career F2", all_outcomes_career_spider, key="spider_career2")
    spider_career3 = st.multiselect("Career F3", all_outcomes_career_spider, key="spider_career3")

spider_data = all_fights_display.copy()
mask = pd.Series(True, index=spider_data.index)
if spider_wc: mask &= spider_data['WC'].isin(spider_wc)
if spider_stance: mask &= spider_data['Stance'].isin(spider_stance)
if spider_country: mask &= spider_data['Country'].isin(spider_country)
if spider_sched_rounds: mask &= spider_data['ScheduledRounds'].isin(spider_sched_rounds)
if spider_title_fight != "All": mask &= spider_data['Title'] == spider_title_fight
if spider_hometown != "All": mask &= spider_data['HometownFighter'] == spider_hometown
if spider_event_country: mask &= spider_data['EventCountry'].isin(spider_event_country)
if spider_new_wc: mask &= spider_data['IsNewWeightClass'] == True
if spider_prev_title != "All": mask &= spider_data['Prev1_Title'] == spider_prev_title
if spider_prev1: mask &= spider_data[spider_prev1_col].isin(spider_prev1)
if spider_prev2: mask &= spider_data[spider_prev2_col].isin(spider_prev2)
if spider_prev3: mask &= spider_data[spider_prev3_col].isin(spider_prev3)
if spider_career1: mask &= spider_data[spider_career1_col].isin(spider_career1)
if spider_career2: mask &= spider_data[spider_career2_col].isin(spider_career2)
if spider_career3: mask &= spider_data[spider_career3_col].isin(spider_career3)

valid_fight_ids = spider_data.loc[mask, 'FightID'].unique()
spider_data = spider_data[spider_data['FightID'].isin(valid_fight_ids)]
spider_upcoming = spider_data[spider_data['Win?'].isna() | (spider_data['Win?'] == '')]

if spider_upcoming.empty:
    st.write("No upcoming fights after spider filters.")
else:
    fight_counts = spider_upcoming.groupby('FightID').size()
    complete_ids = fight_counts[fight_counts == 2].index
    spider_upcoming = spider_upcoming[spider_upcoming['FightID'].isin(complete_ids)]
    if spider_upcoming.empty:
        st.warning("No upcoming fight has both fighters after spider filters.")
    else:
        spider_hist = spider_data[spider_data['Win?'].isin(['Yes','No'])].sort_values('FightDate')
        numeric_cols = [c for c in spider_upcoming.columns if pd.api.types.is_numeric_dtype(spider_upcoming[c])]
        clean_cols = [c for c in numeric_cols if not re.match(r'Prev\d+_', c) and not c.startswith('Opponent_Prev')]
        wanted_keys = [
            'Age', 'Height', 'Reach',
            'DaysSincePrev', 'Avg3DaysGap',
            'FightNumber', 'Opponent_FightNumber',
            'FighterOddsNum', 'PrevFighterOddsNum',
            'CareerWinPct', 'Prev7Wins', 'Prev7Losses', 'Opponent_Prev7Wins', 'Opponent_Prev7Losses',
            'FighterColleyRating', 'OpponentColleyRating', 'ColleyRating_Diff',
            'CareerAvg_', 'Opponent_CareerAvg_',
            '_Diff'
        ]
        spider_vars = sorted([c for c in clean_cols if any(c.startswith(k) or k in c for k in wanted_keys)])
        if not spider_vars:
            st.warning("No numeric variables found.")
        else:
            selected_vars = st.multiselect("Select variables for models", spider_vars, default=spider_vars[:5], max_selections=8, key="spider_vars")
        if selected_vars:
            train_spider = spider_hist.dropna(subset=selected_vars)
            if len(train_spider) < 10 or train_spider['Win?'].nunique() < 2:
                st.warning("Not enough historical data to train models.")
            else:
                train_spider['target'] = (train_spider['Win?'] == 'Yes').astype(int)
                X_train = train_spider[selected_vars].values.astype(np.float64)
                y_train = train_spider['target'].values

                # LR
                lr_spider = LogisticRegression(max_iter=1000)
                lr_spider.fit(X_train, y_train)
                y_prob_lr_in = lr_spider.predict_proba(X_train)[:, 1]
                ll_lr_spider = log_loss(y_train, y_prob_lr_in)
                bs_lr_spider = brier_score_loss(y_train, y_prob_lr_in)

                # KNN (Platt)
                k_spider = st.slider("KNN neighbors", min_value=1, max_value=20, value=5, key="knn_spider")
                scaler_knn = StandardScaler()
                X_scaled = scaler_knn.fit_transform(X_train)
                base_knn = KNeighborsClassifier(n_neighbors=k_spider, weights='distance')
                calibrated_knn = CalibratedClassifierCV(base_knn, method='sigmoid', cv=5)
                calibrated_knn.fit(X_scaled, y_train)
                y_prob_knn_in = calibrated_knn.predict_proba(X_scaled)[:, 1]
                y_prob_knn_in = np.clip(y_prob_knn_in, 0.1, 0.9)
                ll_knn_spider = log_loss(y_train, y_prob_knn_in)
                bs_knn_spider = brier_score_loss(y_train, y_prob_knn_in)

                col_sm1, col_sm2 = st.columns(2)
                with col_sm1:
                    st.metric("LogReg Log‑loss", f"{ll_lr_spider:.3f}")
                    st.metric("LogReg Brier", f"{bs_lr_spider:.3f}")
                with col_sm2:
                    st.metric("KNN Log‑loss", f"{ll_knn_spider:.3f}")
                    st.metric("KNN Brier", f"{bs_knn_spider:.3f}")

                up_ids = sorted(spider_upcoming['FightID'].unique())
                chosen_fight = st.selectbox("Choose an upcoming fight", up_ids, key="spider_fight")
                if chosen_fight:
                    fight_rows = spider_upcoming[spider_upcoming['FightID'] == chosen_fight]
                    f1 = fight_rows.iloc[0]
                    f2 = fight_rows.iloc[1]

                    # Radar
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

                    # Predictions
                    means = X_train.mean(axis=0)
                    up_vals = []
                    for i, var in enumerate(selected_vars):
                        raw = f1[var]
                        try:
                            v = float(raw) if pd.notna(raw) else means[i]
                        except (ValueError, TypeError):
                            v = means[i]
                        up_vals.append(v)
                    up_vec = np.array([up_vals], dtype=np.float64)
                    prob_lr_f1 = lr_spider.predict_proba(up_vec)[0, 1]
                    up_scaled = scaler_knn.transform(up_vec)
                    prob_knn_f1 = calibrated_knn.predict_proba(up_scaled)[0, 1]
                    prob_knn_f1 = np.clip(prob_knn_f1, 0.1, 0.9)

                    overall_wr_spider = (spider_hist['Win?'] == 'Yes').mean() * 100 if len(spider_hist) > 0 else 0.0
                    recent_spider = spider_hist.tail(recent_window)
                    recent_wr_spider = (recent_spider['Win?'] == 'Yes').mean() * 100 if len(recent_spider) > 0 else 0.0
                    recent_count_spider = len(recent_spider)
                    if recent_count_spider > 0:
                        shrunk_recent = (prior_weight * overall_wr_spider + recent_count_spider * recent_wr_spider) / (prior_weight + recent_count_spider)
                    else:
                        shrunk_recent = overall_wr_spider
                    shrunk_lr = (prior_weight * (shrunk_recent / 100) + prob_lr_f1) / (prior_weight + 1)
                    shrunk_knn = (prior_weight * (shrunk_recent / 100) + prob_knn_f1) / (prior_weight + 1)

                    col_sp1, col_sp2, col_sp3 = st.columns(3)
                    with col_sp1:
                        st.metric("LogReg", f"{prob_lr_f1:.1%}")
                        st.metric("LogReg shrunken", f"{shrunk_lr:.1%}")
                    with col_sp2:
                        st.metric("KNN", f"{prob_knn_f1:.1%}")
                        st.metric("KNN shrunken", f"{shrunk_knn:.1%}")
                    with col_sp3:
                        st.metric("Overall Win% (filtered)", f"{overall_wr_spider:.1f}%")
                        st.metric(f"Recent Win% (last {recent_window})", f"{recent_wr_spider:.1f}%")

                    # Similarity
                    st.subheader(f"Most Similar Historical Fights (from last {recent_window} fights)")
                    scaler_sim = StandardScaler()
                    X_scaled_sim = scaler_sim.fit_transform(X_train)
                    up_scaled_sim = scaler_sim.transform(up_vec)
                    dists = cdist(up_scaled_sim, X_scaled_sim, 'euclidean').flatten()
                    sim_scores = 100 * (1 - dists / (dists.max() or 1))
                    sim_df = train_spider[['FightDate', 'Fighter', 'Opponent', 'Win?']].copy()
                    sim_df['Similarity'] = sim_scores.round(1)
                    sim_df = sim_df.sort_values('FightDate', ascending=False).head(recent_window)
                    top_sim = sim_df.sort_values('Similarity', ascending=False).head(20)
                    st.dataframe(top_sim, use_container_width=True)
