import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import gdown
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist

st.set_page_config(page_title="UFC Pre‑Fight Dashboard", layout="wide")

# -----------------------------------------------
# LOAD DATA
# -----------------------------------------------
PARQUET_FILE_ID = "1uIpfbGFmDolA8P2vc15VvA1qbNzWetxf"

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

def get_diff_range(df, col_name):
    if col_name not in df.columns: return -1.0, 1.0
    vals = df[col_name].dropna()
    if len(vals) == 0: return -1.0, 1.0
    return float(vals.min()), float(vals.max())

# ---------- VARIABLE DEFINITIONS ----------
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

numeric_features = [c for c in data.columns
                    if c.endswith('_opp_diff')
                    or (c.startswith('adj_') and c.endswith('_diff'))
                    or c in rating_raw_cols
                    or c in rating_avg7_cols]

abs_rating_cols = [c for c in rating_raw_cols if not c.endswith('Diff')] + \
                  [c for c in rating_avg7_cols if not c.endswith('_diff')]

# Session state
for key, default in [
    ('overall_wr', 0.0), ('recent_wr', 0.0), ('recent_count', 0),
    ('selected_fight_row', None), ('auto_selected_vars', None),
    ('manual_spider_vars', []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------------------------
# LAST 20 COMPLETED FIGHTS
# -----------------------------------------------
st.title("UFC Pre‑Fight Performance Dashboard")
st.header("Last 20 Completed Fights")
completed = data[data['Win?'].notna() & (data['Win?'].astype(str).str.strip() != '')]
last20 = completed.sort_values('FightDate', ascending=False).head(20)
cols = ['FightDate','Fighter','Opponent','Win?','Method','AgeDiff','HeightDiff','ReachDiff','CareerWinPct_diff']
cols = [c for c in cols if c in last20.columns]
st.dataframe(last20[cols], use_container_width=True)

# -----------------------------------------------
# UPCOMING FIGHT MATCHUP
# -----------------------------------------------
st.header("Upcoming Fight Matchup")
upcoming_display = original_data[original_data['Win?'].isna() | (original_data['Win?'] == '')]
st.write(f"**All upcoming fights:** {len(upcoming_display['FightID'].unique())}")

if not upcoming_display.empty:
    upcoming_ids = sorted(upcoming_display['FightID'].unique())
    selected_fight = st.selectbox("Choose an upcoming fight", upcoming_ids, key="upcoming_select")
    if selected_fight:
        fight_rows = upcoming_display[upcoming_display['FightID'] == selected_fight]
        if len(fight_rows) == 2:
            fight_rows = fight_rows.sort_values('Fighter')
            f1 = fight_rows.iloc[0]
            f2 = fight_rows.iloc[1]
            st.session_state.selected_fight_row = f1
            st.write(f"### {f1['Fighter']} vs {f2['Fighter']}")

            sections = {}
            identity_cols = ['WC','Title','ScheduledRounds','Stance','Country','HometownFighter','EventCountry']
            sections["Identity"] = [c for c in identity_cols if c in f1.index]
            physical_cols = ['Age','Height','Reach','AgeDiff','HeightDiff','ReachDiff']
            sections["Physical"] = [c for c in physical_cols if c in f1.index]
            fight_hist_cols = ['FightNumber','DaysSincePrev','Avg3DaysGap','Prev7WinPct','CareerWinPct',
                               'DaysSincePrev_diff','Avg3DaysGap_diff','CareerWinPct_diff','FightNumber_diff']
            sections["Fight History"] = [c for c in fight_hist_cols if c in f1.index]
            sections["Normalized Simple Stats (diff)"] = [c for c in f1.index if c.startswith('adj_') and c.endswith('_diff')]
            odds_cols = ['FighterOddsNum','PrevFighterOddsNum']
            sections["Odds"] = [c for c in odds_cols if c in f1.index]
            sections["Ratings (Raw)"] = [c for c in f1.index if ('Colley' in c or 'Massey' in c) and 'avg7' not in c]
            sections["Ratings (7‑Fight Avg)"] = [c for c in f1.index if 'avg7' in c]
            sections["Striking & Grappling Final Differentials"] = [c for c in f1.index if c.endswith('_opp_diff')]
            sections["Outcomes"] = [c for c in f1.index if 'Outcome' in c]
            other_cols = ['Prev1_Title','IsNewWeightClass','PrevFighterOddsNum']
            sections["Other"] = [c for c in other_cols if c in f1.index]

            rows = []
            for sec_name, cols in sections.items():
                if not cols: continue
                rows.append({"Stat": f"--- {sec_name} ---", f1['Fighter']: "", f2['Fighter']: ""})
                for c in cols:
                    val1 = f1[c]; val2 = f2[c]
                    def fmt(v):
                        if isinstance(v, (int,float)) and pd.notna(v): return f"{v:.2f}"
                        elif pd.isna(v): return ""
                        else: return str(v)
                    rows.append({"Stat": c, f1['Fighter']: fmt(val1), f2['Fighter']: fmt(val2)})
            df_stats = pd.DataFrame(rows)
            st.dataframe(df_stats, use_container_width=True, hide_index=True)

            # Top 10 Differentials
            st.subheader("Top 10 Differentials")
            diffs_f1 = {}
            diffs_f2 = {}
            for c in f1.index:
                if (c.endswith('_opp_diff') or (c.startswith('adj_') and c.endswith('_diff'))):
                    if pd.notna(f1[c]): diffs_f1[c] = f1[c]
                    if pd.notna(f2[c]): diffs_f2[c] = f2[c]
            top10_f1 = sorted(diffs_f1.items(), key=lambda x: x[1], reverse=True)[:10]
            top10_f2 = sorted(diffs_f2.items(), key=lambda x: x[1], reverse=True)[:10]

            colA, colB = st.columns(2)
            with colA:
                st.write(f"**{f1['Fighter']}**")
                if top10_f1:
                    df_f1 = pd.DataFrame(top10_f1, columns=["Stat", "Value"])
                    df_f1["Value"] = df_f1["Value"].apply(lambda x: f"{x:+.2f}")
                    st.dataframe(df_f1, hide_index=True, use_container_width=True)
                else:
                    st.write("No differentials available.")
            with colB:
                st.write(f"**{f2['Fighter']}**")
                if top10_f2:
                    df_f2 = pd.DataFrame(top10_f2, columns=["Stat", "Value"])
                    df_f2["Value"] = df_f2["Value"].apply(lambda x: f"{x:+.2f}")
                    st.dataframe(df_f2, hide_index=True, use_container_width=True)
                else:
                    st.write("No differentials available.")
        else:
            st.warning("Fight data incomplete (expected 2 rows).")
else:
    st.info("No upcoming fights available.")

# -----------------------------------------------
# CACHED FILTER PROCESSING (heavy computation)
# -----------------------------------------------
@st.cache_data
def apply_spider_filters(df, params):
    """
    Apply strict AND filters first, then permutation check on the subset.
    Returns (filtered_df, fighter_mask, opponent_mask).
    The fighter_mask and opponent_mask are built from the row's OWN columns.
    """
    # --- Unpack strict (shared) filter values ---
    wc = params['wc']
    stance = params['stance']
    event_country = params['event_country']
    sched_rounds = params['sched_rounds']
    title_fight = params['title_fight']
    new_wc = params['new_wc']

    age_min, age_max = params['age_min'], params['age_max']
    ad_min, ad_max = params['ad_min'], params['ad_max']
    hd_min, hd_max = params['hd_min'], params['hd_max']
    rd_min, rd_max = params['rd_min'], params['rd_max']
    days_min, days_max = params['days_min'], params['days_max']
    ddiff_min, ddiff_max = params['ddiff_min'], params['ddiff_max']
    avg3_min, avg3_max = params['avg3_min'], params['avg3_max']
    cwp_min, cwp_max = params['cwp_min'], params['cwp_max']

    odds_min, odds_max = params['odds_min'], params['odds_max']
    podds_min, podds_max = params['podds_min'], params['podds_max']

    use_colley = params['use_colley']; colley_range = params.get('colley_range', None)
    use_massey = params['use_massey']; massey_range = params.get('massey_range', None)
    use_wmd = params['use_wmd']; wmd_range = params.get('wmd_range', None)

    # --- Build strict mask (both rows must satisfy) ---
    mask_strict = pd.Series(True, index=df.index)

    def add_strict(condition, col_name=None):
        if condition is None:
            return None
        if col_name and col_name in df.columns:
            return condition | df[col_name].isna()
        return condition

    if wc: mask_strict &= df['WC'].isin(wc)
    if stance: mask_strict &= df['Stance'].isin(stance)
    if sched_rounds: mask_strict &= df['ScheduledRounds'].isin(sched_rounds)
    if title_fight != "All": mask_strict &= df['Title'] == title_fight
    if event_country: mask_strict &= df['EventCountry'].isin(event_country)
    if new_wc and 'IsNewWeightClass' in df.columns: mask_strict &= df['IsNewWeightClass'] == True

    if 'CareerWinPct_diff' in df.columns:
        mask_strict &= add_strict((df['CareerWinPct_diff'] >= cwp_min) & (df['CareerWinPct_diff'] <= cwp_max), 'CareerWinPct_diff')

    for col, (cmin, cmax) in [
        ('Age', (age_min, age_max)), ('AgeDiff', (ad_min, ad_max)),
        ('HeightDiff', (hd_min, hd_max)), ('ReachDiff', (rd_min, rd_max)),
        ('DaysSincePrev', (days_min, days_max)),
        ('DaysSincePrev_diff', (ddiff_min, ddiff_max)),
        ('Avg3DaysGap_diff', (avg3_min, avg3_max)),
        ('FighterOddsNum', (odds_min, odds_max)),
        ('PrevFighterOddsNum', (podds_min, podds_max))
    ]:
        if col in df.columns:
            mask_strict &= add_strict((df[col] >= cmin) & (df[col] <= cmax), col)

    if use_colley and 'ColleyDecayDiff' in df.columns:
        mask_strict &= add_strict((df['ColleyDecayDiff'] >= colley_range[0]) & (df['ColleyDecayDiff'] <= colley_range[1]), 'ColleyDecayDiff')
    if use_massey and 'MasseyFinishDecayDiff' in df.columns:
        mask_strict &= add_strict((df['MasseyFinishDecayDiff'] >= massey_range[0]) & (df['MasseyFinishDecayDiff'] <= massey_range[1]), 'MasseyFinishDecayDiff')
    if use_wmd and 'WeightedMasseyDecayDiff' in df.columns:
        mask_strict &= add_strict((df['WeightedMasseyDecayDiff'] >= wmd_range[0]) & (df['WeightedMasseyDecayDiff'] <= wmd_range[1]), 'WeightedMasseyDecayDiff')

    # Only keep fights where BOTH rows pass strict filters
    fight_ok = mask_strict.groupby(df['FightID']).transform('all')
    df_strict = df[fight_ok].copy()

    # ==================================================================
    # Build identical masks for both sides using row's OWN columns
    # ==================================================================
    def build_side_mask(data_side, side_params):
        """Return a boolean Series based on each row's own attributes."""
        fn_min, fn_max = side_params['fn_min'], side_params['fn_max']
        prev_title = side_params['prev_title']
        hometown = side_params.get('hometown', [])
        country_side = side_params.get('country', [])
        skip_nc = side_params['skip_nc']

        if skip_nc:
            prev1_col = 'Prev1_Outcome_skipNC'
            prev2_col = 'Prev2_Outcome_skipNC'
            prev3_col = 'Prev3_Outcome_skipNC'
            career1_col = 'Career1_Outcome_skipNC'
            career2_col = 'Career2_Outcome_skipNC'
            career3_col = 'Career3_Outcome_skipNC'
        else:
            prev1_col = 'Prev1_Outcome_raw'
            prev2_col = 'Prev2_Outcome_raw'
            prev3_col = 'Prev3_Outcome_raw'
            career1_col = 'Career1_Outcome_raw'
            career2_col = 'Career2_Outcome_raw'
            career3_col = 'Career3_Outcome_raw'

        masks = []
        if 'FightNumber' in data_side.columns:
            masks.append((data_side['FightNumber'] >= fn_min) & (data_side['FightNumber'] <= fn_max))
        else:
            masks.append(pd.Series(True, index=data_side.index))

        def outcome_cond(col, selected):
            if not selected or col not in data_side.columns:
                return pd.Series(True, index=data_side.index)
            cond = pd.Series(False, index=data_side.index)
            if "Win (any)" in selected:
                cond |= data_side[col].str.startswith('Win', na=False)
            if "Loss (any)" in selected:
                cond |= data_side[col].str.startswith('Loss', na=False)
            exact = [s for s in selected if s not in ("Win (any)", "Loss (any)")]
            if exact:
                cond |= data_side[col].isin(exact)
            return cond

        masks.append(outcome_cond(prev1_col, side_params.get('prev1', [])))
        masks.append(outcome_cond(prev2_col, side_params.get('prev2', [])))
        masks.append(outcome_cond(prev3_col, side_params.get('prev3', [])))
        masks.append(outcome_cond(career1_col, side_params.get('career1', [])))
        masks.append(outcome_cond(career2_col, side_params.get('career2', [])))
        masks.append(outcome_cond(career3_col, side_params.get('career3', [])))

        if prev_title != "All" and 'Prev1_Title' in data_side.columns:
            masks.append(data_side['Prev1_Title'].str.strip().str.lower() == prev_title.lower())
        else:
            masks.append(pd.Series(True, index=data_side.index))

        if hometown and 'HometownFighter' in data_side.columns:
            masks.append(data_side['HometownFighter'].isin(hometown))
        else:
            masks.append(pd.Series(True, index=data_side.index))

        if country_side and 'Country' in data_side.columns:
            masks.append(data_side['Country'].isin(country_side))
        else:
            masks.append(pd.Series(True, index=data_side.index))

        mask = masks[0]
        for m in masks[1:]:
            mask &= m
        return mask

    fighter_params = {
        'fn_min': params['fn_min'], 'fn_max': params['fn_max'],
        'prev_title': params['prev_title'],
        'hometown': params['hometown_fighter'],
        'country': params['fighter_country'],
        'skip_nc': params['skip_nc'],
        'prev1': params['prev1'], 'prev2': params['prev2'], 'prev3': params['prev3'],
        'career1': params['career1'], 'career2': params['career2'], 'career3': params['career3']
    }
    opponent_params = {
        'fn_min': params['ofn_min'], 'fn_max': params['ofn_max'],
        'prev_title': params['opp_prev_title'],
        'hometown': params['opp_hometown'],
        'country': params['opponent_country'],
        'skip_nc': params['skip_nc'],
        'prev1': params['opp_prev1'], 'prev2': params['opp_prev2'], 'prev3': params['opp_prev3'],
        'career1': params['opp_career1'], 'career2': params['opp_career2'], 'career3': params['opp_career3']
    }

    fighter_mask = build_side_mask(df_strict, fighter_params)
    opponent_mask = build_side_mask(df_strict, opponent_params)

    def check_permutation(group):
        idx = group.index.tolist()
        if len(idx) != 2:
            return pd.Series(False, index=group.index)
        i1, i2 = idx[0], idx[1]
        perm1 = fighter_mask.loc[i1] and opponent_mask.loc[i2]
        perm2 = fighter_mask.loc[i2] and opponent_mask.loc[i1]
        keep = perm1 or perm2
        return pd.Series([keep, keep], index=group.index)

    perm_ok = df_strict.groupby('FightID', group_keys=False).apply(check_permutation)
    final_mask = perm_ok.reindex(df_strict.index, fill_value=False)
    filtered_df = df_strict[final_mask].copy()

    return filtered_df, fighter_mask, opponent_mask

# -----------------------------------------------
# SPIDER SECTION (UI + cached processing)
# -----------------------------------------------
st.header("Fight Similarity (Independent Filters)")

with st.expander("Spider Filters", expanded=True):
    # ---- General ----
    with st.expander("General", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            wc = st.multiselect("Weight Class", sorted(original_data['WC'].dropna().unique()), key="spider_wc")
            stance = st.multiselect("Stance", sorted(original_data['Stance'].dropna().unique()), key="spider_stance")
            event_country = st.multiselect("Event Country", sorted(original_data['EventCountry'].dropna().unique()), key="spider_event")
        with col2:
            sched_rounds = st.multiselect("Scheduled Rounds", sorted(original_data['ScheduledRounds'].dropna().unique()), key="spider_sched")
            title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"], key="spider_title")
            new_wc = st.checkbox("New Weight Class", key="spider_new_wc")

    colF, colO = st.columns(2)
    with colF:
        st.write("**Fighter‑Side Filters**")
        fighter_country = st.multiselect("Fighter Country", sorted(original_data['Country'].dropna().unique()), key="spider_fighter_country")
        hometown_fighter = st.multiselect("Hometown", sorted(original_data['HometownFighter'].dropna().unique()), key="spider_hometown")
        fn_min = st.number_input("Fighter Min Fight #", value=1, min_value=1, max_value=int(original_data['FightNumber'].max()), key="spider_fn_min")
        fn_max = st.number_input("Fighter Max Fight #", value=int(original_data['FightNumber'].max()), key="spider_fn_max")
        prev_title = st.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"], key="spider_prev_title")
    with colO:
        st.write("**Opponent‑Side Filters**")
        opponent_country = st.multiselect("Opponent Country", sorted(original_data['Country'].dropna().unique()), key="spider_opponent_country")
        opp_hometown = st.multiselect("Opponent Hometown", sorted(original_data['HometownFighter'].dropna().unique()), key="spider_opp_hometown")
        ofn_min = st.number_input("Opponent Min Fight #", value=1, key="spider_ofn_min")
        ofn_max = st.number_input("Opponent Max Fight #", value=int(original_data['FightNumber'].max()), key="spider_ofn_max")
        opp_prev_title = st.selectbox("Opponent Prev Fight Title?", ["All", "Yes", "No"], key="spider_opp_prev_title")

    with st.expander("Physical & Days", expanded=False):
        age_min, age_max = st.slider("Age", int(original_data['Age'].min()), int(original_data['Age'].max()), (int(original_data['Age'].min()), int(original_data['Age'].max())), key="spider_age")
        ad_min, ad_max = st.slider("Age Diff", int(original_data['AgeDiff'].min()), int(original_data['AgeDiff'].max()), (int(original_data['AgeDiff'].min()), int(original_data['AgeDiff'].max())), key="spider_age_diff")
        hd_min, hd_max = st.slider("Height Diff", int(original_data['HeightDiff'].min()), int(original_data['HeightDiff'].max()), (int(original_data['HeightDiff'].min()), int(original_data['HeightDiff'].max())), key="spider_hd")
        rd_min, rd_max = st.slider("Reach Diff", int(original_data['ReachDiff'].min()), int(original_data['ReachDiff'].max()), (int(original_data['ReachDiff'].min()), int(original_data['ReachDiff'].max())), key="spider_rd")
        days_min, days_max = st.slider("Days Since Prev", int(original_data['DaysSincePrev'].min()), int(original_data['DaysSincePrev'].max()), (int(original_data['DaysSincePrev'].min()), int(original_data['DaysSincePrev'].max())), key="spider_days")
        ddiff_min, ddiff_max = st.slider("Days Since Prev Diff", int(original_data['DaysSincePrev_diff'].min()), int(original_data['DaysSincePrev_diff'].max()), (int(original_data['DaysSincePrev_diff'].min()), int(original_data['DaysSincePrev_diff'].max())), key="spider_ddiff")
        avg3_min, avg3_max = st.slider("Avg3DaysGap Diff", int(original_data['Avg3DaysGap_diff'].min()), int(original_data['Avg3DaysGap_diff'].max()), (int(original_data['Avg3DaysGap_diff'].min()), int(original_data['Avg3DaysGap_diff'].max())), key="spider_avg3")
        cwp_min, cwp_max = st.slider("Career Win % Diff", -100, 100, (-100, 100), step=5, key="spider_cwp")

    with st.expander("Odds", expanded=False):
        odds_min, odds_max = st.slider("Fighter Odds", int(original_data['FighterOddsNum'].min()), int(original_data['FighterOddsNum'].max()), (int(original_data['FighterOddsNum'].min()), int(original_data['FighterOddsNum'].max())), step=10, key="spider_odds")
        podds_min, podds_max = st.slider("Prev Fighter Odds", int(original_data['PrevFighterOddsNum'].min()), int(original_data['PrevFighterOddsNum'].max()), (int(original_data['PrevFighterOddsNum'].min()), int(original_data['PrevFighterOddsNum'].max())), step=10, key="spider_podds")

    with st.expander("Previous Outcomes", expanded=False):
        skip_nc = st.checkbox("Skip NC outcomes", key="spider_skip_nc")
        if skip_nc:
            prev1_col = 'Prev1_Outcome_skipNC'; prev2_col = 'Prev2_Outcome_skipNC'; prev3_col = 'Prev3_Outcome_skipNC'
            career1_col = 'Career1_Outcome_skipNC'; career2_col = 'Career2_Outcome_skipNC'; career3_col = 'Career3_Outcome_skipNC'
        else:
            prev1_col = 'Prev1_Outcome_raw'; prev2_col = 'Prev2_Outcome_raw'; prev3_col = 'Prev3_Outcome_raw'
            career1_col = 'Career1_Outcome_raw'; career2_col = 'Career2_Outcome_raw'; career3_col = 'Career3_Outcome_raw'

        all_outcomes_raw = sorted(original_data[prev1_col].dropna().unique())
        all_outcomes_career = sorted(original_data[career1_col].dropna().unique())
        outcome_options_raw = all_outcomes_raw + ["Win (any)", "Loss (any)"]
        outcome_options_career = all_outcomes_career + ["Win (any)", "Loss (any)"]

        col_f, col_o = st.columns(2)
        with col_f:
            st.write("**Fighter Outcomes**")
            prev1 = st.multiselect("Prev Fight 1", outcome_options_raw, key="spider_prev1")
            prev2 = st.multiselect("Prev Fight 2", outcome_options_raw, key="spider_prev2")
            prev3 = st.multiselect("Prev Fight 3", outcome_options_raw, key="spider_prev3")
            career1 = st.multiselect("Career F1", outcome_options_career, key="spider_career1")
            career2 = st.multiselect("Career F2", outcome_options_career, key="spider_career2")
            career3 = st.multiselect("Career F3", outcome_options_career, key="spider_career3")
        with col_o:
            st.write("**Opponent Outcomes**")
            opp_prev1 = st.multiselect("Opp Prev Fight 1", outcome_options_raw, key="spider_opp_prev1")
            opp_prev2 = st.multiselect("Opp Prev Fight 2", outcome_options_raw, key="spider_opp_prev2")
            opp_prev3 = st.multiselect("Opp Prev Fight 3", outcome_options_raw, key="spider_opp_prev3")
            opp_career1 = st.multiselect("Opp Career F1", outcome_options_career, key="spider_opp_career1")
            opp_career2 = st.multiselect("Opp Career F2", outcome_options_career, key="spider_opp_career2")
            opp_career3 = st.multiselect("Opp Career F3", outcome_options_career, key="spider_opp_career3")

    with st.expander("Ratings", expanded=False):
        use_colley = st.checkbox("Filter ColleyDecayDiff", value=False, key="spider_use_colley")
        colley_range = None
        if use_colley:
            min_cd, max_cd = get_diff_range(original_data, 'ColleyDecayDiff')
            colley_range = st.slider("ColleyDecayDiff range", min_cd, max_cd, (min_cd, max_cd), step=0.01, key="spider_colley")

        use_massey = st.checkbox("Filter MasseyFinishDecayDiff", value=False, key="spider_use_massey")
        massey_range = None
        if use_massey:
            min_md, max_md = get_diff_range(original_data, 'MasseyFinishDecayDiff')
            massey_range = st.slider("MasseyFinishDecayDiff range", min_md, max_md, (min_md, max_md), step=0.01, key="spider_massey")

        use_wmd = st.checkbox("Filter WeightedMasseyDecayDiff", value=False, key="spider_use_wmd")
        wmd_range = None
        if use_wmd:
            min_wmd, max_wmd = get_diff_range(original_data, 'WeightedMasseyDecayDiff')
            wmd_range = st.slider("WeightedMasseyDecayDiff range", min_wmd, max_wmd, (min_wmd, max_wmd), step=0.01, key="spider_wmd")

filter_params = {
    'wc': wc, 'stance': stance, 'event_country': event_country,
    'sched_rounds': sched_rounds, 'title_fight': title_fight, 'new_wc': new_wc,
    'fighter_country': fighter_country, 'opponent_country': opponent_country,
    'hometown_fighter': hometown_fighter, 'opp_hometown': opp_hometown,
    'fn_min': fn_min, 'fn_max': fn_max, 'ofn_min': ofn_min, 'ofn_max': ofn_max,
    'prev_title': prev_title, 'opp_prev_title': opp_prev_title,
    'age_min': age_min, 'age_max': age_max, 'ad_min': ad_min, 'ad_max': ad_max,
    'hd_min': hd_min, 'hd_max': hd_max, 'rd_min': rd_min, 'rd_max': rd_max,
    'days_min': days_min, 'days_max': days_max,
    'ddiff_min': ddiff_min, 'ddiff_max': ddiff_max,
    'avg3_min': avg3_min, 'avg3_max': avg3_max,
    'cwp_min': cwp_min, 'cwp_max': cwp_max,
    'odds_min': odds_min, 'odds_max': odds_max,
    'podds_min': podds_min, 'podds_max': podds_max,
    'skip_nc': skip_nc,
    'prev1': prev1, 'prev2': prev2, 'prev3': prev3,
    'career1': career1, 'career2': career2, 'career3': career3,
    'opp_prev1': opp_prev1, 'opp_prev2': opp_prev2, 'opp_prev3': opp_prev3,
    'opp_career1': opp_career1, 'opp_career2': opp_career2, 'opp_career3': opp_career3,
    'use_colley': use_colley, 'colley_range': colley_range,
    'use_massey': use_massey, 'massey_range': massey_range,
    'use_wmd': use_wmd, 'wmd_range': wmd_range,
}

spider_data, fighter_mask_spider, opponent_mask_spider = apply_spider_filters(original_data, filter_params)

spider_upcoming = spider_data[spider_data['Win?'].isna() | (spider_data['Win?'] == '')]
spider_hist = spider_data[spider_data['Win?'].isin(['Yes','No'])].copy()

if len(spider_hist) > 0:
    fighter_mask_aligned = fighter_mask_spider.loc[spider_hist.index]
    spider_hist = spider_hist[fighter_mask_aligned]

total_completed_fights = spider_hist['FightID'].nunique()
total_wins = (spider_hist['Win?'] == 'Yes').sum()
filtered_wr = total_wins / len(spider_hist) * 100 if len(spider_hist) > 0 else 0.0

if spider_upcoming.empty:
    st.write("No upcoming fights for similarity.")
else:
    fight_counts = spider_upcoming.groupby('FightID').size()
    complete_ids = fight_counts[fight_counts == 2].index
    spider_upcoming = spider_upcoming[spider_upcoming['FightID'].isin(complete_ids)]

    if spider_upcoming.empty:
        st.warning("No upcoming fight has both fighters after similarity filters.")
    else:
        sim_features = [c for c in numeric_features if c in spider_data.columns and c not in abs_rating_cols]
        if not sim_features:
            st.warning("No numeric features for similarity.")
        else:
            up_ids = sorted(spider_upcoming['FightID'].unique())
            if 'prev_spider_fight' not in st.session_state:
                st.session_state.prev_spider_fight = None

            selected_fight_spider = st.selectbox("Choose an upcoming fight for similarity",
                                                up_ids, key="spider_fight_select")

            fight_changed = selected_fight_spider != st.session_state.prev_spider_fight
            if fight_changed:
                st.session_state.prev_spider_fight = selected_fight_spider
                st.session_state.auto_vars = None
                st.session_state.manual_spider_vars = []

            if selected_fight_spider:
                fight_rows = spider_upcoming[spider_upcoming['FightID'] == selected_fight_spider]
                fight_rows = fight_rows.sort_values('Fighter')
                f1 = fight_rows.iloc[0]
                f2 = fight_rows.iloc[1]

                mask_f = fighter_mask_spider.loc[fight_rows.index]
                match_info = []
                for i, row in fight_rows.iterrows():
                    match_info.append(f"{row['Fighter']}: {'Yes' if mask_f.loc[i] else 'No'}")
                st.write("**Matches fighter filters:**  " + "  |  ".join(match_info))

                st.write("**Auto‑select top differentials for similarity**")
                c_slider1, c_slider2 = st.columns(2)
                top_n_f1 = c_slider1.slider("Top N (Fighter 1)", 0, 10, 0, key="top_n_f1")
                top_n_f2 = c_slider2.slider("Top N (Fighter 2)", 0, 10, 0, key="top_n_f2")

                if top_n_f1 > 0 or top_n_f2 > 0:
                    diff_cols = [c for c in f1.index if c.endswith('_opp_diff')]
                    f1_diffs = {c: abs(f1[c]) for c in diff_cols if pd.notna(f1[c])}
                    f2_diffs = {c: abs(f2[c]) for c in diff_cols if pd.notna(f2[c])}
                    top_f1 = sorted(f1_diffs, key=f1_diffs.get, reverse=True)[:top_n_f1]
                    top_f2 = sorted(f2_diffs, key=f2_diffs.get, reverse=True)[:top_n_f2]
                    auto_vars = list(set(top_f1 + top_f2).intersection(sim_features))
                    st.session_state.auto_vars = auto_vars if auto_vars else None
                else:
                    top_n_mi = st.slider("Top N by Feature Importance", 0, 10, 0,
                                        help="Select top variables by mutual information with win/loss")
                    if top_n_mi > 0:
                        @st.cache_data
                        def compute_mi_ranking(df_hist, features):
                            tmp = df_hist.copy()
                            y = (tmp['Win?'] == 'Yes').astype(int)
                            available = [c for c in features if c in tmp.columns and np.issubdtype(tmp[c].dtype, np.number)]
                            if not available:
                                return pd.Series(dtype=float)
                            missing_frac = tmp[available].isna().mean()
                            keep_features = missing_frac[missing_frac <= 0.8].index.tolist()
                            if not keep_features:
                                return pd.Series(dtype=float)
                            X = tmp[keep_features].copy()
                            imp = SimpleImputer(strategy='median')
                            X_imp = imp.fit_transform(X)
                            if len(y) < 5:
                                return pd.Series(dtype=float)
                            mi = mutual_info_classif(X_imp, y, discrete_features=False, random_state=42)
                            return pd.Series(mi, index=keep_features).sort_values(ascending=False)

                        mi_series = compute_mi_ranking(spider_hist, sim_features)
                        if mi_series.empty:
                            st.warning("Not enough complete data for feature importance.")
                            st.session_state.auto_vars = None
                        else:
                            mi_vars = mi_series.head(top_n_mi).index.tolist()
                            st.session_state.auto_vars = mi_vars
                    else:
                        st.session_state.auto_vars = None

                st.write("**Optional: manually add variables**")
                st.multiselect(
                    "Select additional variables",
                    sim_features,
                    key="manual_spider_vars"
                )

                auto_list = st.session_state.auto_vars if st.session_state.auto_vars else []
                manual_list = st.session_state.manual_spider_vars
                combined = list(set(auto_list + manual_list).intersection(sim_features))

                if not combined:
                    st.warning("No similarity variables selected.")
                else:
                    available_metrics = ["Euclidean", "Manhattan", "Chebyshev"]
                    distance_metrics = st.multiselect("Distance metrics", available_metrics,
                                                     default=["Euclidean"], key="spider_metrics")
                    if not distance_metrics:
                        st.warning("Please select at least one distance metric.")
                    else:
                        hist_sub = spider_hist[combined].dropna()
                        if len(hist_sub) < 2:
                            st.warning("Not enough historical data for the selected variables.")
                        else:
                            scaler_sim = StandardScaler()
                            scaler_sim.fit(hist_sub)

                            up_vals = [float(f1.get(var, 0.0)) for var in combined]
                            up_vec = np.array([up_vals], dtype=np.float64)
                            up_scaled = scaler_sim.transform(up_vec)
                            hist_scaled = scaler_sim.transform(hist_sub)

                            metric_map = {"Euclidean": "euclidean", "Manhattan": "cityblock", "Chebyshev": "chebyshev"}
                            metric_similarities = {}
                            for metric_display in distance_metrics:
                                metric = metric_map[metric_display]
                                dists = cdist(up_scaled, hist_scaled, metric=metric).flatten()
                                max_dist = dists.max() if dists.max() > 0 else 1.0
                                sim = 100 * (1 - dists / max_dist)
                                metric_similarities[metric_display] = sim

                            combined_sim = sum(metric_similarities.values()) / len(metric_similarities)

                            sim_df = spider_hist.loc[hist_sub.index, ['FightDate', 'Fighter', 'Opponent', 'Win?']].copy()
                            for metric_display in distance_metrics:
                                sim_df[f'Sim_{metric_display}'] = metric_similarities[metric_display].round(1)
                            sim_df['Similarity'] = combined_sim.round(1)
                            sim_df = sim_df.sort_values('Similarity', ascending=False)

                            top_n = sim_df.head(100)
                            count = len(top_n)

                            total_hist_count = len(sim_df)
                            cm1, cm2 = st.columns(2)
                            cm1.metric("Total Completed Fights", total_completed_fights)
                            cm2.metric("Win Rate (filtered)", f"{filtered_wr:.1f}%")

                            st.subheader("Similarity Metrics (Top 100 Fights)")

                            avg_sim = top_n['Similarity'].mean()
                            total_sim = top_n['Similarity'].sum()
                            composite = avg_sim * (count ** 0.5) / 100

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Count", count)
                            col2.metric("Avg Similarity", f"{avg_sim:.1f}%")
                            col3.metric("Total Similarity", f"{total_sim:.1f}")
                            col4.metric("Composite Score", f"{composite:.1f}")

                            weight_power = st.slider("Weighting Exponent (higher = more impact from high scores)",
                                                     1.0, 4.0, 1.0, 0.5, key="weight_power")

                            def weighted_win_rate(subset):
                                if len(subset) == 0:
                                    return 0.0, 0.0
                                wins = subset['Win?'] == 'Yes'
                                weights = subset['Similarity'] ** weight_power
                                total_weight = weights.sum()
                                win_weight = weights[wins].sum() if wins.any() else 0.0
                                wr = wins.mean() * 100
                                wwr = (win_weight / total_weight) * 100 if total_weight > 0 else 0.0
                                return wr, wwr

                            high_sim_90 = top_n[top_n['Similarity'] >= 90]
                            high_sim_80 = top_n[top_n['Similarity'] >= 80]
                            high_sim_70 = top_n[top_n['Similarity'] >= 70]
                            high_sim_60 = top_n[top_n['Similarity'] >= 60]

                            win_rate_90, weighted_wr_90 = weighted_win_rate(high_sim_90)
                            win_rate_80, weighted_wr_80 = weighted_win_rate(high_sim_80)
                            win_rate_70, weighted_wr_70 = weighted_win_rate(high_sim_70)
                            win_rate_60, weighted_wr_60 = weighted_win_rate(high_sim_60)

                            col5, col6, col7, col8 = st.columns(4)
                            col5.metric("Win Rate (≥90%)", f"{win_rate_90:.1f}%", delta=f"{len(high_sim_90)} fights")
                            col6.metric("Weighted WR (≥90%)", f"{weighted_wr_90:.1f}%")
                            col7.metric("Win Rate (≥80%)", f"{win_rate_80:.1f}%", delta=f"{len(high_sim_80)} fights")
                            col8.metric("Weighted WR (≥80%)", f"{weighted_wr_80:.1f}%")

                            col9, col10, col11, col12 = st.columns(4)
                            col9.metric("Win Rate (≥70%)", f"{win_rate_70:.1f}%", delta=f"{len(high_sim_70)} fights")
                            col10.metric("Weighted WR (≥70%)", f"{weighted_wr_70:.1f}%")
                            col11.metric("Win Rate (≥60%)", f"{win_rate_60:.1f}%", delta=f"{len(high_sim_60)} fights")
                            col12.metric("Weighted WR (≥60%)", f"{weighted_wr_60:.1f}%")

                            fig_hist = px.histogram(sim_df, x='Similarity', nbins=20, title="Similarity Distribution (Combined)")
                            st.plotly_chart(fig_hist, use_container_width=True, key="sim_hist_chart")

                            st.subheader(f"Top 100 Most Similar Historical Fights")
                            col_order = ['FightDate','Fighter','Opponent','Win?'] + [f'Sim_{m}' for m in distance_metrics] + ['Similarity']
                            st.dataframe(top_n[col_order], use_container_width=True)

                # ========== ORIGINAL DECISION TREE (ALL FEATURES) ==========
                st.subheader("Decision Tree (All Differential Features)")

                spider_tree_hist = spider_hist.copy()
                if len(spider_tree_hist) < 10:
                    st.warning("Not enough historical fights for decision tree.")
                else:
                    spider_tree_hist['Target'] = (spider_tree_hist['Win?'] == 'Yes').astype(int)
                    spider_features = [c for c in numeric_features if c in spider_data.columns and c not in abs_rating_cols]
                    if not spider_features:
                        st.warning("No features available for decision tree.")
                    else:
                        X_sp = spider_tree_hist[spider_features].fillna(spider_tree_hist[spider_features].median())
                        y_sp = spider_tree_hist['Target']

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            max_depth_sp = st.slider("Max Depth", 1, 10, 3, key="spider_tree_depth")
                        with col2:
                            min_samples_leaf_sp = st.slider("Min Samples Leaf", 1, 100, 5, key="spider_tree_leaf")
                        with col3:
                            criterion_sp = st.selectbox("Splitting Criterion", ["gini", "entropy", "log_loss"], index=0, key="spider_tree_criterion")

                        if st.button("Train Decision Tree", key="train_spider_tree"):
                            with st.spinner("Training..."):
                                dt_sp = DecisionTreeClassifier(max_depth=max_depth_sp, min_samples_leaf=min_samples_leaf_sp,
                                                               criterion=criterion_sp, random_state=42)
                                dt_sp.fit(X_sp, y_sp)

                                st.subheader("Prediction for Selected Upcoming Fight")
                                if len(fight_rows) == 2:
                                    f1_row = fight_rows.iloc[0]
                                    input_vals = []
                                    for c in spider_features:
                                        val = f1_row.get(c, np.nan)
                                        if pd.isna(val):
                                            val = spider_tree_hist[c].median()
                                        input_vals.append(val)
                                    X_input = np.array([input_vals])
                                    try:
                                        prob = dt_sp.predict_proba(X_input)[0, 1]
                                        leaf = dt_sp.apply(X_input)[0]
                                        st.write(f"**{f1_row['Fighter']}** → leaf **{leaf}** with win probability **{prob:.1%}**")
                                    except Exception as e:
                                        st.error(f"Prediction error: {e}")
                                else:
                                    st.warning("Fight data incomplete for prediction.")

                                fig_w = max(16, max_depth_sp * 5)
                                fig_h = max(8,  max_depth_sp * 3)
                                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                                plot_tree(dt_sp, feature_names=spider_features, class_names=['Loss', 'Win'],
                                          filled=True, rounded=True, proportion=False,
                                          impurity=False, fontsize=8, ax=ax)

                                for text_obj, node_id in zip(ax.texts, range(dt_sp.tree_.node_count)):
                                    old_text = text_obj.get_text()
                                    if 'value' not in old_text:
                                        continue
                                    lines = old_text.split('\n')
                                    new_lines = []
                                    for line in lines:
                                        if line.strip().startswith('value'):
                                            values = dt_sp.tree_.value[node_id][0]
                                            total = values.sum()
                                            if total > 0:
                                                win_pct  = values[1] / total * 100
                                                loss_pct = values[0] / total * 100
                                                new_lines.append(f"Loss {loss_pct:.0f}%  Win {win_pct:.0f}%")
                                            else:
                                                new_lines.append(line)
                                        else:
                                            new_lines.append(line)
                                    text_obj.set_text('\n'.join(new_lines))
                                st.pyplot(fig)

                                st.subheader("Leaf Win Percentages")
                                leaf_ids = dt_sp.apply(X_sp)
                                leaf_stats = []
                                for leaf_id in np.unique(leaf_ids):
                                    mask_leaf = leaf_ids == leaf_id
                                    leaf_stats.append({
                                        "Leaf": leaf_id,
                                        "Samples": mask_leaf.sum(),
                                        "Win Rate": f"{y_sp[mask_leaf].mean() * 100:.1f}%"
                                    })
                                leaf_df = pd.DataFrame(leaf_stats)
                                st.dataframe(leaf_df, use_container_width=True, hide_index=True)

                # ========== MODEL COMPARISON SECTION (FIXED) ==========
                st.header("Model Comparison (Filtered Historical Data)")

                # All differential features (same as first tree)
                all_diff_features = [c for c in numeric_features if c in spider_data.columns and c not in abs_rating_cols]
                top_diff_features = st.session_state.auto_vars if st.session_state.auto_vars else []

                if len(spider_hist) < 10:
                    st.warning("Not enough historical fights for model training.")
                else:
                    y = (spider_hist['Win?'] == 'Yes').astype(int)

                    # Helper to safely prepare feature matrix
                    def safe_prepare(df, features):
                        """Fill missing/infinite values: replace inf -> NaN, then fill NaNs with median or 0."""
                        if not features:
                            return pd.DataFrame(index=df.index)
                        X = df[features].copy()
                        X.replace([np.inf, -np.inf], np.nan, inplace=True)
                        for col in X.columns:
                            if X[col].isna().all():
                                X[col] = 0.0
                            else:
                                med = X[col].median()
                                X[col] = X[col].fillna(med)
                        return X

                    X_all = safe_prepare(spider_hist, all_diff_features)

                    # ---- Decision Tree using only TOP SLIDER variables ----
                    if top_diff_features:
                        st.subheader("Decision Tree (Top‑Slider Variables Only)")
                        X_top = safe_prepare(spider_hist, top_diff_features)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            max_depth_top = st.slider("Max Depth (Top‑Var Tree)", 1, 10, 3, key="top_tree_depth")
                        with c2:
                            min_samples_leaf_top = st.slider("Min Samples Leaf (Top‑Var Tree)", 1, 100, 5, key="top_tree_leaf")
                        with c3:
                            criterion_top = st.selectbox("Splitting Criterion (Top‑Var Tree)", ["gini", "entropy", "log_loss"], index=0, key="top_tree_criterion")

                        if st.button("Train Top‑Var Decision Tree", key="train_top_tree"):
                            with st.spinner("Training..."):
                                dt_top = DecisionTreeClassifier(max_depth=max_depth_top,
                                                                min_samples_leaf=min_samples_leaf_top,
                                                                criterion=criterion_top, random_state=42)
                                dt_top.fit(X_top, y)
                                st.success(f"Top‑Var Tree trained. Training accuracy: {dt_top.score(X_top, y):.1%}")
                                if max_depth_top <= 4:
                                    fig, ax = plt.subplots(figsize=(12, 6))
                                    plot_tree(dt_top, feature_names=top_diff_features, class_names=['Loss','Win'],
                                              filled=True, rounded=True, fontsize=8, ax=ax)
                                    st.pyplot(fig)
                    else:
                        st.info("No top‑slider variables selected. Skipping Top‑Var models.")

                    # ---- Logistic Regression models ----
                    st.subheader("Logistic Regression Win Rates")
                    lr_all = LogisticRegression(max_iter=2000, random_state=42)
                    lr_all.fit(X_all, y)
                    acc_all_lr = lr_all.score(X_all, y)

                    if top_diff_features:
                        X_top_lr = safe_prepare(spider_hist, top_diff_features)
                        lr_top = LogisticRegression(max_iter=2000, random_state=42)
                        lr_top.fit(X_top_lr, y)
                        acc_top_lr = lr_top.score(X_top_lr, y)
                    else:
                        acc_top_lr = None

                    col_lr1, col_lr2 = st.columns(2)
                    col_lr1.metric("LR (All Diff Vars) Win Rate", f"{acc_all_lr:.1%}")
                    if acc_top_lr is not None:
                        col_lr2.metric("LR (Top‑Slider Vars) Win Rate", f"{acc_top_lr:.1%}")
                    else:
                        col_lr2.write("N/A")

                    # ---- Random Forest models ----
                    st.subheader("Random Forest Win Rates")
                    rf_all = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
                    rf_all.fit(X_all, y)
                    acc_all_rf = rf_all.score(X_all, y)

                    if top_diff_features:
                        X_top_rf = safe_prepare(spider_hist, top_diff_features)
                        rf_top = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
                        rf_top.fit(X_top_rf, y)
                        acc_top_rf = rf_top.score(X_top_rf, y)
                    else:
                        acc_top_rf = None

                    col_rf1, col_rf2 = st.columns(2)
                    col_rf1.metric("RF (All Diff Vars) Win Rate", f"{acc_all_rf:.1%}")
                    if acc_top_rf is not None:
                        col_rf2.metric("RF (Top‑Slider Vars) Win Rate", f"{acc_top_rf:.1%}")
                    else:
                        col_rf2.write("N/A")

# -----------------------------------------------
# FEATURE IMPORTANCE (MI only)
# -----------------------------------------------
st.header("Feature Importance (Mutual Information)")
hist_imp_full = spider_hist.copy()
if len(hist_imp_full) < 10:
    st.warning("Too few historical fights to compute importance (apply broader filters).")
else:
    hist_imp_full['Target'] = (hist_imp_full['Win?'] == 'Yes').astype(int)
    feats = [c for c in numeric_features if c in hist_imp_full.columns and c not in abs_rating_cols]
    if not feats:
        st.warning("No numeric features (excluding absolute ratings).")
    else:
        missing_frac = hist_imp_full[feats].isna().mean()
        keep_features = missing_frac[missing_frac <= 0.8].index.tolist()
        if not keep_features:
            st.warning("All features are too sparse – try broadening filters.")
        else:
            X = hist_imp_full[keep_features].copy()
            y = hist_imp_full['Target']

            imp = SimpleImputer(strategy='median')
            X_imp = imp.fit_transform(X)

            if len(y) < 5:
                st.warning("Not enough rows to compute importance (apply broader filters).")
            else:
                mi = mutual_info_classif(X_imp, y, discrete_features=False, random_state=42)
                mi_df = pd.DataFrame({'Feature': keep_features, 'MI': mi}).sort_values('MI', ascending=False).head(20)
                fig_mi = px.bar(mi_df, x='MI', y='Feature', orientation='h',
                                title="Top 20 Mutual Information (filtered)")
                st.plotly_chart(fig_mi, use_container_width=True, key="mi_plot")
