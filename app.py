import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import gdown
from sklearn.linear_model import LogisticRegression
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
    # Build identical masks for both fighters using row's OWN columns.
    # (One mask for Fighter A, one mask for Fighter B.)
    # ==================================================================
    def build_side_mask(data_side, side_params):
        """
        Return a boolean Series that checks each row's own statistics
        (FightNumber, HometownFighter, Country, etc.) against the
        side-specific filters.  Does NOT use opponent‑side columns.
        """
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

    # Fighter A (originally "Fighter‑Side")
    fighter_params = {
        'fn_min': params['fn_min'], 'fn_max': params['fn_max'],
        'prev_title': params['prev_title'],
        'hometown': params['hometown_fighter'],
        'country': params['fighter_country'],
        'skip_nc': params['skip_nc'],
        'prev1': params['prev1'], 'prev2': params['prev2'], 'prev3': params['prev3'],
        'career1': params['career1'], 'career2': params['career2'], 'career3': params['career3']
    }
    # Fighter B (originally "Opponent‑Side")
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

    # ----- Permutation check: exactly one row must satisfy Fighter A,
    #       the other row must satisfy Fighter B. -----
    def check_permutation(group):
        idx = group.index.tolist()
        if len(idx) != 2:
            return pd.Series(False, index=group.index)
        i1, i2 = idx[0], idx[1]
        # Permutation 1: i1 is Fighter A, i2 is Fighter B
        perm1 = fighter_mask.loc[i1] and opponent_mask.loc[i2]
        # Permutation 2: i1 is Fighter B, i2 is Fighter A
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
    # ---- General (shared) ----
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

    # ---- Fighter A & Fighter B filters side by side ----
    colA, colB = st.columns(2)
    with colA:
        st.write("**Fighter A Filters**")
        fighter_country = st.multiselect("Country (A)", sorted(original_data['Country'].dropna().unique()), key="spider_fighter_country")
        hometown_fighter = st.multiselect("Hometown (A)", sorted(original_data['HometownFighter'].dropna().unique()), key="spider_hometown")
        fn_min = st.number_input("Min Fight # (A)", value=1, min_value=1, max_value=int(original_data['FightNumber'].max()), key="spider_fn_min")
        fn_max = st.number_input("Max Fight # (A)", value=int(original_data['FightNumber'].max()), key="spider_fn_max")
        prev_title = st.selectbox("Prev Fight Was Title? (A)", ["All", "Yes", "No"], key="spider_prev_title")
    with colB:
        st.write("**Fighter B Filters**")
        opponent_country = st.multiselect("Country (B)", sorted(original_data['Country'].dropna().unique()), key="spider_opponent_country")
        opp_hometown = st.multiselect("Hometown (B)", sorted(original_data['HometownFighter'].dropna().unique()), key="spider_opp_hometown")
        ofn_min = st.number_input("Min Fight # (B)", value=1, key="spider_ofn_min")
        ofn_max = st.number_input("Max Fight # (B)", value=int(original_data['FightNumber'].max()), key="spider_ofn_max")
        opp_prev_title = st.selectbox("Prev Fight Was Title? (B)", ["All", "Yes", "No"], key="spider_opp_prev_title")

    # ---- Physical & Days (shared) ----
    with st.expander("Physical & Days", expanded=False):
        age_min, age_max = st.slider("Age", int(original_data['Age'].min()), int(original_data['Age'].max()), (int(original_data['Age'].min()), int(original_data['Age'].max())), key="spider_age")
        ad_min, ad_max = st.slider("Age Diff", int(original_data['AgeDiff'].min()), int(original_data['AgeDiff'].max()), (int(original_data['AgeDiff'].min()), int(original_data['AgeDiff'].max())), key="spider_age_diff")
        hd_min, hd_max = st.slider("Height Diff", int(original_data['HeightDiff'].min()), int(original_data['HeightDiff'].max()), (int(original_data['HeightDiff'].min()), int(original_data['HeightDiff'].max())), key="spider_hd")
        rd_min, rd_max = st.slider("Reach Diff", int(original_data['ReachDiff'].min()), int(original_data['ReachDiff'].max()), (int(original_data['ReachDiff'].min()), int(original_data['ReachDiff'].max())), key="spider_rd")
        days_min, days_max = st.slider("Days Since Prev", int(original_data['DaysSincePrev'].min()), int(original_data['DaysSincePrev'].max()), (int(original_data['DaysSincePrev'].min()), int(original_data['DaysSincePrev'].max())), key="spider_days")
        ddiff_min, ddiff_max = st.slider("Days Since Prev Diff", int(original_data['DaysSincePrev_diff'].min()), int(original_data['DaysSincePrev_diff'].max()), (int(original_data['DaysSincePrev_diff'].min()), int(original_data['DaysSincePrev_diff'].max())), key="spider_ddiff")
        avg3_min, avg3_max = st.slider("Avg3DaysGap Diff", int(original_data['Avg3DaysGap_diff'].min()), int(original_data['Avg3DaysGap_diff'].max()), (int(original_data['Avg3DaysGap_diff'].min()), int(original_data['Avg3DaysGap_diff'].max())), key="spider_avg3")
        cwp_min, cwp_max = st.slider("Career Win % Diff", -100, 100, (-100, 100), step=5, key="spider_cwp")

    # ---- Odds ----
    with st.expander("Odds", expanded=False):
        odds_min, odds_max = st.slider("Fighter Odds", int(original_data['FighterOddsNum'].min()), int(original_data['FighterOddsNum'].max()), (int(original_data['FighterOddsNum'].min()), int(original_data['FighterOddsNum'].max())), step=10, key="spider_odds")
        podds_min, podds_max = st.slider("Prev Fighter Odds", int(original_data['PrevFighterOddsNum'].min()), int(original_data['PrevFighterOddsNum'].max()), (int(original_data['PrevFighterOddsNum'].min()), int(original_data['PrevFighterOddsNum'].max())), step=10, key="spider_podds")

    # ---- Previous Outcomes (side by side) ----
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
            st.write("**Fighter A Outcomes**")
            prev1 = st.multiselect("Prev Fight 1 (A)", outcome_options_raw, key="spider_prev1")
            prev2 = st.multiselect("Prev Fight 2 (A)", outcome_options_raw, key="spider_prev2")
            prev3 = st.multiselect("Prev Fight 3 (A)", outcome_options_raw, key="spider_prev3")
            career1 = st.multiselect("Career F1 (A)", outcome_options_career, key="spider_career1")
            career2 = st.multiselect("Career F2 (A)", outcome_options_career, key="spider_career2")
            career3 = st.multiselect("Career F3 (A)", outcome_options_career, key="spider_career3")
        with col_o:
            st.write("**Fighter B Outcomes**")
            opp_prev1 = st.multiselect("Prev Fight 1 (B)", outcome_options_raw, key="spider_opp_prev1")
            opp_prev2 = st.multiselect("Prev Fight 2 (B)", outcome_options_raw, key="spider_opp_prev2")
            opp_prev3 = st.multiselect("Prev Fight 3 (B)", outcome_options_raw, key="spider_opp_prev3")
            opp_career1 = st.multiselect("Career F1 (B)", outcome_options_career, key="spider_opp_career1")
            opp_career2 = st.multiselect("Career F2 (B)", outcome_options_career, key="spider_opp_career2")
            opp_career3 = st.multiselect("Career F3 (B)", outcome_options_career, key="spider_opp_career3")

    # ---- Ratings ----
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

# ... (the rest of the spider similarity, decision tree, model comparison, and MI sections remain exactly the same as in your last version)
#      I'm omitting them for brevity – just keep the code you already have after this point.
