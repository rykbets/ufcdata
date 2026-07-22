import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import gdown
from sklearn.linear_model import LogisticRegressionCV
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
    ('selected_fight_row', None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# -----------------------------------------------
# PERFORMANCE SUMMARY
# -----------------------------------------------
st.title("UFC Pre‑Fight Performance Dashboard")
st.header("Performance Summary")
total = len(data)
wins = (data['Win?'] == 'Yes').sum()
win_rate = wins / total * 100 if total > 0 else 0
col1, col2, col3 = st.columns(3)
col1.metric("Total Fights", total); col2.metric("Wins", wins); col3.metric("Win Rate", f"{win_rate:.1f}%")

# -----------------------------------------------
# LAST 20 FIGHTS
# -----------------------------------------------
st.header("Last 20 Fights")
last20 = data.sort_values('FightDate', ascending=False).head(20)
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
            f1 = fight_rows.iloc[0]; f2 = fight_rows.iloc[1]
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

            # Top 5 Differentials – table format side by side
            st.subheader("Top 5 Differentials")
            diffs_f1 = {}
            diffs_f2 = {}
            for c in f1.index:
                if (c.endswith('_opp_diff') or (c.startswith('adj_') and c.endswith('_diff'))):
                    if pd.notna(f1[c]): diffs_f1[c] = f1[c]
                    if pd.notna(f2[c]): diffs_f2[c] = f2[c]
            top5_f1 = sorted(diffs_f1.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_f2 = sorted(diffs_f2.items(), key=lambda x: x[1], reverse=True)[:5]

            colA, colB = st.columns(2)
            with colA:
                st.write(f"**{f1['Fighter']}**")
                if top5_f1:
                    df_f1 = pd.DataFrame(top5_f1, columns=["Stat", "Value"])
                    df_f1["Value"] = df_f1["Value"].apply(lambda x: f"{x:+.2f}")
                    st.dataframe(df_f1, hide_index=True, use_container_width=True)
                else:
                    st.write("No differentials available.")
            with colB:
                st.write(f"**{f2['Fighter']}**")
                if top5_f2:
                    df_f2 = pd.DataFrame(top5_f2, columns=["Stat", "Value"])
                    df_f2["Value"] = df_f2["Value"].apply(lambda x: f"{x:+.2f}")
                    st.dataframe(df_f2, hide_index=True, use_container_width=True)
                else:
                    st.write("No differentials available.")
        else:
            st.warning("Fight data incomplete (expected 2 rows).")
else:
    st.info("No upcoming fights available.")

# -----------------------------------------------
# INDEPENDENT FILTER HELPER (unchanged)
# -----------------------------------------------
def build_independent_filter(df, key_prefix):
    with st.expander(f"{key_prefix} Filters", expanded=True):
        # --- General ---
        with st.expander("General", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                wc = st.multiselect("Weight Class", sorted(df['WC'].dropna().unique()), key=f"{key_prefix}_wc") if 'WC' in df.columns else []
                stance = st.multiselect("Stance", sorted(df['Stance'].dropna().unique()), key=f"{key_prefix}_stance") if 'Stance' in df.columns else []
                country = st.multiselect("Country", sorted(df['Country'].dropna().unique()), key=f"{key_prefix}_country") if 'Country' in df.columns else []
                event_country = st.multiselect("Event Country", sorted(df['EventCountry'].dropna().unique()), key=f"{key_prefix}_event") if 'EventCountry' in df.columns else []
            with col2:
                sched_rounds = st.multiselect("Scheduled Rounds", sorted(df['ScheduledRounds'].dropna().unique()), key=f"{key_prefix}_sched") if 'ScheduledRounds' in df.columns else []
                title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"], key=f"{key_prefix}_title") if 'Title' in df.columns else "All"
                hometown_fighter = st.multiselect("Hometown (Fighter)", sorted(df['HometownFighter'].dropna().unique()), key=f"{key_prefix}_hometown") if 'HometownFighter' in df.columns else []
                opp_hometown = st.multiselect("Opponent Hometown", sorted(df['Opponent_Hometown'].dropna().unique()), key=f"{key_prefix}_opp_hometown") if 'Opponent_Hometown' in df.columns else []

        # --- Fight Numbers & Win% ---
        with st.expander("Fight Numbers & Career Win%", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                fn_min = st.number_input("Min Fight #", value=1, min_value=1, max_value=int(df['FightNumber'].max()), key=f"{key_prefix}_fn_min") if 'FightNumber' in df.columns else 1
                fn_max = st.number_input("Max Fight #", value=int(df['FightNumber'].max()), key=f"{key_prefix}_fn_max") if 'FightNumber' in df.columns else 1000
            with col2:
                ofn_min = st.number_input("Opp Min Fight #", value=1, key=f"{key_prefix}_ofn_min") if 'Opponent_FightNumber' in df.columns else 1
                ofn_max = st.number_input("Opp Max Fight #", value=int(df['Opponent_FightNumber'].max()), key=f"{key_prefix}_ofn_max") if 'Opponent_FightNumber' in df.columns else 1000
            cwp_min, cwp_max = st.slider("Career Win % Diff", -100, 100, (-100, 100), step=5, key=f"{key_prefix}_cwp") if 'CareerWinPct_diff' in df.columns else (-100,100)

        # --- Physical & Days ---
        with st.expander("Physical & Days", expanded=False):
            age_min, age_max = st.slider("Age", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())), key=f"{key_prefix}_age") if 'Age' in df.columns else (0,100)
            ad_min, ad_max = st.slider("Age Diff", int(df['AgeDiff'].min()), int(df['AgeDiff'].max()), (int(df['AgeDiff'].min()), int(df['AgeDiff'].max())), key=f"{key_prefix}_age_diff") if 'AgeDiff' in df.columns else (-100,100)
            hd_min, hd_max = st.slider("Height Diff", int(df['HeightDiff'].min()), int(df['HeightDiff'].max()), (int(df['HeightDiff'].min()), int(df['HeightDiff'].max())), key=f"{key_prefix}_hd") if 'HeightDiff' in df.columns else (-50,50)
            rd_min, rd_max = st.slider("Reach Diff", int(df['ReachDiff'].min()), int(df['ReachDiff'].max()), (int(df['ReachDiff'].min()), int(df['ReachDiff'].max())), key=f"{key_prefix}_rd") if 'ReachDiff' in df.columns else (-50,50)
            days_min, days_max = st.slider("Days Since Prev", int(df['DaysSincePrev'].min()), int(df['DaysSincePrev'].max()), (int(df['DaysSincePrev'].min()), int(df['DaysSincePrev'].max())), key=f"{key_prefix}_days") if 'DaysSincePrev' in df.columns else (0,1000)
            ddiff_min, ddiff_max = st.slider("Days Since Prev Diff", int(df['DaysSincePrev_diff'].min()), int(df['DaysSincePrev_diff'].max()), (int(df['DaysSincePrev_diff'].min()), int(df['DaysSincePrev_diff'].max())), key=f"{key_prefix}_ddiff") if 'DaysSincePrev_diff' in df.columns else (-1000,1000)
            avg3_min, avg3_max = st.slider("Avg3DaysGap Diff", int(df['Avg3DaysGap_diff'].min()), int(df['Avg3DaysGap_diff'].max()), (int(df['Avg3DaysGap_diff'].min()), int(df['Avg3DaysGap_diff'].max())), key=f"{key_prefix}_avg3") if 'Avg3DaysGap_diff' in df.columns else (-1000,1000)

        # --- Odds ---
        with st.expander("Odds", expanded=False):
            odds_min, odds_max = st.slider("Fighter Odds", int(df['FighterOddsNum'].min()), int(df['FighterOddsNum'].max()), (int(df['FighterOddsNum'].min()), int(df['FighterOddsNum'].max())), step=10, key=f"{key_prefix}_odds") if 'FighterOddsNum' in df.columns else (-1000,1000)
            podds_min, podds_max = st.slider("Prev Fighter Odds", int(df['PrevFighterOddsNum'].min()), int(df['PrevFighterOddsNum'].max()), (int(df['PrevFighterOddsNum'].min()), int(df['PrevFighterOddsNum'].max())), step=10, key=f"{key_prefix}_podds") if 'PrevFighterOddsNum' in df.columns else (-1000,1000)

        # --- Previous Outcomes (with Win/Loss options) ---
        with st.expander("Previous Outcomes", expanded=False):
            skip_nc = st.checkbox("Skip NC outcomes", key=f"{key_prefix}_skip_nc")
            if skip_nc:
                prev1_col = 'Prev1_Outcome_skipNC'; prev2_col = 'Prev2_Outcome_skipNC'; prev3_col = 'Prev3_Outcome_skipNC'
                career1_col = 'Career1_Outcome_skipNC'; career2_col = 'Career2_Outcome_skipNC'; career3_col = 'Career3_Outcome_skipNC'
                opp_career1_col = 'Opponent_Career1_Outcome_skipNC'; opp_career2_col = 'Opponent_Career2_Outcome_skipNC'; opp_career3_col = 'Opponent_Career3_Outcome_skipNC'
            else:
                prev1_col = 'Prev1_Outcome_raw'; prev2_col = 'Prev2_Outcome_raw'; prev3_col = 'Prev3_Outcome_raw'
                career1_col = 'Career1_Outcome_raw'; career2_col = 'Career2_Outcome_raw'; career3_col = 'Career3_Outcome_raw'
                opp_career1_col = 'Opponent_Career1_Outcome_raw'; opp_career2_col = 'Opponent_Career2_Outcome_raw'; opp_career3_col = 'Opponent_Career3_Outcome_raw'

            all_outcomes_raw = sorted(df[prev1_col].dropna().unique()) if prev1_col in df.columns else []
            all_outcomes_career = sorted(df[career1_col].dropna().unique()) if career1_col in df.columns else []

            outcome_options_raw = all_outcomes_raw + ["Win (any)", "Loss (any)"]
            outcome_options_career = all_outcomes_career + ["Win (any)", "Loss (any)"]

            prev1 = st.multiselect("Prev Fight 1", outcome_options_raw, key=f"{key_prefix}_prev1")
            prev2 = st.multiselect("Prev Fight 2", outcome_options_raw, key=f"{key_prefix}_prev2")
            prev3 = st.multiselect("Prev Fight 3", outcome_options_raw, key=f"{key_prefix}_prev3")
            opp_prev1 = st.multiselect("Opp Prev 1", outcome_options_raw, key=f"{key_prefix}_opp_prev1")
            opp_prev2 = st.multiselect("Opp Prev 2", outcome_options_raw, key=f"{key_prefix}_opp_prev2")
            opp_prev3 = st.multiselect("Opp Prev 3", outcome_options_raw, key=f"{key_prefix}_opp_prev3")
            career1 = st.multiselect("Career F1", outcome_options_career, key=f"{key_prefix}_career1")
            career2 = st.multiselect("Career F2", outcome_options_career, key=f"{key_prefix}_career2")
            career3 = st.multiselect("Career F3", outcome_options_career, key=f"{key_prefix}_career3")
            opp_career1 = st.multiselect("Opp Career F1", outcome_options_career, key=f"{key_prefix}_opp_career1")
            opp_career2 = st.multiselect("Opp Career F2", outcome_options_career, key=f"{key_prefix}_opp_career2")
            opp_career3 = st.multiselect("Opp Career F3", outcome_options_career, key=f"{key_prefix}_opp_career3")

        # --- Ratings ---
        with st.expander("Ratings", expanded=False):
            use_colley = st.checkbox("Filter ColleyDecayDiff", value=False, key=f"{key_prefix}_use_colley")
            if use_colley:
                min_cd, max_cd = get_diff_range(df, 'ColleyDecayDiff')
                colley_range = st.slider("ColleyDecayDiff range", min_cd, max_cd, (min_cd, max_cd), step=0.01, key=f"{key_prefix}_colley")
            use_massey = st.checkbox("Filter MasseyFinishDecayDiff", value=False, key=f"{key_prefix}_use_massey")
            if use_massey:
                min_md, max_md = get_diff_range(df, 'MasseyFinishDecayDiff')
                massey_range = st.slider("MasseyFinishDecayDiff range", min_md, max_md, (min_md, max_md), step=0.01, key=f"{key_prefix}_massey")
            use_wmd = st.checkbox("Filter WeightedMasseyDecayDiff", value=False, key=f"{key_prefix}_use_wmd")
            if use_wmd:
                min_wmd, max_wmd = get_diff_range(df, 'WeightedMasseyDecayDiff')
                wmd_range = st.slider("WeightedMasseyDecayDiff range", min_wmd, max_wmd, (min_wmd, max_wmd), step=0.01, key=f"{key_prefix}_wmd")

        # --- Other ---
        with st.expander("Other", expanded=False):
            prev_title = st.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"], key=f"{key_prefix}_prev_title")
            opp_prev_title = st.selectbox("Opp Prev Fight Was Title?", ["All", "Yes", "No"], key=f"{key_prefix}_opp_prev_title")
            new_wc = st.checkbox("New Weight Class", key=f"{key_prefix}_new_wc") if 'IsNewWeightClass' in df.columns else False

    # ===== BUILD MASK =====
    mask = pd.Series(True, index=df.index)

    def add_filter(condition, col_name=None):
        if condition is None:
            return None
        if col_name and col_name in df.columns:
            return condition | df[col_name].isna()
        return condition

    if wc: mask &= df['WC'].isin(wc)
    if stance: mask &= df['Stance'].isin(stance)
    if country: mask &= df['Country'].isin(country)
    if sched_rounds: mask &= df['ScheduledRounds'].isin(sched_rounds)
    if title_fight != "All": mask &= df['Title'] == title_fight
    if hometown_fighter: mask &= df['HometownFighter'].isin(hometown_fighter)
    if opp_hometown: mask &= df['Opponent_Hometown'].isin(opp_hometown)
    if event_country: mask &= df['EventCountry'].isin(event_country)
    if new_wc and 'IsNewWeightClass' in df.columns: mask &= df['IsNewWeightClass'] == True
    if prev_title != "All" and 'Prev1_Title' in df.columns:
        mask &= add_filter(df['Prev1_Title'].str.strip().str.lower() == prev_title.lower(), 'Prev1_Title')
    if opp_prev_title != "All" and 'Opponent_Prev1_Title' in df.columns:
        mask &= add_filter(df['Opponent_Prev1_Title'].str.strip().str.lower() == opp_prev_title.lower(), 'Opponent_Prev1_Title')

    # Numeric filters
    if 'FightNumber' in df.columns:
        mask &= add_filter((df['FightNumber'] >= fn_min) & (df['FightNumber'] <= fn_max), 'FightNumber')
    if 'Opponent_FightNumber' in df.columns:
        mask &= add_filter((df['Opponent_FightNumber'] >= ofn_min) & (df['Opponent_FightNumber'] <= ofn_max), 'Opponent_FightNumber')
    if 'CareerWinPct_diff' in df.columns:
        mask &= add_filter((df['CareerWinPct_diff'] >= cwp_min) & (df['CareerWinPct_diff'] <= cwp_max), 'CareerWinPct_diff')

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
            mask &= add_filter((df[col] >= cmin) & (df[col] <= cmax), col)

    # Outcome filter helper – always keep NaN rows
    def apply_outcome_filter(col, selected):
        if not selected or col not in df.columns:
            return None
        cond = pd.Series(False, index=df.index)
        if "Win (any)" in selected:
            cond |= df[col].str.startswith('Win', na=False)
        if "Loss (any)" in selected:
            cond |= df[col].str.startswith('Loss', na=False)
        exact = [s for s in selected if s not in ("Win (any)", "Loss (any)")]
        if exact:
            cond |= df[col].isin(exact)
        return cond | df[col].isna()

    # Fighter outcomes
    for col, val in [(prev1_col, prev1), (prev2_col, prev2), (prev3_col, prev3),
                     (career1_col, career1), (career2_col, career2), (career3_col, career3)]:
        if val:
            c = apply_outcome_filter(col, val)
            if c is not None:
                mask &= c

    # Opponent previous outcomes
    for shift, wlist in [(1, opp_prev1), (2, opp_prev2), (3, opp_prev3)]:
        col = f'Opponent_Prev{shift}_Outcome_raw'
        if wlist and col in df.columns:
            if skip_nc:
                col_use = f'Opponent_Prev{shift}_Outcome_skipNC'
                if col_use in df.columns:
                    c = apply_outcome_filter(col_use, wlist)
                    if c is not None: mask &= c
            else:
                c = apply_outcome_filter(col, wlist)
                if c is not None: mask &= c

    # Opponent career outcomes
    for col, val in [(opp_career1_col, opp_career1), (opp_career2_col, opp_career2), (opp_career3_col, opp_career3)]:
        if val and col in df.columns:
            c = apply_outcome_filter(col, val)
            if c is not None: mask &= c

    # Ratings
    if use_colley and 'ColleyDecayDiff' in df.columns:
        mask &= add_filter((df['ColleyDecayDiff'] >= colley_range[0]) & (df['ColleyDecayDiff'] <= colley_range[1]), 'ColleyDecayDiff')
    if use_massey and 'MasseyFinishDecayDiff' in df.columns:
        mask &= add_filter((df['MasseyFinishDecayDiff'] >= massey_range[0]) & (df['MasseyFinishDecayDiff'] <= massey_range[1]), 'MasseyFinishDecayDiff')
    if use_wmd and 'WeightedMasseyDecayDiff' in df.columns:
        mask &= add_filter((df['WeightedMasseyDecayDiff'] >= wmd_range[0]) & (df['WeightedMasseyDecayDiff'] <= wmd_range[1]), 'WeightedMasseyDecayDiff')

    return df[mask].copy()

# -----------------------------------------------
# SPIDER CHART (unchanged)
# -----------------------------------------------
st.header("Fight Similarity (Independent Filters)")
spider_data_full = original_data.copy()
spider_data = build_independent_filter(spider_data_full, "spider")

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
        sim_features = [c for c in numeric_features if c in spider_data.columns and c not in abs_rating_cols]
        if not sim_features:
            st.warning("No numeric features for similarity.")
        else:
            selected_vars = st.multiselect("Select variables for similarity", sim_features, default=sim_features[:5], max_selections=8, key="spider_vars")
            available_metrics = ["Euclidean", "Manhattan", "Chebyshev"]
            distance_metrics = st.multiselect("Distance metrics", available_metrics, default=["Euclidean"], key="spider_metrics")
            if not distance_metrics:
                st.warning("Please select at least one distance metric.")
            elif selected_vars:
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

                        total_hist_count = len(sim_df)
                        st.metric("Total historical fights matching filters", total_hist_count)

                        st.subheader("Similarity Metrics (Top N)")
                        n_top = st.slider("Number of top similar fights", 5, 100, 50, step=5, key="spider_top_n")
                        top_n = sim_df.head(n_top)
                        count = len(top_n)
                        avg_sim = top_n['Similarity'].mean()
                        total_sim = top_n['Similarity'].sum()
                        composite = avg_sim * (count ** 0.5) / 100

                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Count (Top N)", count)
                        col2.metric("Avg Similarity", f"{avg_sim:.1f}%")
                        col3.metric("Total Similarity", f"{total_sim:.1f}")
                        col4.metric("Composite Score", f"{composite:.1f}")

                        high_sim_90 = top_n[top_n['Similarity'] >= 90]
                        high_sim_80 = top_n[top_n['Similarity'] >= 80]

                        wins_90 = (high_sim_90['Win?'] == 'Yes').sum() if len(high_sim_90) > 0 else 0
                        win_rate_90 = wins_90 / len(high_sim_90) * 100 if len(high_sim_90) > 0 else 0.0
                        weight_sum_wins_90 = high_sim_90.loc[high_sim_90['Win?'] == 'Yes', 'Similarity'].sum() if wins_90 > 0 else 0.0
                        weight_sum_all_90 = high_sim_90['Similarity'].sum() if len(high_sim_90) > 0 else 1
                        weighted_wr_90 = (weight_sum_wins_90 / weight_sum_all_90) * 100 if weight_sum_all_90 > 0 else 0.0

                        wins_80 = (high_sim_80['Win?'] == 'Yes').sum() if len(high_sim_80) > 0 else 0
                        win_rate_80 = wins_80 / len(high_sim_80) * 100 if len(high_sim_80) > 0 else 0.0
                        weight_sum_wins_80 = high_sim_80.loc[high_sim_80['Win?'] == 'Yes', 'Similarity'].sum() if wins_80 > 0 else 0.0
                        weight_sum_all_80 = high_sim_80['Similarity'].sum() if len(high_sim_80) > 0 else 1
                        weighted_wr_80 = (weight_sum_wins_80 / weight_sum_all_80) * 100 if weight_sum_all_80 > 0 else 0.0

                        col5, col6, col7, col8 = st.columns(4)
                        col5.metric("Win Rate (≥90%)", f"{win_rate_90:.1f}%", delta=f"{len(high_sim_90)} fights")
                        col6.metric("Weighted Win Rate (≥90%)", f"{weighted_wr_90:.1f}%")
                        col7.metric("Win Rate (≥80%)", f"{win_rate_80:.1f}%", delta=f"{len(high_sim_80)} fights")
                        col8.metric("Weighted Win Rate (≥80%)", f"{weighted_wr_80:.1f}%")

                        fig_hist = px.histogram(sim_df, x='Similarity', nbins=20, title="Similarity Distribution (Combined)")
                        st.plotly_chart(fig_hist, use_container_width=True, key="sim_hist_chart")

                        st.subheader(f"Top {n_top} Most Similar Historical Fights")
                        col_order = ['FightDate','Fighter','Opponent','Win?'] + [f'Sim_{m}' for m in distance_metrics] + ['Similarity']
                        st.dataframe(top_n[col_order], use_container_width=True)

# -----------------------------------------------
# DECISION TREE (with fixed fight selector display)
# -----------------------------------------------
st.header("Decision Tree Model (with adjustable depth/leaf)")
tree_data = build_independent_filter(original_data.copy(), "tree")

tree_upcoming = tree_data[tree_data['Win?'].isna() | (tree_data['Win?'] == '')]
if not tree_upcoming.empty:
    tree_upcoming_ids = sorted(tree_upcoming['FightID'].unique())
    tree_selected_fight = st.selectbox("Select fight for tree prediction", tree_upcoming_ids, key="tree_fight_selector")
    if tree_selected_fight:
        tree_fight_rows = tree_upcoming[tree_upcoming['FightID'] == tree_selected_fight]
        if len(tree_fight_rows) == 2:
            st.write(f"Selected: **{tree_fight_rows.iloc[0]['Fighter']}** vs **{tree_fight_rows.iloc[1]['Fighter']}**")
        elif len(tree_fight_rows) == 1:
            st.warning("Only one fighter row found for this fight; opponent data is missing.")
        else:
            st.warning("Unexpected number of rows for this fight.")
else:
    st.info("No upcoming fights in filtered tree data.")
    tree_selected_fight = None

tree_hist = tree_data[tree_data['Win?'].isin(['Yes','No'])].copy()
if len(tree_hist) < 10:
    st.warning("Not enough historical fights for decision tree.")
else:
    tree_hist['Target'] = (tree_hist['Win?'] == 'Yes').astype(int)
    tree_features = [c for c in numeric_features if c in tree_hist.columns and c not in abs_rating_cols]
    if not tree_features:
        st.warning("No features available for decision tree.")
    else:
        X = tree_hist[tree_features].fillna(tree_hist[tree_features].median())
        y = tree_hist['Target']

        col1, col2, col3 = st.columns(3)
        with col1:
            max_depth = st.slider("Max Depth", 1, 10, 3, key="tree_depth")
        with col2:
            min_samples_leaf = st.slider("Min Samples Leaf", 1, 100, 5, key="tree_leaf")
        with col3:
            criterion = st.selectbox("Splitting Criterion", ["gini", "entropy", "log_loss"], index=0, key="tree_criterion")

        if st.button("Train Decision Tree", key="train_tree"):
            with st.spinner("Training..."):
                dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                            criterion=criterion, random_state=42)
                dt.fit(X, y)

                fig, ax = plt.subplots(figsize=(24, 12))
                plot_tree(dt, feature_names=tree_features, class_names=['Loss', 'Win'],
                          filled=True, rounded=True, fontsize=10, ax=ax)
                st.pyplot(fig)

                st.subheader("Leaf Win Percentages")
                leaf_ids = dt.apply(X)
                leaf_stats = []
                for leaf_id in np.unique(leaf_ids):
                    mask_leaf = leaf_ids == leaf_id
                    leaf_stats.append({
                        "Leaf": leaf_id,
                        "Samples": mask_leaf.sum(),
                        "Win Rate": f"{y[mask_leaf].mean() * 100:.1f}%"
                    })
                leaf_df = pd.DataFrame(leaf_stats)
                st.dataframe(leaf_df, use_container_width=True, hide_index=True)

                st.subheader("Prediction for Selected Upcoming Fight")
                if tree_selected_fight is not None:
                    fight_rows = tree_upcoming[tree_upcoming['FightID'] == tree_selected_fight]
                    if len(fight_rows) == 2:
                        f1_row = fight_rows.iloc[0]
                        input_vals = []
                        for c in tree_features:
                            val = f1_row.get(c, np.nan)
                            if pd.isna(val):
                                val = tree_hist[c].median()
                            input_vals.append(val)
                        X_input = np.array([input_vals])
                        try:
                            prob = dt.predict_proba(X_input)[0, 1]
                            leaf = dt.apply(X_input)[0]
                            st.write(f"**{f1_row['Fighter']}** → leaf **{leaf}** with win probability **{prob:.1%}**")
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
                    else:
                        st.warning("Fight data incomplete for prediction.")
                else:
                    st.info("No fight selected for the tree. Choose a fight above.")

# -----------------------------------------------
# FEATURE IMPORTANCE (unchanged)
# -----------------------------------------------
st.header("Top 20 Feature Importance (Full Data, No Absolute Ratings)")
hist_imp_full = data[data['Win?'].isin(['Yes','No'])].copy()
if len(hist_imp_full) < 10:
    st.warning("Too few historical fights to compute importance.")
else:
    hist_imp_full['Target'] = (hist_imp_full['Win?'] == 'Yes').astype(int)
    feats = [c for c in numeric_features if c in hist_imp_full.columns and c not in abs_rating_cols]
    if feats:
        X_mi = hist_imp_full[feats].dropna()
        if len(X_mi) >= 10:
            imputer = SimpleImputer(strategy='median')
            X_imp = imputer.fit_transform(X_mi)
            y_mi = hist_imp_full.loc[X_mi.index, 'Target']
            mi = mutual_info_classif(X_imp, y_mi, discrete_features=False, random_state=42)
            mi_df = pd.DataFrame({'Feature': feats, 'MI': mi}).sort_values('MI', ascending=False).head(20)
            fig_mi = px.bar(mi_df, x='MI', y='Feature', orientation='h',
                            title="Top 20 Mutual Information")
            st.plotly_chart(fig_mi, use_container_width=True, key="mi_plot")

            if st.button("Compute Lasso Importance (all features)"):
                with st.spinner("Fitting LassoCV..."):
                    X_lasso = hist_imp_full[feats].copy()
                    y_lasso = hist_imp_full['Target']
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
                    X_rf = hist_imp_full[feats].copy()
                    y_rf = hist_imp_full['Target']
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
            st.warning("Not enough complete rows for MI.")
    else:
        st.warning("No numeric features (excluding absolute ratings).")
