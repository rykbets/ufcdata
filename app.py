import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import gdown
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cdist

st.set_page_config(page_title="UFC Pre‑Fight Dashboard", layout="wide")

PARQUET_FILE_ID = "1UIAgg0cHBW5TMekpoohpiP23Fd6aeqg8"   # ← replace with your actual ID

@st.cache_data
def load_data():
    gdown.download(f"https://drive.google.com/uc?id={PARQUET_FILE_ID}", "data.parquet", quiet=True)
    return pd.read_parquet("data.parquet")

data = load_data()

# ---------- Sidebar Filters (unchanged) ----------
st.sidebar.title("Filters")

with st.sidebar.expander("General", expanded=True):
    wc = st.multiselect("Weight Class", sorted(data['WC'].dropna().unique()))
    stance = st.multiselect("Stance", sorted(data['Stance'].dropna().unique()))
    country = st.multiselect("Country", sorted(data['Country'].dropna().unique()))
    sched_rounds = st.multiselect("Scheduled Rounds", sorted(data['ScheduledRounds'].dropna().unique()))
    title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"])
    hometown = st.selectbox("Hometown", ["All", "Yes", "No"])
    opp_hometown = st.selectbox("Opp Hometown", ["All", "Yes", "No"])
    event_country = st.multiselect("Event Country", sorted(data['EventCountry'].dropna().unique()))

with st.sidebar.expander("Fight Numbers", expanded=False):
    fn_min = st.number_input("Min Fight #", value=1, min_value=1, max_value=int(data['FightNumber'].max()))
    fn_max = st.number_input("Max Fight #", value=int(data['FightNumber'].max()))
    ofn_min = st.number_input("Opp Min Fight #", value=1)
    ofn_max = st.number_input("Opp Max Fight #", value=int(data['Opponent_FightNumber'].max()))

with st.sidebar.expander("Career Win %", expanded=False):
    career_win_pct = st.slider("Career Win %", 0, 100, (0, 100))

with st.sidebar.expander("Physical Attributes", expanded=False):
    age = st.slider("Age", int(data['Age'].min()), int(data['Age'].max()), (int(data['Age'].min()), int(data['Age'].max())))
    height = st.slider("Height (in)", int(data['Height'].min()), int(data['Height'].max()), (int(data['Height'].min()), int(data['Height'].max())))
    reach = st.slider("Reach (in)", int(data['Reach'].min()), int(data['Reach'].max()), (int(data['Reach'].min()), int(data['Reach'].max())))

with st.sidebar.expander("Opponent Physical Attributes", expanded=False):
    if 'Age_opp' in data.columns:
        age_opp_min = int(data['Age_opp'].min()) if not data['Age_opp'].isna().all() else 0
        age_opp_max = int(data['Age_opp'].max()) if not data['Age_opp'].isna().all() else 0
        age_opp = st.slider("Opponent Age", age_opp_min, age_opp_max, (age_opp_min, age_opp_max))
    else:
        age_opp = (0, 0)
    if 'Height_opp' in data.columns:
        h_opp_min = int(data['Height_opp'].min()) if not data['Height_opp'].isna().all() else 0
        h_opp_max = int(data['Height_opp'].max()) if not data['Height_opp'].isna().all() else 0
        height_opp = st.slider("Opponent Height (in)", h_opp_min, h_opp_max, (h_opp_min, h_opp_max))
    else:
        height_opp = (0, 0)
    if 'Reach_opp' in data.columns:
        r_opp_min = int(data['Reach_opp'].min()) if not data['Reach_opp'].isna().all() else 0
        r_opp_max = int(data['Reach_opp'].max()) if not data['Reach_opp'].isna().all() else 0
        reach_opp = st.slider("Opponent Reach (in)", r_opp_min, r_opp_max, (r_opp_min, r_opp_max))
    else:
        reach_opp = (0, 0)

with st.sidebar.expander("Differences", expanded=False):
    age_diff = st.slider("Age Diff", int(data['AgeDiff'].min()), int(data['AgeDiff'].max()), (int(data['AgeDiff'].min()), int(data['AgeDiff'].max())))
    height_diff = st.slider("Height Diff (in)", int(data['HeightDiff'].min()), int(data['HeightDiff'].max()), (int(data['HeightDiff'].min()), int(data['HeightDiff'].max())))
    reach_diff = st.slider("Reach Diff (in)", int(data['ReachDiff'].min()), int(data['ReachDiff'].max()), (int(data['ReachDiff'].min()), int(data['ReachDiff'].max())))

with st.sidebar.expander("Days", expanded=False):
    days = st.slider("Days Since Prev", int(data['DaysSincePrev'].min()), int(data['DaysSincePrev'].max()), (int(data['DaysSincePrev'].min()), int(data['DaysSincePrev'].max())))
    avg3 = st.slider("Avg 3‑Fight Gap", int(data['Avg3DaysGap'].min()), int(data['Avg3DaysGap'].max()), (int(data['Avg3DaysGap'].min()), int(data['Avg3DaysGap'].max())))

with st.sidebar.expander("Odds", expanded=False):
    cur_min = int(data['FighterOddsNum'].min()) if not data['FighterOddsNum'].isna().all() else 0
    cur_max = int(data['FighterOddsNum'].max()) if not data['FighterOddsNum'].isna().all() else 0
    if cur_min != cur_max:
        cur_odds = st.slider("Fighter Odds", cur_min, cur_max, (cur_min, cur_max), step=10)
    else:
        cur_odds = (0, 0)
    prev_min = int(data['PrevFighterOddsNum'].min()) if not data['PrevFighterOddsNum'].isna().all() else 0
    prev_max = int(data['PrevFighterOddsNum'].max()) if not data['PrevFighterOddsNum'].isna().all() else 0
    if prev_min != prev_max:
        prev_odds = st.slider("Prev Fight Odds", prev_min, prev_max, (prev_min, prev_max), step=10)
    else:
        prev_odds = (0, 0)

new_wc = st.sidebar.checkbox("New Weight Class")
skip_nc = st.sidebar.checkbox("Skip NC outcomes")
prev_title = st.sidebar.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"])
opp_prev_title = st.sidebar.selectbox("Opp Prev Fight Was Title?", ["All", "Yes", "No"])

if skip_nc:
    prev1_col = 'Prev1_Outcome_skipNC'; prev2_col = 'Prev2_Outcome_skipNC'; prev3_col = 'Prev3_Outcome_skipNC'
    career1_col = 'Career1_Outcome_skipNC'; career2_col = 'Career2_Outcome_skipNC'; career3_col = 'Career3_Outcome_skipNC'
    opp_career1_col = 'Opponent_Career1_Outcome_skipNC'; opp_career2_col = 'Opponent_Career2_Outcome_skipNC'; opp_career3_col = 'Opponent_Career3_Outcome_skipNC'
else:
    prev1_col = 'Prev1_Outcome_raw'; prev2_col = 'Prev2_Outcome_raw'; prev3_col = 'Prev3_Outcome_raw'
    career1_col = 'Career1_Outcome_raw'; career2_col = 'Career2_Outcome_raw'; career3_col = 'Career3_Outcome_raw'
    opp_career1_col = 'Opponent_Career1_Outcome_raw'; opp_career2_col = 'Opponent_Career2_Outcome_raw'; opp_career3_col = 'Opponent_Career3_Outcome_raw'

all_outcomes_raw = sorted(data[prev1_col].dropna().unique())
all_outcomes_career = sorted(data[career1_col].dropna().unique())

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

with st.sidebar.expander("Rating Gap Analysis", expanded=False):
    rating_system = st.selectbox("Rating system", ['ColleyOrig','ColleyDecay','MasseyOrig','MasseyDecay'], key="gap_system")
    gap_range = st.slider("Rating gap range", min_value=0.0, max_value=1.0, value=(0.0, 0.05), step=0.01, key="gap_range")

# ---------- Apply filters ----------
filtered = data.copy()

if wc: filtered = filtered[filtered['WC'].isin(wc)]
if stance: filtered = filtered[filtered['Stance'].isin(stance)]
if country: filtered = filtered[filtered['Country'].isin(country)]
if sched_rounds: filtered = filtered[filtered['ScheduledRounds'].isin(sched_rounds)]
if title_fight != "All": filtered = filtered[filtered['Title'] == title_fight]
if hometown != "All": filtered = filtered[filtered['HometownFighter'] == hometown]
if opp_hometown != "All": filtered = filtered[filtered['Opponent_Hometown'] == opp_hometown]
if event_country: filtered = filtered[filtered['EventCountry'].isin(event_country)]
if new_wc: filtered = filtered[filtered['IsNewWeightClass'] == True]
if prev_title != "All": filtered = filtered[filtered['Prev1_Title'] == prev_title]
if opp_prev_title != "All": filtered = filtered[filtered['Opponent_Prev1_Title'] == opp_prev_title]
if prev1: filtered = filtered[filtered[prev1_col].isin(prev1)]
if prev2: filtered = filtered[filtered[prev2_col].isin(prev2)]
if prev3: filtered = filtered[filtered[prev3_col].isin(prev3)]
if career1: filtered = filtered[filtered[career1_col].isin(career1)]
if career2: filtered = filtered[filtered[career2_col].isin(career2)]
if career3: filtered = filtered[filtered[career3_col].isin(career3)]
if opp_career1: filtered = filtered[filtered[opp_career1_col].isin(opp_career1)]
if opp_career2: filtered = filtered[filtered[opp_career2_col].isin(opp_career2)]
if opp_career3: filtered = filtered[filtered[opp_career3_col].isin(opp_career3)]

for opp_shift, opp_widget in [(1, opp_prev1), (2, opp_prev2), (3, opp_prev3)]:
    raw_col = f'Opponent_Prev{opp_shift}_Outcome_raw'
    if raw_col in filtered.columns:
        use_col = f'Opponent_Prev{opp_shift}_Outcome_skipNC' if skip_nc else raw_col
        if use_col in filtered.columns and opp_widget:
            filtered = filtered[filtered[use_col].isin(opp_widget)]

filtered = filtered[(filtered['FightNumber'] >= fn_min) & (filtered['FightNumber'] <= fn_max)]
filtered = filtered[(filtered['Opponent_FightNumber'] >= ofn_min) & (filtered['Opponent_FightNumber'] <= ofn_max)]
filtered = filtered[(filtered['Age'] >= age[0]) & (filtered['Age'] <= age[1])]
filtered = filtered[(filtered['Height'] >= height[0]) & (filtered['Height'] <= height[1])]
filtered = filtered[(filtered['Reach'] >= reach[0]) & (filtered['Reach'] <= reach[1])]

if 'Age_opp' in filtered.columns:
    filtered = filtered[(filtered['Age_opp'] >= age_opp[0]) & (filtered['Age_opp'] <= age_opp[1])]
if 'Height_opp' in filtered.columns:
    filtered = filtered[(filtered['Height_opp'] >= height_opp[0]) & (filtered['Height_opp'] <= height_opp[1])]
if 'Reach_opp' in filtered.columns:
    filtered = filtered[(filtered['Reach_opp'] >= reach_opp[0]) & (filtered['Reach_opp'] <= reach_opp[1])]

filtered = filtered[(filtered['AgeDiff'] >= age_diff[0]) & (filtered['AgeDiff'] <= age_diff[1])]
filtered = filtered[(filtered['HeightDiff'] >= height_diff[0]) & (filtered['HeightDiff'] <= height_diff[1])]
filtered = filtered[(filtered['ReachDiff'] >= reach_diff[0]) & (filtered['ReachDiff'] <= reach_diff[1])]
filtered = filtered[(filtered['DaysSincePrev'] >= days[0]) & (filtered['DaysSincePrev'] <= days[1])]
filtered = filtered[(filtered['Avg3DaysGap'] >= avg3[0]) & (filtered['Avg3DaysGap'] <= avg3[1])]
filtered = filtered[(filtered['CareerWinPct'] >= career_win_pct[0]) & (filtered['CareerWinPct'] <= career_win_pct[1])]

if not data['FighterOddsNum'].isna().all() and cur_odds != (0,0):
    filtered = filtered.dropna(subset=['FighterOddsNum'])
    filtered = filtered[(filtered['FighterOddsNum'] >= cur_odds[0]) & (filtered['FighterOddsNum'] <= cur_odds[1])]
if not data['PrevFighterOddsNum'].isna().all() and prev_odds != (0,0):
    filtered = filtered.dropna(subset=['PrevFighterOddsNum'])
    filtered = filtered[(filtered['PrevFighterOddsNum'] >= prev_odds[0]) & (filtered['PrevFighterOddsNum'] <= prev_odds[1])]

data = filtered   # keep the name consistent

# ---------- Dashboard (unchanged, use existing code from previous working version) ----------
# ... paste here your Performance Summary, Rating Gap Analysis, Matchup, Last20,
#     LR/KNN, Feature Importance, Spider Chart, etc.
# They all work because 'data' has exactly the same columns as before.

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

# ---------- Rating Gap Analysis ----------
st.header("Rating Gap Analysis")
diff_col = f'{rating_system}_Diff'

if diff_col in data.columns:
    gap_min, gap_max = gap_range
    gap_fights = data[(data[diff_col] >= gap_min) & (data[diff_col] <= gap_max)]
    total_gap = len(gap_fights)
    wins_gap = (gap_fights['Win?'] == 'Yes').sum()
    win_rate_gap = wins_gap / total_gap * 100 if total_gap > 0 else 0.0

    colg1, colg2, colg3 = st.columns(3)
    with colg1:
        st.metric("Fights in gap", total_gap)
    with colg2:
        st.metric("Wins", wins_gap)
    with colg3:
        st.metric("Win Rate", f"{win_rate_gap:.1f}%")
else:
    st.warning(f"Rating system '{rating_system}' not available.")

# ---------- Matchup area (unfiltered upcoming) ----------
st.header("Upcoming Fight Matchup")
upcoming_data_unfiltered = data[data['Win?'].isna() | (data['Win?'] == '')]
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
                st.write(f"**Ratings:** CO {row['FighterColleyOrig']:.4f} / CD {row['FighterColleyDecay']:.4f} / MO {row['FighterMasseyOrig']:.4f} / MD {row['FighterMasseyDecay']:.4f}")
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
                for diff_col2, unit in [('AgeDiff','yrs'),('HeightDiff','in'),('ReachDiff','in')]:
                    if diff_col2 in row:
                        diff_items.append(f"{diff_col2}: {row[diff_col2]:+.1f} {unit}" if pd.notna(row[diff_col2]) else f"{diff_col2}: --")
                st.write(" · ".join(diff_items) if diff_items else "N/A")

                st.write("**Previous Outcomes (Fighter):**")
                prev_outs = []
                for shift, col2 in [(1, prev1_col), (2, prev2_col), (3, prev3_col)]:
                    val = row[col2] if pd.notna(row[col2]) else '--'
                    prev_outs.append(f"Prev {shift}: {val}")
                st.write(" · ".join(prev_outs))

                st.write("**Career Milestones (Fighter):**")
                career_outs = []
                for shift, col2 in [(1, career1_col), (2, career2_col), (3, career3_col)]:
                    val = row[col2] if pd.notna(row[col2]) else '--'
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
                    col2 = f'Opponent_Career{shift}_Outcome_skipNC' if skip_nc else f'Opponent_Career{shift}_Outcome_raw'
                    val = row[col2] if col2 in row and pd.notna(row[col2]) else '--'
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
                'FighterOddsBFO','OpponentOddsBFO','Prev7Wins','Prev7Losses',
                'FighterColleyOrig','FighterColleyDecay','FighterMasseyOrig','FighterMasseyDecay']
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
        'FighterColleyOrig', 'OpponentColleyOrig', 'ColleyOrig_Diff',
        'FighterColleyDecay', 'OpponentColleyDecay', 'ColleyDecay_Diff',
        'FighterMasseyOrig', 'OpponentMasseyOrig', 'MasseyOrig_Diff',
        'FighterMasseyDecay', 'OpponentMasseyDecay', 'MasseyDecay_Diff']
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
            for col2 in (x_lr, y_lr, z_lr):
                if col2 in hist_base.columns:
                    train_means[col2] = hist_base[col2].mean()
                else:
                    train_means[col2] = 0

            st.subheader("LR Win Probability Estimate")
            all_upcoming = data[data['Win?'].isna() | (data['Win?'] == '')]
            if not all_upcoming.empty:
                up_ids = all_upcoming['FightID'].unique()
                chosen_id = st.selectbox("Select upcoming fight", sorted(up_ids), key="lr_up")
                if chosen_id:
                    up_rows = all_upcoming[all_upcoming['FightID'] == chosen_id]
                    if len(up_rows) == 2:
                        fighter_row = up_rows.iloc[0]

                        def safe_val(col2):
                            try:
                                val = fighter_row[col2]
                                return val if pd.notna(val) else train_means[col2]
                            except (KeyError, ValueError):
                                return train_means[col2]

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
            all_upcoming = data[data['Win?'].isna() | (data['Win?'] == '')]
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
    spider_wc = st.multiselect("Weight Class", sorted(data['WC'].dropna().unique()), key="spider_wc")
    spider_stance = st.multiselect("Stance", sorted(data['Stance'].dropna().unique()), key="spider_stance")
    spider_country = st.multiselect("Country", sorted(data['Country'].dropna().unique()), key="spider_country")
    spider_sched_rounds = st.multiselect("Scheduled Rounds", sorted(data['ScheduledRounds'].dropna().unique()), key="spider_sched")
    spider_event_country = st.multiselect("Event Country", sorted(data['EventCountry'].dropna().unique()), key="spider_eventc")
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

all_outcomes_raw_spider = sorted(data[spider_prev1_col].dropna().unique())
all_outcomes_career_spider = sorted(data[spider_career1_col].dropna().unique())

with st.expander("Previous Outcomes (Spider)"):
    spider_prev1 = st.multiselect("Prev Fight 1", all_outcomes_raw_spider, key="spider_prev1")
    spider_prev2 = st.multiselect("Prev Fight 2", all_outcomes_raw_spider, key="spider_prev2")
    spider_prev3 = st.multiselect("Prev Fight 3", all_outcomes_raw_spider, key="spider_prev3")
    spider_career1 = st.multiselect("Career F1", all_outcomes_career_spider, key="spider_career1")
    spider_career2 = st.multiselect("Career F2", all_outcomes_career_spider, key="spider_career2")
    spider_career3 = st.multiselect("Career F3", all_outcomes_career_spider, key="spider_career3")

spider_data = data.copy()
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
            'FighterColley', 'OpponentColley', 'FighterMassey', 'OpponentMassey',
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
