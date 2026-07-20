import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import itertools
import gdown
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_predict
from scipy.spatial.distance import cdist
import os

st.set_page_config(page_title="UFC Pre‑Fight Dashboard (Adjusted)", layout="wide")

# ============================================================
# Load data
# ============================================================
@st.cache_data
def load_data():
    if os.path.exists("all_fights_adjperf.parquet"):
        return pd.read_parquet("all_fights_adjperf.parquet")
    PARQUET_FILE_ID = "1uIpfbGFmDolA8P2vc15VvA1qbNzWetxf"
    gdown.download(f"https://drive.google.com/uc?id={PARQUET_FILE_ID}", "data.parquet", quiet=True)
    return pd.read_parquet("data.parquet")

data = load_data()
original_data = data.copy()

# ---------- Helper functions ----------
def safe_col(df, col):
    return col if col in df.columns else None

def get_diff_range(df, col_name):
    if col_name not in df.columns:
        return -1.0, 1.0
    vals = df[col_name].dropna()
    if len(vals) == 0:
        return -1.0, 1.0
    return float(vals.min()), float(vals.max())

def get_first_col(df, col_name):
    if col_name not in df.columns:
        return np.full(len(df), np.nan)
    sub = df[col_name]
    if isinstance(sub, pd.DataFrame):
        return sub.iloc[:, 0].to_numpy(dtype=np.float64, na_value=np.nan)
    return pd.to_numeric(sub, errors='coerce').to_numpy(dtype=np.float64)

def normalize_title_col(series):
    if series is None:
        return pd.Series('', index=series.index)
    return series.astype(str).str.strip().str.lower()

# ---------- Define features – include all base stats + adjperf diffs ----------
adjperf_diff_cols = [c for c in data.columns if c.endswith('_diff') and c.startswith('adjperf_')]
base_cols = ['Age', 'AgeDiff', 'HeightDiff', 'ReachDiff',
             'DaysSincePrev', 'DaysSincePrev_diff', 'Avg3DaysGap_diff',
             'FightNumber', 'FightNumber_diff',
             'FighterOddsNum', 'PrevFighterOddsNum',
             'CareerWinPct_diff', 'Prev7WinPct',
             'FighterColleyDecay', 'OpponentColleyDecay', 'ColleyDecayDiff',
             'FighterMasseyDecay', 'OpponentMasseyDecay', 'MasseyDecayDiff',
             'FighterWeightedMasseyDecay', 'OpponentWeightedMasseyDecay', 'WeightedMasseyDecayDiff']

new_features = []
for col in base_cols:
    if col in data.columns:
        new_features.append(col)
for col in adjperf_diff_cols:
    if col in data.columns:
        new_features.append(col)

# Remove duplicates
seen = set()
new_features_unique = []
for col in new_features:
    if col not in seen:
        seen.add(col)
        new_features_unique.append(col)
new_features = new_features_unique

# For 3D features, use numeric columns from new_features with variance
three_d_features = [c for c in new_features if data[c].nunique(dropna=True) >= 2 and np.issubdtype(data[c].dtype, np.number)]

# =========================================================================
# Sidebar Filters (same as before – omitted for brevity, but include full code)
# =========================================================================
# ... (copy the entire filter section from the previous version – it's unchanged)
# =========================================================================
# COMMON DEFINITIONS, TRAINING, PERFORMANCE SUMMARY, MATCHUP, 3D SCATTERS, ETC.
# =========================================================================
# I'll keep the structure identical to the last version, but change the combo builder candidates.

# ... (all code up to the combo builders remains the same)

# =========================================================================
# LR COMBO BUILDER – using all new_features (base stats + adjperf diffs)
# =========================================================================
st.subheader("LR 3‑Variable Combinations (Brier)")
combo_candidates = [c for c in new_features if c != 'FighterOddsNum' and c in data.columns]
if len(combo_candidates) < 3:
    st.warning("Not enough features to test (need at least 3).")
else:
    @st.cache_data
    def numerical_importance(_data, features):
        hist = _data[_data['Win?'].isin(['Yes','No'])].copy()
        hist['Target'] = (hist['Win?'] == 'Yes').astype(int)
        features = list(dict.fromkeys(features))
        X = hist[features].dropna()
        y = hist.loc[X.index, 'Target']
        if len(X) > 10:
            X_imp = SimpleImputer(strategy='median').fit_transform(X)
            mi = mutual_info_classif(X_imp, y, discrete_features=False)
            return pd.DataFrame({'Feature': features, 'Mutual Information': mi}).sort_values('Mutual Information', ascending=False).head(20)
        return pd.DataFrame()
    mi_df = numerical_importance(data, combo_candidates)
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

# =========================================================================
# KNN COMBO BUILDER – using all new_features (base stats + adjperf diffs)
# =========================================================================
st.subheader("KNN 3‑Variable Combinations (Brier, In‑Sample)")
combo_candidates_knn = [c for c in new_features if c != 'FighterOddsNum' and c in data.columns]
if len(combo_candidates_knn) < 3:
    st.warning("Not enough features to test (need at least 3).")
else:
    # Reuse importance from LR or compute again
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

# =========================================================================
# The rest of the dashboard (last fights, feature importance, spider chart) remains unchanged.
# =========================================================================

# =========================================================================
# LAST 20 FIGHTS
# =========================================================================
st.header("Last 20 Fights")
last20 = data.sort_values('FightDate', ascending=False).head(20)
display_cols = ['FightDate','Fighter','Opponent','Win?','Method']
for col in ['AgeDiff','HeightDiff','ReachDiff','CareerWinPct_diff','Prev7WinPct','ColleyDecayDiff','MasseyDecayDiff','WeightedMasseyDecayDiff']:
    if col in last20.columns:
        display_cols.append(col)
# Add a few adjperf diffs
for ks in ['adjperf_KD', 'adjperf_SS', 'adjperf_TD']:
    if f'{ks}_diff' in last20.columns:
        display_cols.append(f'{ks}_diff')
display_cols = [c for c in display_cols if c in last20.columns]
st.dataframe(last20[display_cols])

# =========================================================================
# FEATURE IMPORTANCE (use all new_features)
# =========================================================================
st.header("Top 20 Feature Importance (Current Filter Set)")
hist_imp = data[data['Win?'].isin(['Yes', 'No'])].copy()
if len(hist_imp) < 10:
    st.warning("Too few historical fights after filtering to compute importance.")
else:
    hist_imp['Target'] = (hist_imp['Win?'] == 'Yes').astype(int)
    eligible = list(dict.fromkeys([c for c in three_d_features if c in hist_imp.columns]))
    if eligible:
        X_num = hist_imp[eligible].dropna()
        if len(X_num) > 10 and X_num.shape[1] > 0:
            st.caption(f"Computing importance on **{len(X_num)}** historical fights.")
            imputer = SimpleImputer(strategy='median')
            X_imp = imputer.fit_transform(X_num)
            y_num = hist_imp.loc[X_num.index, 'Target']
            mi = mutual_info_classif(X_imp, y_num, discrete_features=False, random_state=42)
            mi_df_num = pd.DataFrame({'Feature': eligible, 'Mutual Information': mi}).sort_values('Mutual Information', ascending=False).head(20)
            fig_num = px.bar(mi_df_num, x='Mutual Information', y='Feature', orientation='h',
                             title="Top 20 Features by Mutual Information with Win/Loss")
            st.plotly_chart(fig_num, use_container_width=True)
        else:
            st.warning("Not enough complete rows for numerical importance.")
    else:
        st.warning("No numerical features available after filtering.")

# =========================================================================
# SPIDER CHART (using available features)
# =========================================================================
st.header("Fight Similarity & Comparison (Independent Filters)")
st.subheader("Spider Chart Filters (fighter data only)")

col_sp1, col_sp2 = st.columns(2)
with col_sp1:
    spider_wc = st.multiselect("Weight Class", sorted(original_data['WC'].dropna().unique()), key="spider_wc") if 'WC' in original_data.columns else []
    spider_stance = st.multiselect("Stance", sorted(original_data['Stance'].dropna().unique()), key="spider_stance") if 'Stance' in original_data.columns else []
    spider_country = st.multiselect("Country", sorted(original_data['Country'].dropna().unique()), key="spider_country") if 'Country' in original_data.columns else []
    spider_sched_rounds = st.multiselect("Scheduled Rounds", sorted(original_data['ScheduledRounds'].dropna().unique()), key="spider_sched") if 'ScheduledRounds' in original_data.columns else []
    spider_event_country = st.multiselect("Event Country", sorted(original_data['EventCountry'].dropna().unique()), key="spider_eventc") if 'EventCountry' in original_data.columns else []
with col_sp2:
    spider_title_fight = st.selectbox("Title Fight", ["All", "Yes", "No"], key="spider_title") if 'Title' in original_data.columns else "All"
    spider_hometown = st.selectbox("Hometown", ["All", "Yes", "No"], key="spider_home") if 'HometownFighter' in original_data.columns else "All"
    spider_new_wc = st.checkbox("New Weight Class", key="spider_new_wc") if 'IsNewWeightClass' in original_data.columns else False
    spider_prev_title = st.selectbox("Prev Fight Was Title?", ["All", "Yes", "No"], key="spider_prev_title")

# Start with original_data
spider_data = original_data.copy()
mask = pd.Series(True, index=spider_data.index)

if spider_wc and 'WC' in spider_data.columns: mask &= spider_data['WC'].isin(spider_wc)
if spider_stance and 'Stance' in spider_data.columns: mask &= spider_data['Stance'].isin(spider_stance)
if spider_country and 'Country' in spider_data.columns: mask &= spider_data['Country'].isin(spider_country)
if spider_sched_rounds and 'ScheduledRounds' in spider_data.columns: mask &= spider_data['ScheduledRounds'].isin(spider_sched_rounds)
if spider_title_fight != "All" and 'Title' in spider_data.columns: mask &= spider_data['Title'] == spider_title_fight
if spider_hometown != "All" and 'HometownFighter' in spider_data.columns: mask &= spider_data['HometownFighter'] == spider_hometown
if spider_event_country and 'EventCountry' in spider_data.columns: mask &= spider_data['EventCountry'].isin(spider_event_country)
if spider_new_wc and 'IsNewWeightClass' in spider_data.columns: mask &= spider_data['IsNewWeightClass'] == True

# Title filter at fight level
if 'Prev1_Title' in spider_data.columns:
    spider_data['Prev1_Title_clean'] = normalize_title_col(spider_data['Prev1_Title'])
    if spider_prev_title != "All":
        fighter_mask = spider_data['Prev1_Title_clean'] == spider_prev_title.lower()
        matching_fight_ids = spider_data.loc[fighter_mask, 'FightID'].unique()
        mask &= spider_data['FightID'].isin(matching_fight_ids)

filtered_spider = spider_data[mask]
surviving_spider_fight_ids = filtered_spider['FightID'].unique()
spider_data = original_data[original_data['FightID'].isin(surviving_spider_fight_ids)]

spider_upcoming = spider_data[spider_data['Win?'].isna() | (spider_data['Win?'] == '')]
spider_hist = spider_data[spider_data['Win?'].isin(['Yes','No'])].copy()

if spider_upcoming.empty:
    st.write("No upcoming fights after spider filters.")
else:
    fight_counts = spider_upcoming.groupby('FightID').size()
    complete_ids = fight_counts[fight_counts == 2].index
    spider_upcoming = spider_upcoming[spider_upcoming['FightID'].isin(complete_ids)]
    if spider_upcoming.empty:
        st.warning("No upcoming fight has both fighters after spider filters.")
    else:
        sim_features = [c for c in new_features if c in spider_data.columns and c not in ['FightNumber', 'FightNumber_diff']]
        sim_features = [c for c in sim_features if c not in ['Win?', 'Method', 'Round', 'Title']]
        if not sim_features:
            st.warning("No numeric features for similarity.")
        else:
            selected_vars = st.multiselect("Select variables for similarity", sim_features, default=sim_features[:5], max_selections=8, key="spider_vars")
            if selected_vars:
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
                        f1 = fight_rows.iloc[0]
                        f2 = fight_rows.iloc[1]
                        st.write(f"### {f1['Fighter']} vs {f2['Fighter']}")

                        up_vals = []
                        for var in selected_vars:
                            raw = f1[var]
                            try:
                                v = float(raw) if pd.notna(raw) else 0.0
                            except:
                                v = 0.0
                            up_vals.append(v)
                        up_vec = np.array([up_vals], dtype=np.float64)
                        up_scaled = scaler_sim.transform(up_vec)

                        hist_scaled = scaler_sim.transform(hist_sub)
                        dists = cdist(up_scaled, hist_scaled, 'euclidean').flatten()
                        max_dist = dists.max() if dists.max() > 0 else 1.0
                        sim_scores = 100 * (1 - dists / max_dist)

                        sim_df = spider_hist.loc[hist_sub.index, ['FightDate', 'Fighter', 'Opponent', 'Win?']].copy()
                        sim_df['Similarity'] = sim_scores.round(1)
                        sim_df = sim_df.sort_values('Similarity', ascending=False)

                        st.subheader("Similarity Metrics")
                        n_top = st.slider("Number of top similar fights to consider", 5, 100, 50, step=5, key="spider_top_n")
                        top_n = sim_df.head(n_top)
                        count = len(top_n)
                        avg_sim = top_n['Similarity'].mean()
                        total_sim = top_n['Similarity'].sum()
                        composite = avg_sim * (count ** 0.5) / 100
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Count", count)
                        col2.metric("Avg Similarity", f"{avg_sim:.1f}%")
                        col3.metric("Total Similarity", f"{total_sim:.1f}")
                        col4.metric("Composite Score", f"{composite:.1f}")

                        high_sim = top_n[top_n['Similarity'] >= 90]
                        if len(high_sim) > 0:
                            wins_high = (high_sim['Win?'] == 'Yes').sum()
                            win_rate_high = wins_high / len(high_sim) * 100
                            st.metric("Win Rate (Similarity ≥ 90%)", f"{win_rate_high:.1f}%", delta=f"{len(high_sim)} fights")
                        else:
                            st.write("No historical fights with similarity ≥ 90% in the top selection.")

                        st.subheader("Similarity Distribution")
                        fig_hist = px.histogram(sim_df, x='Similarity', nbins=20, title="Similarity Scores (All)")
                        st.plotly_chart(fig_hist, use_container_width=True)

                        st.subheader(f"Top {n_top} Most Similar Historical Fights")
                        st.dataframe(top_n)
