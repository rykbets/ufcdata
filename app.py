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
        # 3D scatter
        plot_data = data[[x_lr, y_lr, z_lr, 'DetailedResult', 'Fight']].dropna()
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

        # ----- Fit LR model & compute overall/recent win rates -----
        hist_lr = data[data['Win?'].isin(['Yes','No'])].copy()
        hist_lr = hist_lr[[x_lr, y_lr, z_lr, 'Win?']].dropna()
        if len(hist_lr) < 10 or hist_lr['Win?'].nunique() < 2:
            st.warning("Not enough historical data for LR model.")
        else:
            hist_lr['target'] = (hist_lr['Win?'] == 'Yes').astype(int)
            X_lr = hist_lr[[x_lr, y_lr, z_lr]].values
            y_lr = hist_lr['target'].values

            lr_model = LogisticRegression(max_iter=1000)
            lr_model.fit(X_lr, y_lr)
            y_prob_lr_in = lr_model.predict_proba(X_lr)[:, 1]
            ll_lr = log_loss(y_lr, y_prob_lr_in)
            bs_lr = brier_score_loss(y_lr, y_prob_lr_in)

            # Overall win rate (all filtered historical fights)
            full_hist = data[data['Win?'].isin(['Yes','No'])].sort_values('FightDate')
            if len(full_hist) > 0:
                overall_wr = (full_hist['Win?'] == 'Yes').mean() * 100
                recent = full_hist.tail(recent_window)
                recent_wr = (recent['Win?'] == 'Yes').mean() * 100 if len(recent) > 0 else 0.0
                recent_count = len(recent)
            else:
                overall_wr = recent_wr = 0.0
                recent_count = 0

            # ---- CORE METRICS (always visible) ----
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric("LR Log‑loss", f"{ll_lr:.3f}")
            with col_m2:
                st.metric("LR Brier", f"{bs_lr:.3f}")
            with col_m3:
                st.metric("Overall Win%", f"{overall_wr:.1f}%")
                st.metric(f"Recent Win% (last {recent_window})", f"{recent_wr:.1f}%")

            # ----- Upcoming fight prediction (first valid fight pre‑selected) -----
            st.subheader("LR Win Probability Estimate")
            all_upcoming = all_fights_display[all_fights_display['Win?'].isna() | (all_fights_display['Win?'] == '')]

            # Build list of upcoming fights where the first fighter has all three features
            valid_ids = []
            for fid in all_upcoming['FightID'].unique():
                rows = all_upcoming[all_upcoming['FightID'] == fid]
                if len(rows) != 2:
                    continue
                f1 = rows.iloc[0]
                ok = True
                for col in (x_lr, y_lr, z_lr):
                    try:
                        if pd.isna(f1[col]):
                            ok = False
                            break
                    except KeyError:
                        ok = False
                        break
                if ok:
                    valid_ids.append(fid)

            if valid_ids:
                # Default selection = first valid fight (sorted)
                chosen_id = st.selectbox(
                    "Select upcoming fight",
                    sorted(valid_ids),
                    index=0,
                    key="lr_up"
                )
                up_rows = all_upcoming[all_upcoming['FightID'] == chosen_id]
                fighter_row = up_rows.iloc[0]
                up_val = np.array([fighter_row[[x_lr, y_lr, z_lr]].values])
                prob_lr = lr_model.predict_proba(up_val)[0, 1]

                # --- Empirical Bayes shrinkage of recent win rate ---
                if recent_count > 0:
                    # Shrink recent win rate toward overall win rate
                    shrunk_recent = (prior_weight * overall_wr + recent_count * recent_wr) / (prior_weight + recent_count)
                else:
                    shrunk_recent = overall_wr   # no recent data

                # Shrink model probability toward the shrunken recent rate
                shrunk_prob = (prior_weight * (shrunk_recent / 100) + prob_lr) / (prior_weight + 1)

                col_p1, col_p2 = st.columns(2)
                with col_p1:
                    st.metric("LR win prob", f"{prob_lr:.1%}")
                with col_p2:
                    st.metric("LR shrunken", f"{shrunk_prob:.1%}")
            else:
                st.write("No upcoming fights have all selected predictor values.")

    # --- LR 3‑Variable Combination Builder (Brier) ---
    st.subheader("LR 3‑Variable Combinations (Brier)")
    combo_candidates = [c for c in numerical_features if c != 'FighterOddsNum' and c in data.columns and data[c].nunique(dropna=True) >= 2]

    # Ensure mi_df exists (compute if missing)
    if 'mi_df' not in dir():
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
                    df = pd.DataFrame(results).sort_values('Brier').head(20)
                    st.write("**Top 20 3‑Variable Combinations (Brier)**")
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Could not evaluate any combination.")
    else:
        st.warning("Not enough features to test (need at least 3).")
else:
    st.warning("Not enough numerical features for a 3D plot (need at least 3).")
