# -----------------------------------------------
# SPIDER CHART (Similarity) + NEW SPIDER DECISION TREE
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
                        fight_rows = fight_rows.sort_values('Fighter')
                        f1 = fight_rows.iloc[0]
                        f2 = fight_rows.iloc[1]
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

                        # ========== NEW: SPIDER‑FILTERED DECISION TREE ==========
                        st.subheader("Decision Tree from Similarity Filters")

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

                                        st.subheader("Prediction for Selected Upcoming Fight")
                                        # Use the already‑selected fight (selected_fight_spider)
                                        fight_rows = spider_upcoming[spider_upcoming['FightID'] == selected_fight_spider]
                                        fight_rows = fight_rows.sort_values('Fighter')
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
# ========== END SPIDER DECISION TREE ==========
