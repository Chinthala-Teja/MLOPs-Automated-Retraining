# ReTrainIQ contract (do not rename anything without telling the team)
Libraries: pandas, numpy, scipy, scikit-learn, joblib, plotly, streamlit, pytest, requests. Environment: Python 3.10 or newer (tested on 3.10). Style: type hints, docstrings, no global state, seeds as arguments, no network calls except in ri_analysis/llm.py. Write complete working code, no placeholders (except the stubs).
FOLDERS (one importable package each):
0-Init/ri_init: store.py worker.py config.py setup_env.py make_demo_project.py (Teja)
1-Profile/ri_profile: understand.py (Module 1, Teja)
2-Drift/ri_drift: drift.py plots.py scenarios.py (Ramakrishna)
3-CARA/ri_cara: cara_t.py replay_baselines.py proxy_eval.py (Udaykiran)
4-Metrics/ri_metrics: metrics.py (Udaykiran)
5-Analysis/ri_analysis: llm.py summary.py chat.py (Teja)
6-Decision/ri_decision: decision.py retrain.py orchestrator.py cli.py calibrate_score.py sensitivity.py (Teja)
frontend: app.py (Teja)
workspace/<project_id>/{repo,data,models,reports}: generated, git-ignored.
DATA: a batch is a pandas DataFrame with the feature columns from the profile and, if known, the target column. Unlabelled = target missing or all NaN.
PROFILE (dict, saved as workspace/<id>/profile.json): {'project_id':str,'problem_statement':str,'task':{'type':'classification'|'regression','subtype':'binary'|'multiclass'|None,'target':str,'positive_class':1},'features':{'numeric':[str],'categorical':[str]},'model':{'framework':'sklearn','artifact':path,'algorithm':str},'data':{'reference_path':path,'holdout_path':path|None,'id_column':str|None,'ordered':bool},'primary_metric':str,'batch_size_B':int,'offline_batches_N':25,'W':1,'seed':42}
RESULT DICTS (JSON-serialisable; each has 'module' and 'status'):
drift -> {'module':'drift','status':'STABLE'|'DRIFT'|'STUB','drift_share':float,'per_feature':[{'feature':str,'type':'numeric'|'categorical','psi':float,'ks_stat':float|None,'p_value':float|None,'flag':bool}],'n_ref':int,'n_new':int}
cara_t -> {'module':'cara_t','status':'OK'|'PENDING'|'ESTIMATED'|'STUB','staleness':float,'tau_star':float,'kappa':float,'s':float,'decision':'KEEP'|'RETRAIN','n_queries':int,'n_points':int,'basis':'measured'|'estimated'|'none','staleness_est':float|None,'staleness_est_lo':float|None,'decision_est':'KEEP'|'RETRAIN'|None}
quality -> {'module':'quality','status':'OK'|'PENDING'|'STUB','n':int,'metrics':{name:{'value':float,'se':float}},'curves':{},'z_m':float,'degraded':bool,'table':[{'metric':str,'reference':float,'current':float,'delta':float}]}
decision -> {'P':float,'zone':'RETRAIN'|'HUMAN'|'KEEP','zc':float,'zm':float,'d':float,'reason':str,'basis':'measured'|'estimated'|'none','terms':dict,'flags':list}
SIGNATURES (exact):
ri_init.store: init_db(db_path=None); add_project(project_id, source, profile_json, status='active'); get_project(project_id); add_batch(project_id, file, n_rows, n_labelled); add_report(project_id, batch_id, drift_json, cara_json, quality_json, score_json, zone, summary); update_report(report_id, **fields); get_report(report_id); get_reports(project_id); add_decision(report_id, action, by, note); add_version(project_id, version, path, trained_on, status='candidate'); set_active_version(project_id, version); get_active_version(project_id)
ri_init.worker: run_in_worker(callable_path:str, kwargs:dict, timeout:int=900)->dict
ri_profile.understand: fetch_repo(url, dest); extract_zip_safe(zip_path, dest, max_mb=300, max_files=5000); scan_project(path); build_context(scan, max_chars=20000); extract_profile(context, llm); heuristic_profile(scan=None, dataset_path=None, target=None); verify_profile(profile); finalize_profile(profile, user_edits=None)
ri_drift.drift: run_drift(reference_df, new_df, profile)->dict; validate_schema(df, profile)->dict; fit_reference(reference_df, profile)->dict; prediction_shift(ref_pred, new_pred, ref_proba=None, new_proba=None)->dict
ri_drift.plots: drift_bar(result); distribution_overlay(reference_df, new_df, feature, ftype); top_drifted(result, k=3)
ri_drift.scenarios: inject_covariate_shift(df, profile, feature, magnitude, start_row); inject_concept_shift(df, profile, region_rule, flip_fraction, start_row); make_invalid_upload(df, kind); hold_back_labels(df, fraction); build_scenario_stream(batches, scenario_name)
ri_cara.cara_t: calibrate(model_path, history_df, profile, cfg)->dict; online_staleness(model_path, new_df, train_sample_df, calib, profile, seed=42)->dict; estimate_staleness(model_path, new_df, train_sample_df, calib, profile, seed=42)->dict
ri_cara.replay_baselines: run_baselines(batches, calib, profile, cfg)->DataFrame
ri_cara.proxy_eval: run_proxy_eval(profile, cfg)->DataFrame
ri_metrics.metrics: REGISTRY; default_plan(profile); validate_plan(plan); run_metrics(model_path, df_labelled, profile, plan, n_boot=200, seed=42)->dict; compute_baseline(model_path, holdout_df, profile, plan)->dict; compare(baseline, current, primary_metric)->dict
ri_analysis.summary: summarize(report); validate_summary(text, report); template_summary(report)
ri_analysis.chat: chat(question, project_id, tools=None)
ri_decision.decision: decide(drift, cara, quality, cfg=None)->dict; what_if(drift, cara, quality, cfg_overrides=None)->dict
ri_decision.retrain: train_candidate(profile, train_df, out_path); verify(candidate_path, holdout_df, profile); promote(candidate_path, active_path); rollback(previous_path, active_path)
ri_decision.orchestrator: setup_project(profile); on_new_data(project_id, file_path, labels_path=None); attach_labels(project_id, batch_id, labels_path); resolve_human(project_id, report_id, choice, force=False)
SANDBOX: the orchestrator calls model-touching functions as ri_init.worker.run_in_worker('ri_cara.cara_t:calibrate', kwargs)->dict. Functions take paths and DataFrames and return dicts.
STUBS: until a teammate replaces them, functions return valid dicts with 'status':'STUB' and the module has IS_STUB = True.
MODEL: a scikit-learn Pipeline that accepts raw rows; train copies with sklearn.base.clone(pipeline).fit(X, y). Never run scripts from a user's repo.

# Contract v2
- cara_t result dict gains keys 'basis' ('measured'|'estimated'|'none'), 'staleness_est' (float|None), 'staleness_est_lo' (float|None), 'decision_est' ('KEEP'|'RETRAIN'|None); 'status' may also be 'ESTIMATED'. Measured results have basis='measured' and the est fields None.
- ri_cara.cara_t gains estimate_staleness(model_path, new_df, train_sample_df, calib, profile, seed=42)->dict (label-free; same dict with basis='estimated', status='ESTIMATED').
- calibrate() result gains 'proxy': {'conf_cal':{'x':[float],'y':[float]},'map':{'x':[float],'y':[float],'resid_q90':float},'spearman':float}.
- ri_decision.decision.decide: if cara['basis']=='estimated', use staleness_est_lo for zc, cap the zone at 'HUMAN', and put 'estimated from model confidence, labels pending' in 'reason'.
- orchestrator: an unlabelled batch runs drift + estimate_staleness and ends in HUMAN with the alert 'request labels'; when labels are attached later the measured result is computed and 'proxy_error' (estimated minus measured staleness) is stored in the report.
- add ri_cara/proxy_eval.py stub: run_proxy_eval(profile, cfg)->DataFrame.
- FOLDERS update: 3-CARA/ri_cara: cara_t.py replay_baselines.py proxy_eval.py (Udaykiran).
