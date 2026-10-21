# Migration and provenance

Original top-level Python scripts are copied into `backup/` using their exact Git blobs. Existing analyses and figures should be regenerated with those archived scripts until each workflow is compared to the modular implementation.

| Original | Intended modular destination |
|---|---|
| A0_detect_events.py, A0_detect_flood_events.py, A0_detect_rain_events.py, idFlood.py | gwflood/events.py and data adapters |
| A1_prepare_data.py | gwflood/data.py; domain-specific rasters / Earth Engine adapters pending |
| A2_causal_model_train.py, A2_causal_model_train_1.py, A2_causal_model_interpret.py | gwflood/causal.py |
| A2_SHAP_model_train.py, model.py | gwflood/predictive.py; TabPFN/SHAP migration pending |
| A3_results_visual.py, A3_results_visual_supl.py, Figures.py | gwflood/plots.py; publication figures pending |
| A4_national_predict.py, A4_national_predict_a.py, A4_support_evaluate.py | gwflood/spatial.py; mapping adapter pending |

## Important differences

* New CLI tools accept prepared CSV tables and never authenticate to external services at import time.
* The new rainfall delineator is intentionally simple; it is **not** guaranteed identical to the historical long-event main-pulse and discharge-response procedures. Before replacing publication event data, add regression tests using historical saved events.
* Historical treatment is `dtp` (catchment-centred, where higher values indicate shallower groundwater). A +0.1 m contrast therefore represents groundwater rising by 10 cm in the historical convention. Verify on your source data.
* The original causal script used `GroupKFold` across catchments for the evaluation loop. Within-model nuisance cross-fitting needs separate scrutiny. The new code exposes grouped outer evaluation and logs sample counts.
* Global-scale estimation is out-of-sample transport of a conditional effect model, not global causal identification or validation. Inspect covariate distributions; only compatible feature definitions and units are accepted.
* Original `A2_SHAP_model_train.py` contains a literal authentication token. Treat it as compromised: revoke/rotate it in the issuing service, remove it from Git history if necessary, and load any new token from the environment. The archived copy is retained solely for provenance, so restrict access as appropriate.

No historical scientific output is claimed to be reproduced by this initial migration.
