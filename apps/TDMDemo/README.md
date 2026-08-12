# TDM ShipBench Demo

Streamlit demo for APPSolver's multi-source ship-flow modeling, training, prediction, rollout warning, visualization, terminal, and export workflow.

## Scope

- Dataset: **ShipBench only** (`DTC`, `KCS`, `KVLCC2`; `1Re`, `2Re`)
- Modeling: APP-Transformer plus FNO and PCNO neural-operator baselines
- Training: launches the repository's existing training scripts as a local background process
- Prediction: runs the canonical APP-Transformer checkpoint for real single-step and autoregressive inference
- Monitoring: replays stored rollout metrics and triggers normalized-RMSE threshold warnings

The demo does not claim that APPSolver currently implements PINN equation-residual training. Instead, the physical-information modeling page exposes the implemented geometric, spectral, and point-cloud operator priors.

## Run

From the repository root:

```bash
/opt/miniconda3/envs/mesh/bin/streamlit run apps/TDMDemo/app.py
```

Or:

```bash
/opt/miniconda3/envs/mesh/bin/python -m streamlit run apps/TDMDemo/app.py
```

The local environment must contain the packages in `apps/TDMDemo/requirements.txt`. Real inference also requires:

- `datasets/shipBench/<HULL>/field/<RE>/flow_cache.npz`
- `outputs/natural_time_v2/joint_all_models_16k/<HULL>/app_transformer/seed42/model_best_mae.pth`
- the adjacent `normalization_stats.npz`

Training runs and logs are written to `outputs/tdm_demo/`, which is excluded from git by the repository's existing `outputs/` rule.
