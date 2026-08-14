# snowrahul
Customer Churn Prediction — End-to-End MLOps Pipeline on Snowflake
An end-to-end machine learning pipeline built entirely on Snowflake: feature engineering with Snowpark, model training with experiment tracking, model registration, real-time inference via Snowpark Container Services (SPCS), and production drift monitoring.

Snowflake Snowpark scikit--learn MLOps Python

Problem Statement
Predict which customers are likely to churn based on account attributes (tenure, monthly/total charges, support call volume, contract type) and serve those predictions as a real-time inference endpoint — with observability into prediction drift over time compared to the training data distribution.

Architecture
                     ┌─────────────────────┐
                     │   Synthetic Source   │
                     │   CUSTOMER_DATA      │
                     └──────────┬──────────┘
                                │  Snowpark pushdown transforms
                                ▼
                     ┌─────────────────────┐
                     │  CUSTOMER_FEATURES   │  (feature engineering)
                     └──────────┬──────────┘
                                │  train/test split (sklearn)
                                ▼
                 ┌───────────────────────────────┐
                 │   RandomForestClassifier       │
                 │   + Snowflake Experiment        │
                 │     Tracking (params/metrics)   │
                 └───────────────┬────────────────┘
                                 │  reg.log_model()
                                 ▼
                 ┌───────────────────────────────┐
                 │   Snowflake Model Registry      │
                 │   CHURN_MODEL (v: RUDE_SHEEP_2)  │
                 └───────────────┬────────────────┘
                                 │  mv.create_service()
                                 ▼
                 ┌───────────────────────────────┐
                 │  CHURN_INFERENCE_SERVICE (SPCS) │
                 │  REST endpoint · Auto-Capture   │
                 └───────────────┬────────────────┘
                                 │  captured request/response events
                                 ▼
                 ┌───────────────────────────────┐
                 │  CHURN_PREDICTIONS_LOG (SOURCE) │
                 │  vs. CHURN_BASELINE_PREDICTIONS │
                 └───────────────┬────────────────┘
                                 │  CREATE MODEL MONITOR
                                 ▼
                 ┌───────────────────────────────┐
                 │   CHURN_DRIFT_MONITOR           │
                 │   PSI · Jensen-Shannon ·         │
                 │   Wasserstein distance           │
                 └───────────────────────────────┘
Pipeline stages:

Feature Engineering — Snowpark DataFrame pushdown transforms (avg monthly spend, support-usage flags, contract-type one-hot flags) computed server-side, no data pulled to the client until training.
Training + Experiment Tracking — RandomForestClassifier trained with Snowflake's ExperimentTracking API logging hyperparameters and evaluation metrics per run.
Model Registry — Model registered to Snowflake Model Registry with auto-inferred signatures (enables explainability out of the box).
Real-Time Inference (SPCS) — Deployed as a REST inference service on Snowpark Container Services with Auto-Capture enabled for request/response logging.
Drift Monitoring — A MODEL MONITOR compares live prediction score distributions (SOURCE) against the training-data baseline (BASELINE) using PSI, Jensen-Shannon divergence, and Wasserstein distance.
Results
Model performance (held-out test set, 20% split):

Metric	Value
Accuracy	0.79
ROC AUC	0.775
F1 Score	0.31
Precision	0.55
Recall	0.22
Feature importance (RandomForest):

Feature Importance

Tenure, monthly charges, and contract type were the strongest churn predictors — consistent with the synthetic data-generating process.

Drift monitoring (sample validation run):

Metric	Value	Sample sizes (source / baseline)
Population Stability Index (PSI)	7.39	20 / 4000
Jensen-Shannon Divergence	0.371	20 / 4000
Wasserstein Distance	0.053	20 / 4000
High PSI here reflects the small validation sample size (20 live predictions vs. 4000 baseline rows), not genuine concept drift — the low Wasserstein distance is a better sanity signal for this sample. In production, this monitor should be fed by real traffic accumulating over days/weeks before PSI is trustworthy.

Tech Stack
Snowflake · Snowpark Python · Snowflake ML (Model Registry, Experiment Tracking, Model Monitor) · Snowpark Container Services (SPCS) · scikit-learn · RandomForestClassifier · Python · pandas · matplotlib

Tags: #snowflake #mlops #machine-learning #snowpark #model-registry #spcs #drift-detection #data-science

Repository Structure
churn-prediction-snowflake/
├── README.md
├── notebook/
│   └── ml_e2e_project.ipynb        # Full end-to-end pipeline
├── images/
│   └── feature_importance.png      # Exported chart
└── sql/
    └── create_monitor.sql          # CREATE MODEL MONITOR statement
Notable Engineering Decisions & Debugging
This project intentionally documents two real issues hit and resolved during deployment — useful context for anyone reviewing the work:

Compute pool mismatch: Initially deployed the inference service on a GPU compute pool. Deployment failed (missing 'nvidia.com/gpu' in container spec) because the sklearn model doesn't request GPU resources. Fixed by redeploying on a CPU compute pool.
Model signature mismatch: The model was first registered with a manually specified signatures dict inferred from the target column (1 output), which broke predict_proba (actually 2 output columns — per-class probability). Fixed by re-registering with sample_input_data, letting Snowflake auto-infer per-method signatures correctly (this also enabled model explainability).
Auto-Capture routing: Discovered that calls through the SQL service-function wrapper (mv.run()) are not captured by Auto-Capture when DISABLE_AUTOCAPTURE_FOR_SERVICE_FUNCTION is set — only raw REST calls to the service endpoint are logged. Switched test traffic to hit the REST endpoint directly to validate Auto-Capture end-to-end.
How to Reproduce
Create a database/schema in Snowflake (e.g. ML_DEMO.PUBLIC).
Run the notebook cells in order — it will:
Generate a synthetic customer dataset and write it to a table
Engineer features with Snowpark
Train and register the model
Deploy the inference service (requires a compute pool)
Query Auto-Capture logs and set up the drift monitor
Adjust compute pool name, service name, and database/schema to match your environment.
License
MIT
