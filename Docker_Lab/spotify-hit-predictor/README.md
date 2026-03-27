# Lab 4 – Docker Lab - Spotify Hit Predictor

# What I Learned From This Lab

## 1. Docker Is About Reproducibility, Not Just Containerisation

Before this lab, I thought Docker was mainly about packaging code into containers.

Now I understand the full MLOps pipeline:

* Separate training and serving into independent services
* Pass artifacts between containers via shared volumes
* Control startup order with dependency conditions
* Keep datasets and model files out of version control

Every design decision changes how reproducible and portable the pipeline actually is.

---

## 2. Named Volumes Replace Manual File Copying

In earlier approaches, I would have copied model files manually between steps.

In this pipeline, I used a Docker named volume:

* Training container writes `model.joblib`, `scaler.joblib`, `metrics.json` to `/exchange`
* Serving container reads from `/exchange` at startup

This meant:

* No file paths hardcoded between services
* The volume persists between container restarts
* Both services stay completely decoupled

I understood that volumes are not just storage — they are the communication channel between services.

---

## 3. Service Ordering Is a Real Engineering Problem

The serving container cannot start before training completes.

So I used:

```yaml
depends_on:
  model-training:
    condition: service_completed_successfully
```

This meant:

* Docker waits for the training container to exit with code 0
* If training fails, serving never starts
* No race conditions, no missing model files

Orchestration logic is not automatic — it requires explicit structure.

---

## 4. Class Imbalance Affects What the Model Learns

The dataset was imbalanced:

* Non-hits: ~78.5%
* Hits (popularity ≥ 70): ~21.5%

If trained normally, the model would prefer predicting Non-hit.

So I used:

* `scale_pos_weight` in XGBoost to correct class imbalance
* Stratified train/test split to preserve hit rate in both sets
* AUC-ROC as the primary metric — more meaningful than accuracy under imbalance

Result:

AUC-ROC ≈ 0.80
Accuracy ≈ 74%

This shows the model learned the minority class properly, not just the majority.

---

## 5. Environment Parity Matters More Than You Think

Running locally on Python 3.14 and Docker on `python:3.10-slim` caused silent version conflicts.

Packages like `scikit-learn==1.8.0` require Python ≥ 3.11 — no error at install time, just failure at runtime.

So I:

* Bumped the Docker image to `python:3.11-slim`
* Locked all package versions explicitly in `requirements.txt`
* Validated the same `requirements.txt` installs cleanly in both environments

This showed me that version pinning is not optional in production pipelines.

---

## 6. Production Structure Matters

I moved from a single script to a modular project:

```
spotify-hit-predictor/
├── docker-compose.yml
├── requirements.txt
├── create_subset.py
├── .gitignore
├── src/
│   ├── model_training.py
│   └── app.py
└── data/
    ├── spotify_tracks.csv     ← local only, git-ignored
    └── spotify_subset.csv     ← committed, 5k stratified rows
```

Each file has one responsibility.
The model can be retrained, evaluated, or redeployed independently.
Anyone cloning the repo can run `docker compose up` without any setup.

This felt much closer to real MLOps practice.

---

# Customisations I Made

## 1. Independent Dataset and Use Case

Instead of reusing the lab dataset, I chose the Spotify Tracks Dataset from Kaggle and framed it as a hit prediction problem.

This made the project more realistic and domain-focused.

---

## 2. Stratified Subset for Git

The full dataset (114k rows, ~20MB) cannot be committed to git.

So I wrote `create_subset.py` which:

* Bins tracks into popularity buckets (low / mid / high / hit)
* Samples proportionally across buckets
* Produces a 5k-row `spotify_subset.csv` (~500KB)

This means anyone cloning the repo has real, representative data without needing to download anything.

---

## 3. XGBoost With Imbalance Correction

Used XGBoost classifier with:

* `scale_pos_weight` = ratio of negatives to positives
* `n_estimators=300`, `max_depth=5`, `learning_rate=0.05`
* `subsample=0.8`, `colsample_bytree=0.8`

Result:

AUC-ROC ≈ 0.80
Hit recall improved vs baseline without weighting

---

## 4. Explainable Streamlit Dashboard

The app is not just a prediction form — it includes:

* Live hit probability gauge (updates with every slider move)
* Audio radar chart showing the track's fingerprint
* Feature importance bar chart from the trained model
* Decision explainer showing per-feature contribution toward hit/not-hit
* Dataset stats panel with popularity distribution and correlations
* How It Works tab explaining the model in plain English
* Pipeline diagram tab explaining the Docker architecture

The model is not just trained — it is explainable and deployable.

---

# How To Run (For TA)

### Step 1 – One-time dataset setup

Download the Spotify Tracks Dataset from Kaggle:
👉 https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset

Place it at `data/spotify_tracks.csv`, then generate the committed subset:

```bash
python create_subset.py
```

This writes `data/spotify_subset.csv`. Commit this file — it is the only data file in git.

---

### Step 2 – Run with Docker

```bash
docker compose up
```

Expected logs:

```
spotify_trainer  | [data] Loaded 5,000 rows
spotify_trainer  | [train] Hit rate: 21.5%  |  rows: 5,000
spotify_trainer  | [train] Fitting XGBoost …
spotify_trainer  | [eval] Accuracy: 0.744  |  AUC-ROC: 0.805
spotify_trainer  | [save] model.joblib → /exchange/model.joblib
spotify_dashboard | Streamlit running on http://0.0.0.0:8501
```

Open **http://localhost:8501**

---

### Step 3 – Run locally (venv)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

mkdir -p exchange
python src/model_training.py
streamlit run src/app.py
```

---

# Final Reflection

This lab helped me see that ML engineering is mostly about decisions outside the model:

* How containers communicate (volumes vs copying)
* How services coordinate (depends_on conditions)
* How environments stay consistent (image versions, pinned deps)
* How data is versioned without bloating the repo (stratified subsets)
* How predictions are made explainable (not just a number)

One key insight was the difference between running code and deploying code.
Training a model in a notebook is easy. Making it reproducible across machines, environments, and teams requires intentional structure.

Overall, this lab connected data engineering, model training, containerisation, and deployment into one coherent system.