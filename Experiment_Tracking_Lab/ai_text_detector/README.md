# Lab 5 – MLflow Experiment Tracking Lab - AI vs Human Text Detector

## What I Learned From This Lab

### 1. MLflow Is About Reproducibility Across Model Versions
Before this lab, I thought MLflow was just a logging tool.

Now I understand the full experiment tracking lifecycle:

- Log every run with params, metrics, and artifacts so nothing is lost
- Register models with versioned aliases (`baseline`, `production`)
- Compare runs side by side to justify which model gets promoted
- Store the full pipeline (vectorizer + model) so inference is always consistent

Every design decision — what to log, how to version, what to register — changes how reproducible and auditable the pipeline actually is.

### 2. Pipelines Replace Manual Artifact Passing
In earlier approaches, I would have saved the vectorizer and model as separate files and manually loaded both at inference time.

In this pipeline, I used a sklearn `Pipeline`:

- Training fits `TfidfVectorizer` + classifier in one object
- Inference calls `pipeline.predict_proba([text])` — one line, no separate vectorizer
- The pipeline is pickled once and loaded once in Streamlit

This meant:

- No mismatch between training and inference vocabulary
- No missing vectorizer errors at serving time
- MLflow logs the full pipeline as a single artifact with a model signature

I understood that pipelines are not just convenience — they are the contract between training and serving.

### 3. Model Registry Solves the "Which Model?" Problem
Saving a `.pkl` file tells you nothing about which experiment produced it.

So I used MLflow Model Registry to:

- Register the Random Forest as version 1 with alias `baseline`
- Register the best XGBoost as version 2 with alias `production`
- Attach run IDs, params, and metrics to each registered version

This meant:

- The Streamlit app loads by alias, not by filename
- Promoting a new model is an alias update, not a file swap
- Any version can be rolled back without retraining

Model versioning is not automatic — it requires explicit registration structure.

### 4. Dataset Quality Affects Everything Downstream
The dataset had a fundamental structural imbalance:

- AI texts: ~44 words average (short summaries)
- Human texts: ~270 words average (full papers)

If trained with raw length-based features, the model learned "short = AI, long = Human" — not actual writing style.

So I iterated through three approaches:

1. Handcrafted stylometric features → model learned word count as a proxy
2. Length-independent ratio features → still leaked via sentence count
3. TF-IDF vocabulary features in a Pipeline → model learned actual word patterns

Result:

- AUC ≈ 0.9996 (RF) and 0.9994 (XGBoost) on held-out test set
- Sanity check on real dataset examples confirmed correct predictions
- Toy "AI-sounding" sentences failed because they were out-of-distribution

This showed me that feature engineering decisions have more impact than model choice.

### 5. Windows + MLflow Has Real Compatibility Issues
Running MLflow 3.10.1 on Windows caused three separate failures:

- `file://` URI scheme not valid on Windows paths
- Model registry unsupported without SQLite backend
- MLflow UI rendering broken on 3.x (frontend bug)

So I:

- Switched tracking URI to `sqlite:///mlruns/mlflow.db` with forward-slash normalisation
- Downgraded to `mlflow==2.22.4` for stable UI rendering
- Pinned `mlflow-skinny==2.22.4` to match and remove version mismatch warnings

This showed me that version pinning is not optional in MLOps pipelines — it is the difference between a lab that runs and one that does not.

### 6. Explainability Is a Separate Layer From the Model
The TF-IDF model cannot explain its predictions in human terms — it learned vocabulary weights, not semantic rules.

So I built a separate explainability layer in the Streamlit app using regex phrase matching:

- Hedge words (`however`, `furthermore`, `moreover`) → AI signal
- Filler phrases (`in conclusion`, `this study aims`) → AI signal
- Citation markers (`[12]`, `(2023)`) → Human signal
- Passive voice patterns → Human signal

This layer runs independently of the model and gives the TA something concrete to inspect and edit against.

---

## Customisations I Made

### 1. Independent Dataset and Use Case
Instead of the Wine Quality dataset from the lab template, I chose the **AI and Human Text Dataset** from Kaggle — a collection of 6,069 academic abstracts labelled as AI-generated or human-written.

This connected directly to my previous Rubrix project (research paper evaluation using RAG), making the lab meaningful rather than just a template exercise.

### 2. Balanced Subset for Git
The full dataset (6,069 rows) is kept out of version control.

Instead, `prepare_data.py` generates `processed_data.csv` locally:

- Samples 2,000 AI and 2,000 Human texts (balanced)
- Strips the index column and normalises label names
- Outputs a clean 4,000-row CSV ready for training

Anyone cloning the repo runs `prepare_data.py` once with the raw CSV and gets a reproducible processed dataset.

### 3. TF-IDF Pipeline With Two Models
Used sklearn `Pipeline(TfidfVectorizer + classifier)` with:

**Baseline — Random Forest:**
- `n_estimators=200`, `max_depth=10`
- TF-IDF: 300 features, bigrams, sublinear TF

**Production — XGBoost (grid search across 3 configs):**
- Best config: `n_estimators=100`, `max_depth=4`, `learning_rate=0.1`, `subsample=0.8`
- TF-IDF: same vectorizer fitted once on train, shared across grid runs

Result:

- RF AUC ≈ 0.9996 | Accuracy ≈ 0.9950
- XGBoost AUC ≈ 0.9994 | Accuracy ≈ 0.9938

### 4. Explainable Streamlit Dashboard
The app is not just a prediction form — it includes:

- **Model comparison panel** showing both MLflow-tracked models with full metrics and hyperparameters
- **Phrase analysis** with colour-coded highlights showing exactly which phrases triggered AI or Human signals
- **Per-model confidence bars** with AI/Human probability breakdown
- **Explainability block** per model showing signal strength bars for each phrase category and why each one matters
- **Edit & Retest tips** giving specific, actionable suggestions to make text sound more human based on detected phrases
- **Sample texts** drawn from the real dataset so predictions are always confident and meaningful

---

## Issues I Faced and What I Learned

| Issue | Root Cause | Fix | Learning |
|---|---|---|---|
| `file://` URI error on Windows | MLflow 3.x rejects Windows backslash paths | `.replace('\\', '/')` + SQLite URI | Path handling is OS-specific |
| Model registry unsupported | File-based store has no registry support | Switch to `sqlite:///mlruns/mlflow.db` | Registry needs a proper backend |
| MLflow UI shows "Something went wrong" | Frontend rendering bug in MLflow 3.x on Windows | Downgrade to `mlflow==2.22.4` | Always pin versions in requirements |
| `name=` kwarg error after downgrade | `name` param added in MLflow 3.x | Change to `artifact_path=` | API changes between major versions |
| Model predicts everything as AI | Word count leaked the label (AI=44 words, Human=270) | Switch to TF-IDF Pipeline | Feature engineering > model choice |
| Toy sanity check texts misclassified | Out-of-distribution vocabulary, not in training data | Use real dataset examples as samples | Test with in-distribution data |
| venv not activating in PowerShell | Execution policy blocked `.ps1` scripts | `Set-ExecutionPolicy RemoteSigned` | Windows security policy affects dev tools |

---

## Project Structure

```
ai_text_detector/
├── data/
│   ├── data_for_preprocessing.csv   ← raw download, git-ignored
│   └── processed_data.csv           ← generated by prepare_data.py, git-ignored
├── src/
│   ├── features.py                  ← phrase matching for explainability
│   ├── prepare_data.py              ← generates processed_data.csv from raw
│   ├── train.py                     ← MLflow training pipeline
│   └── app.py                       ← Streamlit demo app
├── models/
│   ├── rf_pipeline.pkl              ← saved RF pipeline
│   ├── xgb_pipeline.pkl             ← saved XGBoost pipeline
│   └── model_summary.json           ← metrics + run IDs for Streamlit
├── mlruns/
│   └── mlflow.db                    ← SQLite MLflow backend
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How To Run

### Step 1 – Clone and set up environment
```bash
git clone <repo-url>
cd ai_text_detector

python -m venv venv
venv\Scripts\Activate.ps1       # Windows PowerShell
# or
source venv/bin/activate         # Mac/Linux

pip install -r requirements.txt
```

### Step 2 – Download dataset
Download the **AI and Human Text Dataset** from Kaggle:
👉 https://www.kaggle.com/datasets/hasanyiitakbulut/ai-and-human-text-dataset

Place it at:
```
data/data_for_preprocessing.csv
```

### Step 3 – Prepare data
```bash
python src/prepare_data.py
```

Expected output:
```
Saved: data/processed_data.csv | Shape: (4000, 2)
Balance: {1: 2000, 0: 2000}
Next: python src/train.py
```

### Step 4 – Train models and log to MLflow
```bash
python src/train.py
```

Expected output:
```
MLflow DB: .../mlruns/mlflow.db
Train: 2400 | Val: 800 | Test: 800

-- Run 1: Baseline Random Forest --
  test_auc=0.9996  accuracy=0.9950

-- Run 2: Tuned XGBoost (grid search) --
  Grid 1: val_auc=...  test_auc=...
  Grid 2: val_auc=...  test_auc=...
  Grid 3: val_auc=...  test_auc=...

-- Registering Models --
  RF v1 -> baseline | XGB v2 -> production

-- Sanity Check --
  AI sample:  RF -> AI (96%) | XGB -> AI (99%)
  Human sample: RF -> Human | XGB -> Human
```

### Step 5 – View MLflow UI
```bash
mlflow ui --backend-store-uri "sqlite:///mlruns/mlflow.db"
```

Open `http://127.0.0.1:5000`

You will see:
- Experiment: `ai_text_detector`
- 4 runs: `baseline_random_forest`, `xgboost_grid_1`, `xgboost_grid_2`, `xgboost_grid_3`
- Metrics, params, and feature importances logged per run
- Model Registry: `ai_human_text_classifier` with versions 1 (baseline) and 2 (production)

### Step 6 – Launch Streamlit app
```bash
streamlit run src/app.py
```

Open `http://localhost:8501`

- Click **Load AI Sample** or **Load Human Sample** to test with real dataset examples
- Type any academic abstract and click **Analyze Text**
- See phrase highlights, model predictions, confidence bars, and explainability breakdown
- Edit the text and re-analyze to see predictions change in real time

---

## Final Reflection

This lab helped me see that MLOps is mostly about decisions outside the model:

- How experiments are tracked (params, metrics, artifacts per run)
- How models are versioned (registry, aliases, promotion workflow)
- How training and serving stay consistent (pipelines, not loose files)
- How environments stay reproducible (pinned versions, SQLite backend)
- How predictions are made explainable (phrase-level signal layer on top of TF-IDF)

One key insight was the difference between a model that works and a model that is deployable. Training a classifier in a notebook is easy. Making it versioned, registered, explainable, and serveable requires intentional structure at every step.

The dataset quality issue — where word count leaked the label — was the most educational challenge. No amount of model tuning fixed it. Only rethinking the features from scratch (TF-IDF vocabulary instead of handcrafted length ratios) resolved it. That is a lesson that applies to every future ML project.