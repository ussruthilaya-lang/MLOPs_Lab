# Lab 6 – GCP Cloud Functions: Titanic Survival Predictor

## What I Learned From This Lab

### 1. Cloud Functions Are Serverless ML Inference Endpoints
Before this lab, I thought deploying a model meant setting up a server.

Now I understand the serverless deployment lifecycle:

- Write a Python function with an HTTP trigger — no server config needed
- GCP provisions, scales, and tears down instances automatically
- The function handles one request at a time, scales to thousands with no code changes
- Free tier covers 2 million invocations per month — zero cost for experimentation

Every design decision — entry point naming, cold start handling, input validation — changes how reliable and production-ready the endpoint actually is.

### 2. Cold Start Optimisation With `/tmp` Caching
In earlier approaches, I would have retrained the model on every request.

In this function, I used `/tmp` caching:

- First request (cold start) fetches the Titanic CSV, trains the model, saves to `/tmp`
- Subsequent requests (warm instances) load directly from `/tmp` — no retraining
- Latency drops from seconds to milliseconds on warm requests

This meant:

- No redundant training cycles on every invocation
- Warm instance latency of ~5ms vs cold start of ~3–5 seconds
- The same pattern used in production ML inference systems

I understood that cold start handling is not optional in serverless ML — it is the difference between a slow endpoint and a fast one.

### 3. Real Dataset Over Hardcoded Data
Loading a real dataset from a public URL changes what the model actually learns:

- Raw Titanic CSV: 891 passengers from `datasciencedojo/datasets` on GitHub
- After cleaning: 714 rows (dropped nulls in Age, Fare, Sex)
- Features: Pclass, Sex, Age, SibSp, Parch, Fare
- Label: Survived (0 or 1)

This meant:

- Model trained on real survival patterns, not 15 hardcoded rows
- Logistic regression learned that Sex and Pclass are the strongest predictors
- Predictions reflect actual historical outcomes — women and 1st class passengers survive at higher rates

Hardcoded training data is fine for hello-world. Real datasets are required for meaningful predictions.

### 4. Input Validation Is a First-Class Concern
A deployed endpoint receives arbitrary input - not clean notebook data.

So I built a validation layer before any model inference:

- Checks Content-Type is `application/json`
- Checks `features` key exists in request body
- Checks exactly 6 numeric values are provided
- Returns structured error messages with field names on failure

This meant:

- Bad requests return 400 with a clear explanation, not a 500 crash
- The model only ever receives clean, shaped input
- Error handling is testable independently of the model

Validation is not defensive programming — it is the API contract with whoever calls the endpoint.

### 5. CORS Is a Real Deployment Concern
Building a browser-based UI that calls a Cloud Function exposed a fundamental web security constraint:

- Browsers block cross-origin requests unless the server explicitly allows them
- Cloud Run does not add CORS headers automatically
- The function must return `Access-Control-Allow-Origin: *` on every response
- OPTIONS preflight requests must be handled separately before the POST

So I:

- Added a CORS header to every `jsonify` response via a `make_response` wrapper
- Handled `OPTIONS` method explicitly to return 204 with correct preflight headers
- Tested with both curl (no CORS) and browser UI (CORS enforced) to catch the difference

This showed me that an API that works in curl does not automatically work in a browser — CORS is a separate deployment concern.

### 6. Serverless Has Constraints That Shape Architecture
Running ML inference on Cloud Run Functions exposed real constraints:

- No persistent disk — model must be retrained or fetched on cold start
- `/tmp` is ephemeral — cleared between cold starts, shared within warm instance lifetime
- No background threads — everything must complete within the request lifecycle
- Memory limit matters — large models or datasets can hit the default 256MB cap

So I:

- Chose logistic regression over heavier models to stay within memory limits
- Fetched the dataset from a public URL rather than bundling it in the container
- Used `joblib` for fast model serialisation into `/tmp`
- Kept the function stateless — no global mutable state between requests

Serverless architecture is not just a deployment choice — it constrains what the model and data pipeline can look like.

---

## Customisations I Made

### 1. Real Titanic Dataset Instead of Iris
Instead of the Iris dataset from the lab template, I used the real Titanic passenger dataset fetched live from GitHub on cold start — 891 rows, cleaned to 714 usable records.

This made predictions historically meaningful: the model correctly identifies that Jack (3rd class, male, age 22, fare $7) has a 9% survival chance and Rose (1st class, female, age 17, fare $100) has an 85% survival chance — matching real historical outcomes.

### 2. Structured JSON Responses With Confidence Scores
The endpoint returns more than just a class label:

```json
{
  "survived": false,
  "survival_label": "Did not survive",
  "confidence": {
    "not_survived": 0.906,
    "survived": 0.094
  }
}
```

This gives callers probability scores, not just a binary prediction — more useful for downstream decision-making and easier to inspect during grading.

### 3. Interactive HTML Demo UI
Built a standalone HTML file that calls the live Cloud Function endpoint:

- Dark-themed UI with ship emoji header — no external dependencies
- Quick test presets for Jack, Rose, Rich 1st class man, 2nd class mum — one click loads and predicts
- Sliders and dropdowns for all 6 features — fully adjustable
- Meme GIFs on result — random each prediction (celebration if survived, "this is fine" if not)
- Input recap in monospace showing exact JSON sent to the function
- Confidence pills showing survival vs death probability

The UI runs locally and calls the live GCP endpoint — no hosting required.

---

## Project Structure

```
titanic_gcp_function/
├── main.py                  ← Cloud Function entry point + model training + inference
├── requirements.txt         ← Python dependencies pinned for Cloud Run
└── titanic_predictor.html   ← Standalone demo UI (open locally in browser)
```

---

## How To Run

### Step 1 – Enable Cloud Functions API
1. Go to `console.cloud.google.com`
2. Search `Cloud Functions API` → click **Enable**

### Step 2 – Create the Cloud Run Function
1. Search `Cloud Run Functions` → click **+ Create Function**
2. Select **"Use an inline editor to create a function"**
3. Configure:

| Field | Value |
|---|---|
| Service name | `titanic-predictor` |
| Region | `us-central1` |
| Runtime | `Python 3.11` |
| Entry point | `predict_survival` |
| Authentication | Allow unauthenticated invocations |

### Step 3 – Paste the code
In the inline editor:

- `main.py` tab → paste full function code
- `requirements.txt` tab → paste dependencies

```
flask==3.1.0
scikit-learn==1.6.1
numpy==2.2.4
joblib==1.4.2
pandas==2.2.2
```

Click **Deploy** and wait for the green checkmark (~2–3 minutes).

### Step 4 – Test via curl

**Mac/Linux:**
```bash
# Jack — should not survive
curl -X POST https://YOUR-URL.run.app \
  -H "Content-Type: application/json" \
  -d '{"features": [3, 1, 22, 1, 0, 7.25]}'

# Rose — should survive
curl -X POST https://YOUR-URL.run.app \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 0, 17, 1, 2, 100]}'

# Bad input — should return 400
curl -X POST https://YOUR-URL.run.app \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 0]}'
```

**Windows CMD:**
```cmd
curl -X POST https://YOUR-URL.run.app -H "Content-Type: application/json" -d "{\"features\": [3, 1, 22, 1, 0, 7.25]}"

curl -X POST https://YOUR-URL.run.app -H "Content-Type: application/json" -d "{\"features\": [1, 0, 17, 1, 2, 100]}"

curl -X POST https://YOUR-URL.run.app -H "Content-Type: application/json" -d "{\"features\": [1, 0]}"
```

Expected responses:

```json
// Jack
{"confidence":{"not_survived":0.906,"survived":0.094},"survived":false,"survival_label":"Did not survive"}

// Rose
{"confidence":{"not_survived":0.15,"survived":0.85},"survived":true,"survival_label":"Survived"}

// Bad input
{"error":"Need exactly 6 values","fields":["Pclass(1/2/3)","Sex(0=female,1=male)","Age","SibSp","Parch","Fare"]}
```

### Step 5 – Open the demo UI
1. Download `titanic_predictor.html`
2. Open it directly in Chrome (double-click or drag into browser)
3. Click any preset passenger to auto-load and predict
4. Adjust sliders for custom passengers and hit **Predict my fate**

### Step 6 – Verify logs in GCP
1. Go to **Cloud Run** → `titanic-predictor` → **Logs** tab
2. Confirm 200 responses for valid requests and 400 for bad input
3. On first cold start, look for: `Titanic model trained and saved.`

---

## Final Reflection

This lab helped me see that deploying ML is mostly about decisions outside the model:

- How the model survives cold starts (caching in `/tmp`, not retraining every request)
- How the API handles bad input (validation layer before any model code runs)
- How training and serving stay consistent (same scaler fitted once, applied at inference)
- How browser clients differ from curl clients (CORS is enforced by browsers, invisible in terminal)
- How serverless constraints shape architecture (no disk, memory limits, stateless design)

One key insight was the difference between a model that works in a notebook and one that is deployable. Training logistic regression on Titanic data takes three lines. Making it versioned, validated, cached, and accessible over HTTP from any client requires intentional structure at every step.

The CORS debugging was the most educational challenge. The function worked perfectly in curl but was completely unreachable from the browser UI. No amount of model or logic changes fixed it — only understanding the browser security model and adding the correct response headers resolved it. That is a lesson that applies to every future API deployment.