# Lab 2 – Airflow ML Pipeline

# What I Learned From This Lab

## 1. How Airflow Actually Runs Things

Before this, I understood DAG concept theoretically, but now I understood practically:

* Scheduler schedules
* Worker runs task
* Webserver only shows UI
* XCom passes data between tasks

I also faced XCom serialization issues (numpy.int64 error), which made me understand that Airflow needs JSON safe return types.

---

## 2. Order of ML Logic Matters

Originally the model training logic was:

* Train models from k=1 to k=49
* Save the last trained model

But elbow method was detecting optimal k separately. That means the saved model was not actually the optimal one.

So I fixed this.

Now the workflow is:

Train models → detect optimal k → select correct model → save only optimal model.

This small fix made big logical difference.

---

## 3. Importance of Saving the Scaler

In ML, preprocessing must match between training and inference.

Earlier only model was saved.

Now I save:

* model
* scaler
* optimal_k

So prediction is consistent.

This made the system more correct from ML perspective.

---

# Customizations I Made

## 1. Changed Model Implementation

Instead of using only standard `KMeans`, I switched to:

MiniBatchKMeans

Reason:

* Faster for large data
* More scalable
* Shows clear modification from original lab

In logs it now prints:

Model type used: minibatch

So it is visible in Airflow logs.

---

## 2. Added Model Versioning

Earlier, every run was overwriting:

```
model/model.sav
```

Now each run creates a new folder:

```
model/<timestamp>/model.pkl
```

Example:

```
model/
  2026-02-14_13-22-10/
  2026-02-14_13-25-48/
```

This is closer to real MLOps practice.

Each model version stores:

* model
* scaler
* optimal_k
* model_type

---

## 3. Fixed Elbow Method Usage

Previously, the elbow logic was disconnected from model saving.

Now:

* Train models
* Compute SSE
* Use KneeLocator
* Select optimal model
* Save that model

If elbow fails, default fallback is k=3.

---

# How To Run (For TA)

### Step 1 – Start Docker

From project root:

```
docker compose up -d
```

Wait until all containers are healthy.

---

### Step 2 – Open Airflow UI

Open:

```
http://localhost:8080
```

Login:

Username: airflow
Password: airflow

---

### Step 3 – Trigger DAG

1. Enable DAG `Airflow_Lab1`
2. Click Trigger
3. Open Graph view

All tasks should turn green.

---

### Step 4 – Verify Custom Changes

### Check model type

Go to:

build_save_model_task → Logs

You should see:

```
Model type used: minibatch
```

---

### Check Model Versioning

Go to project folder:

```
model/
```

You should see timestamp-based folders.

---

### Check Prediction

Open:

load_model_task → Logs

You should see:

```
Done. Returned value was: <cluster_number>
```

---

# Final Reflection

This lab was more than just running Airflow.

The most important thing I learned is:

Even if pipeline runs successfully, logic can still be wrong.

Fixing model selection, reproducibility, and versioning made it more aligned with real ML systems.

Overall, this lab helped me connect ML theory with orchestration and production mindset.

---
