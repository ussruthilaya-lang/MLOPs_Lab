# Lab 1 – API Labs Submission 

- Course: MLOps
- Lab: API Labs (FastAPI + Streamlit)
- Submission Date: 30 Jan 2026

## Overview

This lab focuses on building and deploying an API-based machine learning system.

I trained a flower species classification model (Iris with extended Lily classes), exposed it through a FastAPI backend, and built a Streamlit dashboard to interact with the API. The backend was containerized and deployed using Google Cloud Build and Cloud Run.

The goal of this lab was to understand end-to-end API-based model serving and real cloud deployment constraints.

## What was implemented

- Trained a classification model on Iris + extended Lily dataset

- Built a FastAPI backend for model inference

- Exposed a /predict endpoint returning species, family, confidence, and probabilities

- Dockerized the FastAPI service

- Deployed the backend to Google Cloud Run using Cloud Build

- Built a Streamlit dashboard as a thin client for the API

---

# Lab 2 – Airflow ML Pipeline

- Course: MLOps
- Lab: Airflow Labs
- Submission Date: 14 Feb 2026

## Overview

In this lab, I worked on building and modifying an Airflow DAG that runs a machine learning pipeline using clustering on customer credit card data.

## What was implemented

The pipeline basically:

- Loads dataset

- Preprocesses data

- Trains clustering model

- Finds optimal number of clusters

- Saves model

- Loads model and predicts

But during the lab, I realised some architectural issues in the original workflow and I improved them.

This lab helped me understand how ML pipelines actually work inside orchestration tools like Airflow.

Here is a cleaner, slightly more concise student-style version while keeping it professional and structured.

---

# Lab 3 — LLM Data Pipeline

* **Course:** MLOps
* **Lab:** Distributed Financial Risk Classifier
* **Submission Date:** 28 Feb 2026

---

## Overview

In this lab, I applied concepts from Labs 1–4 to build a complete end-to-end financial sentiment classifier. I fine-tuned **DistilBERT** on the Financial PhraseBank dataset to classify financial sentences as positive, neutral, or negative.

Unlike earlier labs that focused on isolated concepts, this project combines everything into one working system: data handling, imbalance correction, distributed-ready training, modular scripts, and a deployed Streamlit interface.

The goal was not just model training, but building a clean, reproducible pipeline that reflects real MLOps practices.

---

## What Was Implemented

Pipeline steps:

* Load and inspect Financial PhraseBank (`sentences_allagree`)
* First pass: estimate class distribution and compute class weights
* Shuffle dataset with fixed seed and split sequentially (80/20)
* Tokenize using DistilBERT tokenizer (`max_length=128`)
* Fine-tune DistilBERT with weighted `CrossEntropyLoss`
* Evaluate with per-class precision, recall, F1
* Save checkpoint and serve predictions via Streamlit

Each decision directly reflects a lab concept.

---
# Lab 4 — Docker MLOps Pipeline

* **Course:** MLOps
* **Lab:** Spotify Hit Predictor — Dockerised ML Pipeline
* **Submission Date:** 13 Mar 2026

---

## Overview

In this lab, I applied Docker containerisation concepts to build a complete end-to-end hit prediction pipeline using Spotify audio features.

I trained an XGBoost classifier on the Spotify Tracks Dataset to predict whether a track is a hit (popularity ≥ 70), packaged the entire workflow into two Docker services, and served predictions via an interactive Streamlit dashboard.

Unlike earlier labs that focused on model training in isolation, this project combines data engineering, model training, container orchestration, and deployment into one working system.

The goal was not just to train a model, but to build a clean, reproducible pipeline that reflects real MLOps practices — where anyone can clone the repo and run `docker compose up` to get a working application.

---

## What Was Implemented

Pipeline steps:

* Download Spotify Tracks Dataset (114k rows) and generate a 5k stratified subset for version control
* Train XGBoost classifier with `scale_pos_weight` to handle class imbalance (21.5% hit rate)
* Save model artifacts (`model.joblib`, `scaler.joblib`, `metrics.json`) to a Docker named volume
* Serve predictions via Streamlit dashboard reading artifacts from the shared volume
* Orchestrate both services with `docker-compose.yml` using `depends_on: service_completed_successfully`

Each decision directly reflects a Docker and MLOps concept.
