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

---

# **Lab 5 – MLflow Experiment Tracking Lab Submission**
- **Course:** MLOps
- **Lab:** Experiment Tracking (MLflow + Streamlit)
- **Submission Date:** 28 Mar 2026

## **Overview**

This lab focuses on experiment tracking and model versioning using MLflow. I trained an AI vs Human text detection model on a dataset of 6,069 academic abstracts, tracked all experiments with MLflow, registered the best models with versioned aliases, and built a Streamlit dashboard with phrase-level explainability for live predictions.

The goal was to understand the full MLflow lifecycle from logging runs to registering production models and connect it to a meaningful real-world use case aligned with my Rubrix research paper evaluation project.

## **What was implemented**
- Downloaded and processed the AI and Human Text Dataset (6,069 academic abstracts, 51/49 split)
- Built a TF-IDF + sklearn Pipeline feature extraction approach
- Trained a baseline Random Forest and tuned XGBoost via grid search (3 configurations)
- Logged all runs to MLflow : params, metrics, feature importances, and model artifacts
- Registered both models in MLflow Model Registry with aliases `baseline` and `production`
- Built a Streamlit app showing model comparison, live predictions, phrase-level highlights, explainability bars, and edit-and-retest functionality
- Resolved MLflow 3.x Windows compatibility issues by downgrading to `mlflow==2.22.4` with SQLite backend

---
# Lab 6 – GCP Cloud Functions: Titanic Survival Predictor Submission
- **Course:** MLOps / Cloud Computing
- **Lab:** Serverless ML Deployment (GCP Cloud Functions)
- **Submission Date:** 07 Apr 2026

## Overview

This lab focuses on deploying a machine learning model as a serverless REST API using Google Cloud Run Functions. I trained a Logistic Regression model on the real Titanic dataset (714 passengers), deployed it as an HTTP-triggered Cloud Function, and built a standalone HTML demo UI with quick test presets and meme GIFs for live predictions.

The goal was to understand the full serverless deployment lifecycle — from writing an HTTP-triggered function to handling cold starts, input validation, CORS, and testing a live endpoint via curl and a browser UI.

## What was implemented
- Fetched real Titanic dataset (891 passengers, cleaned to 714 usable rows) live from GitHub on cold start
- Built a Logistic Regression model with StandardScaler using sklearn on 6 features: Pclass, Sex, Age, SibSp, Parch, Fare
- Implemented cold start optimisation by caching trained model to `/tmp` — warm instances skip retraining and respond in ~5ms
- Deployed as an HTTP-triggered Cloud Run Function on GCP with unauthenticated public access
- Added structured input validation returning 400 errors with field-level descriptions for bad requests
- Added CORS headers to all responses and handled OPTIONS preflight to support browser-based clients
- Tested endpoint via curl on Windows CMD confirming 200s for valid input and 400s for bad input
- Built a standalone HTML demo UI with dark theme, quick test presets (Jack, Rose, Rich 1st class man, 2nd class mum), adjustable sliders, random meme GIFs on result, and input recap showing exact JSON sent to the function
- Resolved CORS blocking in browser by adding `Access-Control-Allow-Origin: *` headers and handling OPTIONS method in the function
