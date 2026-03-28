"""
MLflow training pipeline — AI vs Human Text Detection.
Uses TF-IDF as the primary feature (vocabulary signal).
Runs: Baseline RF → Tuned XGBoost grid search.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os, time, json, pickle, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from mlflow.models.signature import infer_signature
from xgboost import XGBClassifier

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'processed_data.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
EXPERIMENT = "ai_text_detector"
MODEL_NAME = "ai_human_text_classifier"
os.makedirs(MODEL_DIR, exist_ok=True)


def compute_metrics(y_true, y_pred, y_prob):
    return {
        'auc':       round(roc_auc_score(y_true, y_prob), 4),
        'accuracy':  round(accuracy_score(y_true, y_pred), 4),
        'f1':        round(f1_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred), 4),
        'recall':    round(recall_score(y_true, y_pred), 4),
    }


def load_data():
    df = pd.read_csv(DATA_PATH)
    train_df, rem = train_test_split(df, train_size=0.6, random_state=42, stratify=df['label'])
    val_df, test_df = train_test_split(rem, test_size=0.5, random_state=42, stratify=rem['label'])
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def make_vectorizer():
    return TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        strip_accents='unicode',
        analyzer='word',
    )


def run_baseline(train_df, val_df, test_df):
    print("\n-- Run 1: Baseline Random Forest --")
    with mlflow.start_run(run_name='baseline_random_forest') as run:
        vec = make_vectorizer()
        X_train = vec.fit_transform(train_df['text'])
        X_val   = vec.transform(val_df['text'])
        X_test  = vec.transform(test_df['text'])
        y_train, y_val, y_test = train_df['label'].values, val_df['label'].values, test_df['label'].values

        params = {'n_estimators': 200, 'max_depth': 10, 'random_state': 42}
        mlflow.log_params(params)
        mlflow.log_param('model_type', 'RandomForestClassifier')
        mlflow.log_param('tfidf_features', 300)

        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        for split, X, y in [('val', X_val, y_val), ('test', X_test, y_test)]:
            m = compute_metrics(y, model.predict(X), model.predict_proba(X)[:,1])
            for k, v in m.items(): mlflow.log_metric(f'{split}_{k}', v)

        test_m = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:,1])

        # Save pipeline (vectorizer + model together)
        pipeline = Pipeline([('tfidf', vec), ('clf', model)])
        sig = infer_signature(train_df[['text']], model.predict_proba(X_train)[:,1])
        mlflow.sklearn.log_model(pipeline, artifact_path="random_forest_model", signature=sig)

        with open(os.path.join(MODEL_DIR, 'rf_pipeline.pkl'), 'wb') as f:
            pickle.dump(pipeline, f)

        print(f"  test_auc={test_m['auc']} accuracy={test_m['accuracy']} run_id={run.info.run_id[:8]}")
        return run.info.run_id, test_m, model, vec


def run_xgboost(train_df, val_df, test_df):
    print("\n-- Run 2: Tuned XGBoost --")
    param_grid = [
        {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1,  'subsample': 0.8},
        {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.8},
        {'n_estimators': 150, 'max_depth': 5, 'learning_rate': 0.1,  'subsample': 0.9},
    ]

    # Fit vectorizer once on train
    vec = make_vectorizer()
    X_train = vec.fit_transform(train_df['text'])
    X_val   = vec.transform(val_df['text'])
    X_test  = vec.transform(test_df['text'])
    y_train, y_val, y_test = train_df['label'].values, val_df['label'].values, test_df['label'].values

    best_auc, best_run_id, best_pipeline, best_params, best_m = 0, None, None, None, None

    for i, params in enumerate(param_grid):
        with mlflow.start_run(run_name=f'xgboost_grid_{i+1}') as run:
            mlflow.log_params(params)
            mlflow.log_param('model_type', 'XGBClassifier')
            mlflow.log_param('tfidf_features', 300)

            model = XGBClassifier(**params, eval_metric='logloss', random_state=42, verbosity=0)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            val_m  = compute_metrics(y_val,  model.predict(X_val),  model.predict_proba(X_val)[:,1])
            test_m = compute_metrics(y_test, model.predict(X_test), model.predict_proba(X_test)[:,1])
            for k, v in val_m.items():  mlflow.log_metric(f'val_{k}', v)
            for k, v in test_m.items(): mlflow.log_metric(f'test_{k}', v)

            pipeline = Pipeline([('tfidf', vec), ('clf', model)])
            mlflow.sklearn.log_model(pipeline, artifact_path="xgboost_model", signature=infer_signature(
                 train_df[['text']], model.predict_proba(X_train)[:,1]
            ))

            print(f"  Grid {i+1}: val_auc={val_m['auc']} test_auc={test_m['auc']} acc={test_m['accuracy']}")

            if val_m['auc'] > best_auc:
                best_auc, best_run_id, best_pipeline, best_params, best_m = \
                    val_m['auc'], run.info.run_id, pipeline, params, test_m

    with open(os.path.join(MODEL_DIR, 'xgb_pipeline.pkl'), 'wb') as f:
        pickle.dump(best_pipeline, f)

    return best_run_id, best_m, best_pipeline, best_params


def register_models(rf_run_id, xgb_run_id):
    print("\n-- Registering Models --")
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    rf_mv  = mlflow.register_model(f"runs:/{rf_run_id}/random_forest_model", MODEL_NAME)
    time.sleep(3)
    client.set_registered_model_alias(MODEL_NAME, "baseline", rf_mv.version)
    xgb_mv = mlflow.register_model(f"runs:/{xgb_run_id}/xgboost_model", MODEL_NAME)
    time.sleep(3)
    client.set_registered_model_alias(MODEL_NAME, "production", xgb_mv.version)
    print(f"  RF v{rf_mv.version} -> baseline | XGB v{xgb_mv.version} -> production")


def save_summary(rf_run_id, xgb_run_id, rf_m, xgb_m, xgb_params):
    def fmt(m): return {k.upper() if len(k)<=3 else k.capitalize(): v for k,v in m.items()}
    summary = {
        "models": {
            "Random Forest (Baseline)": {
                "run_id": rf_run_id, "model_type": "RandomForestClassifier",
                "params": {"n_estimators": 200, "max_depth": 10, "tfidf_features": 300},
                "metrics": fmt(rf_m), "alias": "baseline",
                "pkl": "rf_pipeline.pkl"
            },
            "XGBoost (Production)": {
                "run_id": xgb_run_id, "model_type": "XGBClassifier",
                "params": {**xgb_params, "tfidf_features": 300},
                "metrics": fmt(xgb_m), "alias": "production",
                "pkl": "xgb_pipeline.pkl"
            }
        },
        "experiment": EXPERIMENT, "model_name": MODEL_NAME,
    }
    with open(os.path.join(MODEL_DIR, 'model_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("  Saved model_summary.json")


def sanity_check():
    print("\n-- Sanity Check (real dataset examples) --")
    with open(os.path.join(MODEL_DIR, 'rf_pipeline.pkl'),  'rb') as f: rf  = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'xgb_pipeline.pkl'), 'rb') as f: xgb = pickle.load(f)

    # Real examples from the dataset
    tests = [
        ("AI sample",
         'Recommender systems on social graphs are computationally heavy. We propose LightGCN, a simplified Graph Convolutional Network that removes non-linear activation functions and feature transformation to reduce complexity while maintaining accuracy.'),
        ("AI sample 2",
         'Geothermal energy is limited to volcanic regions. We investigate Supercritical Geothermal Systems where water is injected into hot rock at depths exceeding 5 km to reach a supercritical state.'),
        ("Human sample",
         'Armed conflicts present complex, multidimensional challenges that severely compromise both access to and the quality of healthcare, including the adequate prescription of essential medicines. This study examines antibiotic prescription patterns across three conflict-affected regions, finding significant deviations from WHO guidelines in over 60% of cases reviewed.'),
        ("Human sample 2",
         'Purpose: To determine the prevalence and causes of blindness and vision impairment among adults aged 50 years in Western Uganda. Methods: A population-based cross-sectional survey was conducted across 12 districts. Results showed that cataract accounted for 43% of blindness cases.'),
    ]
    for label, text in tests:
        rf_prob  = rf.predict_proba([text])[0]
        xgb_prob = xgb.predict_proba([text])[0]
        print(f"  {label}:")
        print(f"    RF  -> {'AI' if rf_prob[1]>0.5 else 'Human'} ({rf_prob[1]*100:.1f}% AI)")
        print(f"    XGB -> {'AI' if xgb_prob[1]>0.5 else 'Human'} ({xgb_prob[1]*100:.1f}% AI)")


def main():
    db_path = os.path.join(BASE_DIR, 'mlruns', 'mlflow.db').replace('\\', '/')
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment(EXPERIMENT)
    print(f"MLflow DB: {db_path}")

    train_df, val_df, test_df = load_data()

    rf_run_id,  rf_m,  _, _            = run_baseline(train_df, val_df, test_df)
    xgb_run_id, xgb_m, _, xgb_params   = run_xgboost(train_df, val_df, test_df)

    register_models(rf_run_id, xgb_run_id)
    save_summary(rf_run_id, xgb_run_id, rf_m, xgb_m, xgb_params)
    sanity_check()

    print(f"\nDone! RF AUC={rf_m['auc']} | XGB AUC={xgb_m['auc']}")

if __name__ == '__main__':
    main()