from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from data import load_data, split_data
from pathlib import Path


def fit_model(X_train, y_train):
    """
    Train a Random Forest classifier and persist it to disk.
    """
    rf_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    rf_classifier.fit(X_train, y_train)

    MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
    MODEL_DIR.mkdir(exist_ok=True)

    MODEL_PATH = MODEL_DIR / "flower_model_v2.pkl"

    joblib.dump(rf_classifier, MODEL_PATH)

    return rf_classifier, MODEL_PATH


if __name__ == "__main__":
    X, y, label_encoder = load_data()

    X_train, X_test, y_train, y_test = split_data(X, y)

    model, model_path = fit_model(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 Validation Performance (v2):\n")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
        )
    )

    ENCODER_PATH = model_path.with_suffix(".labels.pkl")
    joblib.dump(label_encoder, ENCODER_PATH)

    print(f"\n✅ Model saved to: {model_path}")
    print(f"✅ Label encoder saved to: {ENCODER_PATH}")
