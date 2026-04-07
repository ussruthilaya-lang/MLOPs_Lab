import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from flask import request, jsonify
import joblib

MODEL_PATH = '/tmp/titanic_model.joblib'

def get_training_data():
    """
    Hardcoded Titanic-style training data.
    Features: [Pclass, Sex (0=female,1=male), Age, SibSp, Parch, Fare]
    Label: 0 = did not survive, 1 = survived
    """
    X = np.array([
        [3, 1, 22, 1, 0, 7.25],   # Jack-like: 3rd class, male, young
        [1, 0, 38, 1, 0, 71.28],  # Rose-like: 1st class, female
        [3, 0, 26, 0, 0, 7.92],   # 3rd class female
        [1, 0, 35, 1, 0, 53.10],  # 1st class female
        [3, 1, 35, 0, 0, 8.05],   # 3rd class male
        [1, 1, 54, 0, 0, 51.86],  # 1st class male older
        [3, 1,  2, 3, 1, 21.07],  # Child 3rd class
        [3, 0, 27, 0, 2, 11.13],  # 3rd class female w/ children
        [2, 1, 29, 0, 0, 13.00],  # 2nd class male
        [1, 0, 58, 0, 0, 26.55],  # 1st class older female
        [3, 1, 20, 0, 0, 7.85],   # 3rd class young male
        [2, 0, 30, 1, 0, 26.00],  # 2nd class female
        [1, 1, 40, 0, 0, 30.50],  # 1st class male
        [3, 0, 14, 1, 0, 11.24],  # 3rd class young female
        [2, 1, 45, 0, 0, 13.50],  # 2nd class older male
    ])
    y = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    return X, y

def train_and_save_model():
    """Trains the Titanic model and saves it."""
    X, y = get_training_data()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=200)
    model.fit(X_scaled, y)
    joblib.dump((scaler, model), MODEL_PATH)
    print("Titanic model trained and saved.")

def load_model():
    """Loads model, trains if not found (cold start)."""
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()
    return joblib.load(MODEL_PATH)

def validate_features(features):
    """
    Validates input: must be a list of 6 numeric values.
    [Pclass, Sex, Age, SibSp, Parch, Fare]
    """
    if not isinstance(features, list) or len(features) != 6:
        return False
    try:
        [float(x) for x in features]
        return True
    except ValueError:
        return False

def predict_survival(request):
    """Entry point for Cloud Function - predicts Titanic survival."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400

        body = request.get_json()

        if not body or "features" not in body:
            return jsonify({
                "error": "Missing 'features'",
                "expected_format": {
                    "features": "[Pclass, Sex, Age, SibSp, Parch, Fare]",
                    "example": [3, 1, 22, 1, 0, 7.25]
                }
            }), 400

        features = body["features"]

        if not validate_features(features):
            return jsonify({
                "error": "Invalid input. Need exactly 6 numeric values.",
                "fields": ["Pclass(1/2/3)", "Sex(0=female,1=male)",
                           "Age", "SibSp", "Parch", "Fare"]
            }), 400

        scaler, model = load_model()
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)

        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0].tolist()

        return jsonify({
            "survived": bool(prediction),
            "survival_label": "Survived ✅" if prediction == 1 else "Did not survive ❌",
            "confidence": {
                "not_survived": round(probabilities[0], 3),
                "survived": round(probabilities[1], 3)
            },
            "input_received": {
                "Pclass": features[0],
                "Sex": "female" if features[1] == 0 else "male",
                "Age": features[2],
                "SibSp": features[3],
                "Parch": features[4],
                "Fare": features[5]
            }
        })

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500