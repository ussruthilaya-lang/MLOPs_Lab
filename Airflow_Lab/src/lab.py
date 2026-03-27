import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from kneed import KneeLocator
import pickle
import os
import base64
from datetime import datetime

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "../data/file.csv")
    df = pd.read_csv(data_path)

    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")

def data_preprocessing(data_b64: str):
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    df = df.dropna()
    clustering_data = df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    bundle = {
        "scaled_data": scaled_data,
        "scaler": scaler
    }

    serialized_bundle = pickle.dumps(bundle)
    return base64.b64encode(serialized_bundle).decode("ascii")


def build_save_model(data_b64: str):
    data_bytes = base64.b64decode(data_b64)
    bundle = pickle.loads(data_bytes)

    scaled_data = bundle["scaled_data"]
    scaler = bundle["scaler"]

    model_type = "minibatch"  # I have used minibatch model here instead of k-means as a part of the customisation.

    kmeans_kwargs = {
        "init": "k-means++",
        "n_init": 3,
        "max_iter": 100,
        "random_state": 42
    }

    sse = []
    models = []

    for k in range(1, 20):
        if model_type == "minibatch":
            model = MiniBatchKMeans(
                n_clusters=k,
                batch_size=1024,
                **kmeans_kwargs
            )
        else:
            model = KMeans(n_clusters=k, **kmeans_kwargs) #we have used k-means as a baseline model.

        model.fit(scaled_data)
        sse.append(model.inertia_)
        models.append(model)

    kl = KneeLocator(
        range(1, len(sse) + 1),
        sse,
        curve="convex",
        direction="decreasing"
    )

    optimal_k = kl.elbow
    if optimal_k is None:
        optimal_k = 3

    final_model = models[int(optimal_k) - 1]

    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S") #here I have customised to keep a version control

    base_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model"
    )

    version_dir = os.path.join(base_model_dir, timestamp)
    os.makedirs(version_dir, exist_ok=True)

    model_path = os.path.join(version_dir, "model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": final_model,
            "scaler": scaler,
            "optimal_k": int(optimal_k),
            "model_type": model_type
        }, f)

    print(f"Model type used: {model_type}")
    print(f"Model version saved at: {model_path}")
    print(f"Optimal k selected: {optimal_k}")

    return timestamp  # version ID


def load_model_elbow(version_id: str):
    base_model_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "model"
    )

    model_path = os.path.join(base_model_dir, version_id, "model.pkl")

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    scaler = bundle["scaler"]

    test_path = os.path.join(
        os.path.dirname(__file__),
        "../data/test.csv"
    )

    test_df = pd.read_csv(test_path)
    test_features = test_df[["BALANCE", "PURCHASES", "CREDIT_LIMIT"]]
    test_scaled = scaler.transform(test_features)

    prediction = model.predict(test_scaled)

    return int(prediction[0])