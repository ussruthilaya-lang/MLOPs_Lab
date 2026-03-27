# What I Learned from the Base Lab

- How to load a dataset, split it into training and testing sets, and train a basic ML model.

- How to expose a model through a FastAPI endpoint.

- How request handling and routing works in FastAPI.

These steps established the basic end-to-end flow from data → model → API.

# Improvements I Made & Why
## Model Training (train_v2.py)

- Constrained the Decision Tree (max_depth=3) to avoid overfitting on a small dataset.

- Fixed random_state to ensure reproducible training.

- Used absolute file paths (Path(__file__)) when saving the model to avoid path-related runtime errors.

Why:
Keeps training reproducible, explainable, and robust to different execution contexts.

## Prediction Logic (predict_v2.py)

- Loaded the model once at module level instead of reloading it per request.

- Added prediction confidence using predict_proba alongside the predicted class.

Why:
Improves runtime efficiency and exposes model uncertainty, which is critical in real ML systems.

## API Layer (main.py)

- Introduced explicit request and response schemas using Pydantic.

- Returned structured responses including prediction, confidence, model version, and input echo.

- Loaded the model once at application startup.

Why:
Creates a clear API contract, improves input validation, and makes responses more interpretable and debuggable.

## Summary

While the dataset and model are simple, the changes focus on good ML system design: separating training from serving, avoiding unnecessary computation, and exposing model confidence. These improvements make the lab closer to how ML APIs are built and maintained in practice.
