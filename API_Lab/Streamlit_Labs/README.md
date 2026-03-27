# Streamlit Dashboard
## What I did

I built a Streamlit dashboard to interact with the flower classification model.
The dashboard does not load the model or do any prediction on its own. It only talks to the FastAPI backend.

All ML logic lives in the backend.

### Why I did it this way

While deploying to Cloud Run, I realized keeping model logic in the UI causes unnecessary issues. So I made Streamlit a thin layer that just sends inputs and shows results.

### How it works

User inputs the four flower features using sliders.

The app checks if the backend is running.

On clicking Predict, the input is sent to the FastAPI /predict endpoint.

The backend returns species, family, confidence, and probabilities.

Streamlit just displays this data along with images.

### How to run locally

Start the FastAPI backend:

uvicorn main:app --reload

Runs at http://127.0.0.1:8000

Start Streamlit:

streamlit run Dashboard.py


Open http://localhost:8501

Backend URL

The backend URL is configurable. By default it points to localhost:

FASTAPI_BACKEND_ENDPOINT = os.getenv(
    "FASTAPI_URL",
    "http://127.0.0.1:8000"
)


When deployed, this is replaced with the Cloud Run URL.
