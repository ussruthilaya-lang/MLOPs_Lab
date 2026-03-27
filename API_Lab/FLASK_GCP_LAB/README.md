# Flower Species Classification
## Learnings from this project

While building and deploying this project, I learned the following key points:

- Local environments and cloud containers are completely separate. If a dependency works locally, it does not mean it exists inside the Docker container unless it is explicitly listed in requirements.txt.

- FastAPI applications must be run using an ASGI server such as uvicorn. Running the app with python main.py works locally in some cases but fails in production environments like Cloud Run.

- Cloud Run requires the container to listen on the port provided via the PORT environment variable. Hardcoding ports or using framework-specific flags causes deployment failures.

- Imports that work locally can fail inside Docker if the project structure is not treated as a proper Python package. Using absolute, package-based imports and adding __init__.py files is necessary.

- Separating the backend and frontend makes the system easier to deploy, debug, and scale.

## Changes made based on these learnings

Based on the above learnings, I made the following changes to the project:

- Converted the backend to a clean FastAPI service that only handles model loading and inference.

- Ensured all dependencies such as fastapi, uvicorn, scikit-learn, and joblib are explicitly listed in requirements.txt so the container environment is reproducible.

- Updated the Dockerfile to run the application using uvicorn and to bind to the port provided by Cloud Run through the PORT environment variable.

- Fixed all imports to use package-based paths (for example from src.predict_v2 import predict_data) and added __init__.py so the src directory is treated as a package.

- Kept the Streamlit dashboard as a thin client that only calls the backend API instead of loading models directly.

- Deployed the backend and frontend as separate services to avoid coupling UI logic with inference logic.

These changes were made to ensure the application runs reliably in a cloud environment and follows standard MLOps and deployment practices.

## How to run (Reference for TA)
### Prerequisites

- Google Cloud SDK installed and authenticated

- A GCP project with Cloud Build and Cloud Run APIs enabled

- Docker installed (optional for local testing)

Replace [PROJECT_ID] with the actual project ID.

### Deploying the backend (FastAPI)

This deploys the trained flower classification model as a web API.

From the backend project directory (the folder containing the Dockerfile):

gcloud builds submit --tag gcr.io/[PROJECT_ID]/flower-api

After the build succeeds, deploy to Cloud Run:

gcloud run deploy flower-api --image gcr.io/[PROJECT_ID]/flower-api --platform managed --region us-central1 --allow-unauthenticated

After deployment, copy the service URL shown in the output. This URL exposes:

/ for health check

/docs for Swagger UI

/predict for inference

### Deploying the dashboard (Streamlit)

The dashboard is a frontend that calls the backend API.

Open Dashboard.py

Update the backend URL reference:

FASTAPI_BACKEND_ENDPOINT = os.getenv("FASTAPI_URL", "https://flower-api-xyz.a.run.app")

Build and deploy the dashboard:

gcloud builds submit --config cloudbuild_dashboard.yaml .
gcloud run deploy flower-dashboard --image gcr.io/[PROJECT_ID]/flower-dashboard --platform managed --region us-central1 --allow-unauthenticated --port 8080

Open the dashboard URL shown after deployment to use the application.
