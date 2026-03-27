from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
import numpy as np

# v2 predictor (loads model + label encoder once)
from predict_v2 import predict_data


app = FastAPI(
    title="Flower Classification API",
    version="v2",
)

MODEL_VERSION = "v2"

class FlowerData(BaseModel):
    """
    Request body schema for /predict.
    Units are centimeters.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    family: str
    species: str
    confidence: float
    probabilities: dict
    model_version: str

def species_to_family(species: str) -> str:
    """
    Derive family from species (rule-based, not learned).
    """
    return "lily" if "lily" in species.lower() else "iris"


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {
        "status": "healthy",
        "model_version": MODEL_VERSION,
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_flower(features: FlowerData):
    try:
        X = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width,
        ]])

        result = predict_data(X)

        species = result["species"]

        return PredictionResponse(
            family=species_to_family(species),
            species=species,
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            model_version=MODEL_VERSION,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference error: {str(e)}"
        )
