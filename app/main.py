from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

# Load model
model = joblib.load("app/model.pkl")

# Define input schema
class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

@app.post("/predict")
def predict(features: HouseFeatures):
    input_array = np.array([[features.MedInc, features.HouseAge, features.AveRooms,
                             features.AveBedrms, features.Population, features.AveOccup,
                             features.Latitude, features.Longitude]])
    prediction = model.predict(input_array)
    return {"PredictedHouseValue": prediction[0]}
