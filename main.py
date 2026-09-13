from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI(title="AI Smart Traffic Backend API", version="1.0")

class TrafficData(BaseModel):
    intersection_id: str
    vehicle_count: int
    average_speed: float

@app.get("/")
def home():
    return {"status": "AI Smart Traffic Backend is Online", "system": "Operational"}

@app.post("/predict-traffic")
def predict_traffic(data: TrafficData):
    # Dynamic 10/20/30 min prediction based on input
    current = data.vehicle_count
    pred_10 = int(current * 1.25)
    pred_20 = int(current * 1.65)
    pred_30 = int(current * 2.15)
    
    co2_current = round(current * 0.012, 2)
    co2_predicted = round(pred_30 * 0.012, 2)
    
    return {
        "intersection_id": data.intersection_id,
        "current_vehicles": current,
        "predictions": {
            "plus_10_min": pred_10,
            "plus_20_min": pred_20,
            "plus_30_min": pred_30
        },
        "congestion_forecast": "VERY HIGH" if pred_30 > 500 else "HIGH",
        "co2_estimation_kg_min": {
            "current": co2_current,
            "predicted_peak": co2_predicted
        },
        "adaptive_signal_recommendation": {
            "north": "45 sec",
            "south": "30 sec",
            "east": "60 sec (High Priority)",
            "west": "25 sec"
        }
    }