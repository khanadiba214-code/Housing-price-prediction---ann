import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "ann_pipeline.joblib"

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run: python run_models.py")

    model = joblib.load(MODEL_PATH)
    sample = pd.DataFrame([{
        "city": "Bengaluru",
        "locality_tier": "Mid",
        "property_type": "Apartment",
        "area_sqft": 1250,
        "bedrooms": 3,
        "bathrooms": 2,
        "balconies": 2,
        "age_years": 5,
        "floor": 7,
        "total_floors": 15,
        "furnishing": "Semi-Furnished",
        "parking": 1,
        "facing": "East",
        "near_metro": "Yes",
        "lift": "Yes"
    }])

    predicted_lakh = float(model.predict(sample)[0])
    predicted_rupees = predicted_lakh * 100000
    print(f"Estimated housing price: ₹{predicted_rupees:,.0f}")
    print("Note: This is an academic estimate based on simulated training data.")

if __name__ == "__main__":
    main()
