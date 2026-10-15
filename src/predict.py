import argparse
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    0: "Very low / clear",
    1: "Low",
    2: "Moderate",
    3: "High",
    4: "Very high / severe"
}

def main():
    parser = argparse.ArgumentParser(description="Predict traffic congestion.")
    parser.add_argument("--road-id", type=int, required=True)
    parser.add_argument("--day", type=int, required=True, help="Monday=0 ... Sunday=6")
    parser.add_argument("--hour", type=int, required=True, help="0 ... 23")
    args = parser.parse_args()
    if not 0 <= args.day <= 6 or not 0 <= args.hour <= 23:
        parser.error("--day must be 0..6 and --hour must be 0..23")
    path = ROOT / "outputs" / "model.joblib"
    if not path.exists():
        parser.error("Model not found. Run `python src/train.py` first.")
    bundle = joblib.load(path)
    sample = pd.DataFrame([{
        "road_id": args.road_id, "day_of_week": args.day, "hour": args.hour
    }])
    model = bundle["model"]
    prediction = int(model.predict(sample[bundle["features"]])[0])
    print(f"Model: {bundle['model_name']}")
    print(f"Predicted congestion level: {prediction} — {LABELS.get(prediction, 'Unknown')}")
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(sample[bundle["features"]])[0]
        print("Class probabilities:")
        for cls, prob in zip(model.classes_, probabilities):
            print(f"  Level {int(cls)}: {prob:.1%}")

if __name__ == "__main__":
    main()
