from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "outputs" / "model.joblib"
LABELS = {
    0: "Very low / clear", 1: "Low", 2: "Moderate",
    3: "High", 4: "Very high / severe"
}
st.set_page_config(page_title="Traffic Congestion Predictor", page_icon="🚦")
st.title("🚦 Urban Traffic Congestion Predictor")
st.write("Predict a congestion class from road ID, day of week, and hour.")
st.warning("Predictions are only as reliable as the training data. The bundled demo dataset is synthetic.")
if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python src/train.py` in the project folder first.")
    st.stop()
bundle = joblib.load(MODEL_PATH)
road_id = st.number_input("Road ID", min_value=0, value=3, step=1)
day = st.selectbox("Day of week", options=list(range(7)),
                   format_func=lambda x: ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"][x])
hour = st.slider("Hour of day", 0, 23, 17)
if st.button("Predict congestion", type="primary"):
    sample = pd.DataFrame([{"road_id": int(road_id), "day_of_week": day, "hour": hour}])
    model = bundle["model"]
    pred = int(model.predict(sample[bundle["features"]])[0])
    st.subheader(f"Level {pred}: {LABELS[pred]}")
    if hasattr(model, "predict_proba"):
        probs = pd.DataFrame({"Congestion level": model.classes_,
                              "Probability": model.predict_proba(sample[bundle["features"]])[0]})
        st.bar_chart(probs.set_index("Congestion level"))
st.caption(f"Model loaded: {bundle['model_name']}")
