# Machine Learning for Traffic Flow Prediction in Urban Areas

A mini-project implementing traffic congestion-level prediction as a 5-class machine-learning classification task.

## Problem
Predict congestion level for a road at a particular day and hour. Labels follow the reference paper's scale:
- `0`: very low / clear traffic
- `1`: low
- `2`: moderate
- `3`: high
- `4`: very high / severe congestion

**Important data note:** the supplied research paper describes an OutScraper-derived dataset, but the raw dataset was not included with the assignment files. `data/traffic_demo.csv` is a clearly labelled synthetic demonstration dataset, not real traffic observations. Replace it with an authorized real dataset before reporting real-world performance.

## Features
- Road identifier (`road_id`)
- Day of week (`day_of_week`, Monday=0 ... Sunday=6)
- Hour of day (`hour`, 0 ... 23)

Target: `congestion_level` (0–4).

## Setup (macOS)
```bash
cd traffic_flow_ml_project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate
```bash
python src/train.py
```
This trains a neural-network model and a Random Forest baseline, prints evaluation metrics, and saves model/plots/reports to `outputs/`.

## Predict a new observation
After training:
```bash
python src/predict.py --road-id 3 --day 4 --hour 17
```
`day` is Monday=0 through Sunday=6. `hour` is 0 through 23.

## Optional interactive dashboard
```bash
streamlit run src/app.py
```
Train first so `outputs/model.joblib` exists.

## Use a real dataset
Place a CSV at `data/traffic.csv` with these columns:
```text
road_id,day_of_week,hour,congestion_level
```
The target must be integers 0–4. Then run:
```bash
python src/train.py --data data/traffic.csv
```
Do not mix the synthetic demo data into a real-data evaluation.

## Project structure
```text
traffic_flow_ml_project/
├── data/traffic_demo.csv
├── src/
│   ├── data_utils.py
│   ├── train.py
│   ├── predict.py
│   └── app.py
├── outputs/                 # generated after training
├── requirements.txt
├── report.pdf
├── presentation.pptx
└── README.md
```

## Method
1. Validate and clean columns.
2. Split data into train/test sets with stratification.
3. Train an MLP neural network and a Random Forest baseline.
4. Evaluate using accuracy, macro-F1, confusion matrix, and classification report.
5. Save the selected model and preprocessing pipeline.
6. Predict the congestion class for a road/day/hour input.

## Limitations
- The three input features do not directly represent live vehicle counts, weather, incidents, road capacity, or spatial relationships between nearby roads.
- Synthetic demo results are only for checking that the code runs; they are not evidence of real-world accuracy.
- A future version should use timestamped traffic measurements and time-aware validation.
