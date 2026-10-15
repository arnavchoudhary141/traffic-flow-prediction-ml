import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from data_utils import load_dataset, FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

def main():
    parser = argparse.ArgumentParser(description="Train traffic congestion classifiers.")
    parser.add_argument("--data", default=str(ROOT / "data" / "traffic_demo.csv"),
                        help="CSV with road_id,day_of_week,hour,congestion_level")
    args = parser.parse_args()
    df = load_dataset(args.data)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    models = {
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(8,), activation="relu", solver="adam",
            learning_rate_init=0.01, max_iter=1200, random_state=42,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30
        ),
        "Random Forest baseline": RandomForestClassifier(
            n_estimators=250, max_depth=10, class_weight="balanced",
            random_state=42
        )
    }
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0))
        })
        fitted[name] = model
        report = classification_report(y_test, pred, zero_division=0)
        (OUT / (name.lower().replace(" ", "_") + "_report.txt")).write_text(report)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, pred, labels=[0,1,2,3,4], display_labels=[0,1,2,3,4],
            cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(f"Confusion Matrix — {name}")
        fig.tight_layout()
        fig.savefig(OUT / (name.lower().replace(" ", "_") + "_confusion.png"), dpi=160)
        plt.close(fig)

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print("\nEvaluation on held-out test split")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    results.to_csv(OUT / "model_comparison.csv", index=False)
    best_name = results.iloc[0]["model"]
    joblib.dump({"model": fitted[best_name], "features": FEATURES,
                 "model_name": best_name, "classes": [0,1,2,3,4]},
                OUT / "model.joblib")
    metadata = {
        "data_file": str(Path(args.data).name),
        "rows_after_cleaning": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "results": rows,
        "selected_model": best_name,
        "note": "If using traffic_demo.csv, results are synthetic-demo results only."
    }
    (OUT / "run_summary.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nSelected model: {best_name}")
    print(f"Saved model, comparison, reports, and confusion matrices in: {OUT}")

if __name__ == "__main__":
    main()
