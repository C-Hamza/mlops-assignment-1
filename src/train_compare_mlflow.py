# src/train_compare_mlflow.py
import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------
# Configuration
# -------------------------
EXPERIMENT_NAME = "mlops-assignment-1"
RESULTS_DIR = "results"
MODELS_DIR = "models"
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# make sure MLflow writes to local ./mlruns (default)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment(EXPERIMENT_NAME)

def load_data():
    data = load_iris()
    X, y = data.data, data.target
    return X, y, data.target_names

def build_models(random_state=RANDOM_STATE):
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "RandomForest": Pipeline([
            ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, random_state=random_state))
        ])
    }
    return models

def plot_confusion_matrix(cm, labels, out_path, title="Confusion matrix"):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.imshow(cm, interpolation="nearest")
    ax.set_title(title)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def run():
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = build_models()
    rows = []

    for name, model in models.items():
        # prepare params to log (extract classifier from pipeline)
        clf = model.named_steps.get("clf")
        params = {}
        if clf is not None:
            # pick a few useful hyperparameters
            p = clf.get_params()
            # safe picks (not all keys)
            for key in ["n_estimators", "max_depth", "C", "kernel", "penalty", "max_iter", "probability"]:
                if key in p:
                    params[key] = p[key]

        # Start MLflow run
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            # train
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            metrics = {
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1
            }
            mlflow.log_metrics(metrics)

            # Save local model (optional)
            local_model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
            joblib.dump(model, local_model_path)
            # log model artifact via MLflow (recommended)
            mlflow.sklearn.log_model(model, artifact_path="model")

            # classification report
            report = classification_report(y_test, y_pred, digits=4)
            report_path = os.path.join(RESULTS_DIR, f"{name}_classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)

            # confusion matrix plot
            cm = confusion_matrix(y_test, y_pred)
            cm_path = os.path.join(RESULTS_DIR, f"{name}_confusion_matrix.png")
            plot_confusion_matrix(cm, target_names, cm_path, title=f"Confusion Matrix - {name}")
            mlflow.log_artifact(cm_path)

            # append row for summary table
            rows.append({
                "model": name,
                **metrics
            })

            print(f"[{name}] metrics: {metrics}")
            print(f"[{name}] saved local model: {local_model_path}")
            print(f"[{name}] logged model + artifacts to MLflow")

    # save consolidated metrics table and log it
    metrics_df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    metrics_csv = os.path.join(RESULTS_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    mlflow.log_artifact(metrics_csv)
    print("Saved metrics summary to", metrics_csv)

    # save best model explicitly (local copy)
    best_name = metrics_df.iloc[0]["model"]
    best_local = os.path.join(MODELS_DIR, f"{best_name}_best.joblib")
    joblib.dump(models[best_name], best_local)
    mlflow.log_artifact(best_local)
    print("Best model (local) saved to", best_local)

if __name__ == "__main__":
    run()
