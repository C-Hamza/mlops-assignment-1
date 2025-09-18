# src/train_compare.py
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

def load_data():
    data = load_iris()
    X, y = data.data, data.target
    target_names = data.target_names
    return X, y, target_names

def build_models(random_state=42):
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

def evaluate_and_save(models, X_train, X_test, y_train, y_test):
    rows = []
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"{name} -> accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}")

        # save model
        model_path = os.path.join("models", f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        # save classification report
        report = classification_report(y_test, y_pred, digits=4)
        with open(os.path.join("results", f"{name}_classification_report.txt"), "w") as f:
            f.write(report)

        rows.append({
            "model": name,
            "accuracy": acc,
            "precision_macro": prec,
            "recall_macro": rec,
            "f1_macro": f1
        })

    metrics_df = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    metrics_df.to_csv(os.path.join("results", "metrics.csv"), index=False)
    print("\nSaved metrics to results/metrics.csv")
    return metrics_df

def plot_metrics(metrics_df):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    ax = metrics_df.set_index("model")[metrics].plot.bar(rot=0, figsize=(10,5))
    plt.title("Model comparison")
    plt.tight_layout()
    plt.legend(loc="lower right")
    plt.savefig(os.path.join("results", "metrics_comparison.png"))
    print("Saved plot to results/metrics_comparison.png")
    plt.close()

def main():
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = build_models(random_state=42)
    metrics_df = evaluate_and_save(models, X_train, X_test, y_train, y_test)
    print("\nMetrics summary:\n", metrics_df)
    plot_metrics(metrics_df)

    # Save best model explicitly
    best_model_name = metrics_df.iloc[0]["model"]
    best_model = models[best_model_name]
    joblib.dump(best_model, os.path.join("models", "best_model.joblib"))
    print(f"Best model: {best_model_name} saved as models/best_model.joblib")

if __name__ == "__main__":
    main()
