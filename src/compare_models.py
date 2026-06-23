"""
src/compare_models.py

After training all 4 backbones, run this script to pull their MLflow
metrics into a single comparison table and bar chart. Useful for the
README screenshots and the final report's "Results" section.

Usage:
    python src/compare_models.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    mlflow_cfg = config["mlflow"]
    paths_cfg = config["paths"]

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(mlflow_cfg["experiment_name"])
    if experiment is None:
        raise RuntimeError(
            f"No MLflow experiment named '{mlflow_cfg['experiment_name']}' found. "
            f"Train at least one model first."
        )

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        raise RuntimeError("No MLflow runs found. Train at least one model first.")

    rows = []
    for run in runs:
        params = run.data.params
        metrics = run.data.metrics
        rows.append({
            "model_name": params.get("model_name", run.info.run_name),
            "test_accuracy": metrics.get("test_accuracy"),
            "test_loss": metrics.get("test_loss"),
            "val_accuracy": metrics.get("val_accuracy"),
            "val_loss": metrics.get("val_loss"),
            "epochs": params.get("epochs"),
            "learning_rate": params.get("learning_rate"),
            "run_id": run.info.run_id,
        })

    df = pd.DataFrame(rows).sort_values("test_accuracy", ascending=False).reset_index(drop=True)
    print("\n=== Model Comparison (by test accuracy) ===\n")
    print(df.to_string(index=False))

    artifacts_dir = Path(paths_cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv_path = artifacts_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved comparison table -> {csv_path}")

    # Bar chart of test accuracy across models
    plot_df = df.dropna(subset=["test_accuracy"])
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(plot_df["model_name"], plot_df["test_accuracy"], color="#1f5fb5")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("Test Accuracy by Backbone Architecture")
        ax.set_ylim(0, 1.0)
        for i, v in enumerate(plot_df["test_accuracy"]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
        fig.tight_layout()

        chart_path = artifacts_dir / "model_comparison_chart.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"Saved comparison chart -> {chart_path}")


if __name__ == "__main__":
    main()
