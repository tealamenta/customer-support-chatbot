"""
Training script with MLflow tracking.

Usage:
    python -m src.training.train
"""

import mlflow
from datasets import load_dataset


def main():
    # Configuration
    config = {
        "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "lora_r": 16,
        "lora_alpha": 32,
        "epochs": 1,
        "batch_size": 4,
        "learning_rate": 2e-4,
    }

    # Start MLflow run
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("customer-support-finetune")

    with mlflow.start_run(run_name="tinyllama-lora"):
        # Log parameters
        mlflow.log_params(config)

        # Log training metrics (from your Colab results)
        mlflow.log_metric("final_loss", 0.538)
        mlflow.log_metric("train_samples", 26872)
        mlflow.log_metric("training_time_minutes", 46)

        # Log step-by-step loss
        losses = [(500, 0.636), (1000, 0.571), (1512, 0.538)]
        for step, loss in losses:
            mlflow.log_metric("loss", loss, step=step)

        print("Metrics logged to MLflow!")
        print("View at: http://localhost:5001")


if __name__ == "__main__":
    main()
