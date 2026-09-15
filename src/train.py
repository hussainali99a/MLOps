import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.logger import get_logger


logger = get_logger(__name__)


def load_data(data_path, target_column):
    logger.info("Loading dataset from %s", data_path)

    df = pd.read_csv(data_path)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    logger.info("Dataset loaded successfully")
    logger.info("Dataset shape: %s", df.shape)

    return X, y


def train_model(X_train, y_train, model_config):
    logger.info("Starting model training")

    model = RandomForestClassifier(
        n_estimators=model_config["n_estimators"],
        random_state=model_config["random_state"]
    )

    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model


def evaluate_model(model, X_test, y_test):
    logger.info("Evaluating model")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    logger.info("Model accuracy: %.4f", accuracy)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return {
        "accuracy": accuracy
    }


def save_model(model, model_path):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)

    logger.info("Model saved to %s", model_path)


def save_metrics(metrics, metrics_path):
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Metrics saved to %s", metrics_path)


def main():

    logger.info("========== TRAINING PIPELINE STARTED ==========")

    config = load_config()

    data_config = config["data"]
    training_config = config["training"]
    model_config = config["model"]
    artifacts_config = config["artifacts"]

    X, y = load_data(
        data_config["path"],
        data_config["target_column"]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=training_config["test_size"],
        random_state=training_config["random_state"],
        stratify=y
    )

    logger.info(
        "Training samples: %d",
        len(X_train)
    )

    logger.info(
        "Testing samples: %d",
        len(X_test)
    )

    model = train_model(
        X_train,
        y_train,
        model_config
    )

    metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_model(
        model,
        artifacts_config["model_path"]
    )

    save_metrics(
        metrics,
        artifacts_config["metrics_path"]
    )

    logger.info("========== TRAINING PIPELINE COMPLETED ==========")


if __name__ == "__main__":
    main()