"""Credit Card Fraud Detection - Logistic Regression with under-sampling.

Run from the project root:
    python main.py
    python main.py --data data/creditcard_data.csv
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "creditcard_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_model.joblib"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        sys.exit(
            f"\nDataset not found at: {path}\n"
            "Paste your 'creditcard_data.csv' into the 'data/' folder "
            "(or pass --data <path>) and run again.\n"
        )
    return pd.read_csv(path)


def explore(df: pd.DataFrame) -> None:
    print("First 5 rows:")
    print(df.head(), "\n")
    print("Dataset info:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum(), "\n")
    print("Class distribution (0 = normal, 1 = fraud):")
    print(df["Class"].value_counts(), "\n")

    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    print("Legit shape:", legit.shape, "| Fraud shape:", fraud.shape, "\n")
    print("Legit Amount stats:")
    print(legit.Amount.describe(), "\n")
    print("Fraud Amount stats:")
    print(fraud.Amount.describe(), "\n")
    print("Mean values by class:")
    print(df.groupby("Class").mean(), "\n")


def undersample(df: pd.DataFrame, random_state: int = 2) -> pd.DataFrame:
    """Balance the dataset: sample as many legit rows as there are fraud rows."""
    legit = df[df.Class == 0]
    fraud = df[df.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=random_state)
    return pd.concat([legit_sample, fraud], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Credit card fraud detection")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH,
                        help="Path to creditcard_data.csv")
    parser.add_argument("--max-iter", type=int, default=1000,
                        help="Max iterations for Logistic Regression")
    args = parser.parse_args()

    df = load_data(args.data)
    explore(df)

    # Under-sampling to fix the heavy class imbalance
    new_dataset = undersample(df)
    print("Balanced class distribution:")
    print(new_dataset["Class"].value_counts(), "\n")

    # Features / target split
    X = new_dataset.drop(columns="Class")
    Y = new_dataset["Class"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, stratify=Y, random_state=2
    )
    print("Shapes (all, train, test):", X.shape, X_train.shape, X_test.shape, "\n")

    # Train
    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train, Y_train)

    # Evaluate
    train_acc = accuracy_score(Y_train, model.predict(X_train))
    test_pred = model.predict(X_test)
    test_acc = accuracy_score(Y_test, test_pred)
    print(f"Accuracy on Training data : {train_acc:.4f}")
    print(f"Accuracy on Test data     : {test_acc:.4f}\n")
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(confusion_matrix(Y_test, test_pred), "\n")
    print("Classification report:")
    print(classification_report(Y_test, test_pred, target_names=["Normal", "Fraud"]))

    # Save the trained model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
