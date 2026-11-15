# Credit Card Fraud Detection

This project trains a **Logistic Regression** model to classify credit-card
transactions as normal or fraudulent. It demonstrates data exploration,
class-imbalance handling, model evaluation, and saving a trained model.

> **Note:** This is an educational example, not a production fraud-detection
> system. The model is trained and evaluated on an under-sampled dataset, so
> its metrics do not represent performance on the original transaction
> distribution.

## What the project does

1. Loads and summarizes the transaction dataset.
2. Balances the classes by randomly under-sampling normal transactions.
3. Splits the balanced data into stratified training and test sets.
4. Trains Logistic Regression and reports accuracy, a confusion matrix, and a
   classification report.
5. Saves the trained model to `models/fraud_model.joblib`.

## Requirements

- Python 3.10 or newer
- The transaction CSV described below

## Setup

Open this folder in VS Code, then open a terminal at the project root.

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell activation is restricted, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset

The dataset is not included in this repository. Obtain a copy of the
credit-card fraud dataset separately and place it at:

```text
data/creditcard_data.csv
```

The CSV must contain a `Class` column (`0` for normal transactions, `1` for
fraud). The commonly used dataset also includes `Time`, `Amount`, and
anonymized `V1` through `V28` feature columns. The file is intentionally
excluded from Git because it is large.

To use a CSV at another location:

```powershell
python main.py --data "C:\path\to\creditcard_data.csv"
```

## Run

From the project root, with the virtual environment active:

```bash
python main.py
```

Optional: set a different maximum number of Logistic Regression iterations:

```bash
python main.py --max-iter 1500
```

The script prints data summaries and evaluation metrics, then writes the
trained model to `models/fraud_model.joblib`. The generated model file is
ignored by Git.

## Notebook

The exploratory notebook is
[`notebooks/Project_10_Credit_Card_Fraud_Detection.ipynb`](notebooks/Project_10_Credit_Card_Fraud_Detection.ipynb).
Open it in VS Code, select the project's `.venv` Python environment as the
kernel, and make sure the dataset is in `data/`.

## Project layout

```text
credit-card-fraud-detection/
├── data/                 # Add creditcard_data.csv here (not tracked by Git)
├── models/               # Trained model output (not tracked by Git)
├── notebooks/            # Exploratory notebook
├── main.py               # Training and evaluation pipeline
├── requirements.txt      # Python dependencies
└── README.md
```
