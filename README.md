# SpendSense AI 💸

A GitHub-ready AI expense categorizer for **educational use**. Classifies transaction descriptions into Food, Transport, Shopping, Bills, Entertainment, Healthcare, Travel, and Education. Runs locally with no paid API keys.

## Features
- TF-IDF + Logistic Regression text classifier
- Streamlit single-transaction form and CSV upload
- Category spending totals visualization and categorized CSV export
- FastAPI `/predict` and `/health` endpoints
- Included synthetic dataset and pre-trained model
- Pytest tests and GitHub Actions CI

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

Visit http://localhost:8501

### REST API

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs. Example JSON:

```json
{"descriptions": ["Uber ride to airport", "Netflix subscription"]}
```

### Retrain

```powershell
.venv\Scripts\python.exe -m src.train
```

### Tests

```powershell
.venv\Scripts\python.exe -m pytest -q
```

### Batch CSV format

```csv
description,amount
Uber ride to downtown,12.50
Monthly internet bill,44.99
```

## How it works
TF-IDF extracts word and phrase features; Logistic Regression estimates a probability for each spending category. The highest-scoring category is returned. Predictions below 65% confidence get a manual-review flag. Confidence scores are model estimates, **not guaranteed correctness**.

## Limitations and privacy
Training data is synthetic and intentionally repetitive; no external real-world accuracy is claimed. Transactions are processed locally. Do not upload confidential financial data to publicly deployed instances without authentication, encryption, and a suitable data handling policy. Only load model files from trusted sources (joblib uses pickle).

## GitHub
Create a repository named `SpendSense-AI`, commit this folder, and push. Includes a permissive MIT license and a GitHub Actions test workflow.
