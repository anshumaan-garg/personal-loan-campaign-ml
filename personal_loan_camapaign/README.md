# Personal Loan Campaign Prediction

Production-ready Machine Learning pipeline for predicting whether a customer will accept a personal loan offer.

---

## Features

- Decision Tree Classifier
- Optuna Hyperparameter Optimization
- Weights & Biases Experiment Tracking
- FastAPI Inference API
- Ray Serve Deployment
- Joblib Model Serialization

---

## Project Structure

```text
personal_loan_campaign/
│
├── data/
├── models/
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
python -m venv venv

source venv/bin/activate

pip install -r requirements.txt
```

---

## Login to W&B

```bash
wandb login
```

---

## Data Preprocessing

```bash
python src/preprocess.py
```

---

## Model Training

```bash
python src/train.py
```

---

## Evaluation

```bash
python src/evaluate.py
```

---

## Serve Model

```bash
serve run src.serve_api:deployment
```

Open

```
http://127.0.0.1:8000/docs
```

for Swagger UI.

---

## Example Prediction

```json
POST /predict

{
  "Age":45,
  "Experience":20,
  "Income":80,
  "ZIP_Code":91107,
  "Family":3,
  "CCAvg":2.5,
  "Education":2,
  "Mortgage":100,
  "Securities_Account":0,
  "CD_Account":0,
  "Online":1,
  "CreditCard":1
}
```

Response

```json
{
  "prediction": 1,
  "probability": 0.94
}
```