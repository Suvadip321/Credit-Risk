# Credit Risk Prediction

This project predicts whether a loan applicant is likely to default based on their financial profile. It's trained on LendingClub data and uses XGBoost under the hood.

## The Problem

Lenders need to decide quickly whether to approve or reject loan applications. This model takes in applicant details—income, credit score, existing debt, employment history—and returns a risk score along with a recommendation.

## How It Works

The pipeline does three things:

1. **Cleans the data** — Handles missing values, parses weird formats like "36 months" or "10+ years", and removes columns that would leak future information (like repayment history, which you obviously don't have at application time).

2. **Engineers features** — Creates useful signals like credit history length, payment-to-income ratio, and average FICO score.

3. **Predicts risk** — An XGBoost classifier trained with time-based cross-validation. The model accounts for class imbalance (most loans get repaid, defaults are rarer).

## Running It

Install dependencies:
```bash
pip install -r requirements.txt
```

Train the model:
```bash
cd src
python train_model.py
```

Start the API:
```bash
uvicorn api:app --reload
```

Make a prediction:
```bash
curl -X POST "http://localhost:8000/predict_risk" \
  -H "Content-Type: application/json" \
  -d '{"loan_amnt": 15000, "annual_inc": 85000, "dti": 18.5, "fico_range_low": 710}'
```

## Tests

```bash
py -m pytest tests/ -v
```

## Why These Design Choices?

- **Time-based split**: Train on older loans, test on newer ones. This mirrors real-world usage and avoids data leakage.
- **Custom transformers**: All preprocessing is sklearn-compatible and gets saved with the model, so inference matches training exactly.
- **Class weighting**: Defaults are rare (~20%), so the model is weighted to pay more attention to them.
