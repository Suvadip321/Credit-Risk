import warnings
import pandas as pd
import joblib
import numpy as np
import sys
from pathlib import Path
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
sys.path.append(str(BASE_DIR / "src"))

MODEL_PATH = BASE_DIR / "models" / "credit_risk_model.pkl"
DATA_PATH = BASE_DIR / "predict" / "lending_club_template.csv"

if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train the model first.")
        
model = joblib.load(MODEL_PATH)

if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

def predict(applicant_data):
    """Predicts credit risk for a single applicant."""

    template_cols = df.drop(columns=["loan_status"]).columns.tolist()
    
    row_data = {col: applicant_data.get(col, np.nan) for col in template_cols}
    input_df = pd.DataFrame([row_data])

    prediction_class = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return prediction_class, probability

if __name__ == "__main__":
    print("Running prediction for sample applicant...")

    applicant_data = {
        "loan_amnt": 15000.0,
        "term": "36 months",
        "installment": 450.00,
        "purpose": "debt_consolidation",
        "issue_d": "Dec-2023",  
        "emp_length": "10+ years",
        "home_ownership": "MORTGAGE",
        "annual_inc": 85000.0,
        "verification_status": "Source Verified",
        "zip_code": "940xx",    
        "addr_state": "CA",
        "dti": 18.5,         
        "revol_bal": 14000.0,
        "revol_util": 55.0,   
        "earliest_cr_line": "Jan-2005",
        "fico_range_low": 700.0,
        "fico_range_high": 704.0,
        "inq_last_6mths": 1.0,
        "open_acc": 12.0,      
        "total_acc": 24.0,      
        "mort_acc": 1.0,       
        "delinq_2yrs": 0.0,
        "pub_rec": 0.0,     
        "pub_rec_bankruptcies": 0.0,
        "mths_since_last_delinq": 999.0
    }

    try:
        is_default, prob = predict(applicant_data)
        print(f"\nDefault Probability: {prob:.2%}")
        print(f"Decision: {'REJECT (High Risk)' if is_default == 1 else 'APPROVE (Low Risk)'}")
    except Exception as e:
        print(f"\nError: {e}")
