import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from api import app

# Create the Test Client
client = TestClient(app)

# Define Valid Data (A "Good" Applicant)
good_applicant = {
    "loan_amnt": 15000, "term": "36 months", "installment": 450, 
    "int_rate": 14.5, "grade": "C", "sub_grade": "C1", 
    "emp_length": "10+ years", "home_ownership": "MORTGAGE", 
    "annual_inc": 85000, "verification_status": "Source Verified", 
    "issue_d": "Jan-2023", "purpose": "debt_consolidation", 
    "dti": 18.5, "delinq_2yrs": 0, "earliest_cr_line": "Jan-2005", 
    "inq_last_6mths": 1, "open_acc": 12, "pub_rec": 0, 
    "revol_bal": 14000, "revol_util": 55.0, "total_acc": 24, 
    "mort_acc": 1, "pub_rec_bankruptcies": 0, 
    "fico_range_low": 710, "fico_range_high": 714,
    "zip_code": "940xx", "addr_state": "CA", 
    "mths_since_last_delinq": 999.0
}

# Define Risky Data (A "Bad" Applicant)
bad_applicant = {
    "loan_amnt": 35000, "term": "60 months", "installment": 900, 
    "int_rate": 24.5, "grade": "F", "sub_grade": "F1", 
    "emp_length": "< 1 year", "home_ownership": "RENT", 
    "annual_inc": 40000, "verification_status": "Not Verified", 
    "issue_d": "Jan-2023", "purpose": "vacation", 
    "dti": 45.0, "delinq_2yrs": 2, "earliest_cr_line": "Jan-2020", 
    "inq_last_6mths": 5, "open_acc": 5, "pub_rec": 1, 
    "revol_bal": 10000, "revol_util": 95.0, "total_acc": 8, 
    "mort_acc": 0, "pub_rec_bankruptcies": 1, 
    "fico_range_low": 660, "fico_range_high": 664,
    "zip_code": "100xx", "addr_state": "NY", 
    "mths_since_last_delinq": 2.0
}

def test_health_check():
    """Ensure the API is actually running."""
    response = client.get("/")
    # 404 is fine because we didn't define a root endpoint, 
    # but it proves the server is UP.
    assert response.status_code in [200, 404]

def test_predict_endpoint_structure():
    """Check if the API returns the correct JSON keys."""
    response = client.post("/predict_risk", json=good_applicant)
    assert response.status_code == 200
    
    data = response.json()
    assert "risk_class" in data
    assert "probability_of_default" in data
    assert "decision" in data

def test_low_risk_applicant():
    """Ensure a good applicant gets a Low Risk score."""
    response = client.post("/predict_risk", json=good_applicant)
    data = response.json()
    
    assert data["decision"] == "APPROVE"
    assert data["probability_of_default"] < 0.40 

def test_high_risk_applicant():
    """Ensure a bad applicant gets a High Risk score."""
    response = client.post("/predict_risk", json=bad_applicant)
    data = response.json()
    
    assert data["decision"] == "REJECT"
    assert data["probability_of_default"] > 0.60

def test_invalid_input_handling():
    """Ensure the API blocks bad data types."""
    broken_data = good_applicant.copy()
    broken_data["loan_amnt"] = "ONE MILLION DOLLARS"
    
    response = client.post("/predict_risk", json=broken_data)
    
    # 422 = Unprocessable Entity (FastAPI caught the error)
    assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
