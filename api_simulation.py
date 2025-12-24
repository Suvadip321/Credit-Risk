import requests
import time

API_URL = "http://127.0.0.1:8000/predict_risk"

applicants = [
    {
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
        "mths_since_last_delinq": 999
    },
    {
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
        "mths_since_last_delinq": 2
    }
]

print(f"Connecting to Credit Risk Engine at {API_URL}...\n")
print("-" * 30)
for i, applicant in enumerate(applicants):
    try:
        start_time = time.time()
        response = requests.post(API_URL, json=applicant)
        latency = (time.time() - start_time) * 1000 # ms
        
        if response.status_code == 200:
            result = response.json()
            risk = result['probability_of_default']
            decision = "REJECT" if risk > 0.40 else "APPROVE" 
            
            print(f"Applicant #{i+1}:")
            print(f"   Status: {decision} (Risk: {risk:.1%})")
            print(f"   Latency: {latency:.2f}ms")
            print("-" * 30)
        else:
            print(f"Applicant #{i+1}: FAILED - {response.text}")
            
    except Exception as e:
        print(f"Connection Error: {e}")

