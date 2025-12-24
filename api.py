from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import uvicorn
from predict.predict import predict

app = FastAPI(title="Credit Risk Predictor", version="1.0")

# Define input schema
class Applicant(BaseModel):
    loan_amnt: float
    term: str
    installment: float              
    purpose: str
    issue_d: str             
    emp_length: str
    home_ownership: str
    annual_inc: float
    verification_status: str
    zip_code: str      
    addr_state: str      
    dti: float
    revol_bal: float
    revol_util: float
    earliest_cr_line: str
    fico_range_low: float
    fico_range_high: float
    inq_last_6mths: float
    open_acc: float
    total_acc: float
    mort_acc: float
    delinq_2yrs: float
    pub_rec: float
    pub_rec_bankruptcies: float
    mths_since_last_delinq: float # Added (Use 999.0 if none)

# NEW ROOT ENDPOINT
@app.get("/")
def read_root():
    return RedirectResponse(url="/docs")

# Define the Endpoint
@app.post("/predict_risk")
def get_prediction(applicant: Applicant):
    try:
        # Convert Pydantic object to dictionary
        data_dict = applicant.model_dump()
        # Call backend
        prediction_class, probability = predict(data_dict)
        
        return {
            "risk_class": int(prediction_class),
            "probability_of_default": float(probability),
            "decision": "REJECT" if prediction_class == 1 else "APPROVE"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)