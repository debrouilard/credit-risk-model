from fastapi import FastAPI, HTTPException
from src.api.pydantic_models import CreditInferenceRequest, CreditInferenceResponse
from src.predict import CreditRiskPredictor

app = FastAPI(
    title="Bati Bank Real-Time Credit Scoring Core Service Engine",
    version="1.0.0",
    description="Basel II-compliant risk prediction pipeline serving alternative data scores."
)

# Initialize our core inference tracking object
predictor_engine = CreditRiskPredictor()

@app.get("/")
def read_root():
    """Service status health check verification endpoint."""
    return {"status": "ONLINE", "service": "Bati Bank Credit Risk Core Routing System"}

@app.post("/predict", response_model=CreditInferenceResponse)
def compute_credit_score_inference(request: CreditInferenceRequest):
    """
    Task 6: Evaluates features on the fly, applies validation checks, 
    and returns a credit risk decision using the predict layer.
    """
    # Convert Pydantic request structure to standard primitive dictionary format
    input_data = request.model_dump()
    
    # Run the prediction through our dedicated predictor class
    prediction_result = predictor_engine.predict_risk(input_data)
    
    if "ERROR" in prediction_result["status"]:
        raise HTTPException(status_code=500, detail=prediction_result["status"])
        
    return CreditInferenceResponse(
        risk_probability=prediction_result["risk_probability"],
        credit_decision=prediction_result["credit_decision"]
    )