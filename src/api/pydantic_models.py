from pydantic import BaseModel, Field

class CreditInferenceRequest(BaseModel):
    """Data structure verification model for incoming API request payloads."""
    Amount: float = Field(..., description="Numerical transaction amount value", examples=[1500.50])
    Value: float = Field(..., description="Absolute absolute numerical value metrics", examples=[1500.50])
    ProductCategory: str = Field(..., description="Category group classifying item", examples=["utility"])
    ChannelId: str = Field(..., description="Platform routing source tracking point", examples=["Channel_3"])
    PricingStrategy: int = Field(..., description="Xente internal financial tier system rating indicator", examples=[2])

    class Config:
        populate_by_name = True

class CreditInferenceResponse(BaseModel):
    """Structured response model returned to downstream calling microservices."""
    risk_probability: float = Field(..., description="Estimated statistical probability score output", examples=[0.145])
    credit_decision: int = Field(..., description="Binary categorization action rule outcome: 1=High Risk, 0=Approved/Low Risk", examples=[0])