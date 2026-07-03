from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    species: str = Field(..., description="Predicted snake species")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
