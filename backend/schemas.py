from pydantic import BaseModel, Field
from typing import List

class PredictionClassProbability(BaseModel):
    species: str = Field(..., description="Name of the snake species")
    probability: float = Field(..., description="Probability score between 0.0 and 1.0")

class SnakeMetadata(BaseModel):
    scientific_name: str = Field(..., description="Scientific name of the snake species")
    venomous: bool = Field(..., description="Whether the snake species is venomous")
    first_aid: str = Field(..., description="Recommended emergency first-aid guidelines")

class PredictionResponse(BaseModel):
    predicted_species: str = Field(..., description="Most likely predicted snake species")
    confidence: float = Field(..., description="Confidence level of the prediction between 0.0 and 1.0")
    metadata: SnakeMetadata = Field(..., description="Safety and taxonomic metadata about the predicted species")
    probabilities: List[PredictionClassProbability] = Field(..., description="Breakdown of probabilities for all classes")
    inference_time_ms: float = Field(..., description="Model inference latency in milliseconds")
