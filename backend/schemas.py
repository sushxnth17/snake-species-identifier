from pydantic import BaseModel, Field

class SnakeMetadata(BaseModel):
    common_name: str = Field(..., description="Common name of the snake species")
    scientific_name: str = Field(..., description="Scientific name of the snake species")
    venomous: bool = Field(..., description="Whether the snake is venomous")
    description: str = Field(..., description="Description of the snake species")
    habitat: str = Field(..., description="Habitat details of the snake species")
    first_aid: str = Field(..., description="First aid information in case of a bite")

class PredictionResponse(BaseModel):
    species: str = Field(..., description="Predicted snake species")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")
    metadata: SnakeMetadata = Field(..., description="Safety and taxonomic metadata about the predicted species")

