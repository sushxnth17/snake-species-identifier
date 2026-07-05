from pydantic import BaseModel, Field

class SnakeMetadata(BaseModel):
    common_name: str = Field(
        ...,
        description="Common name of the snake species.",
        examples=["Spectacled Cobra"]
    )
    scientific_name: str = Field(
        ...,
        description="Scientific name of the snake species (genus and species).",
        examples=["Naja naja"]
    )
    venomous: bool = Field(
        ...,
        description="Indicates whether the snake species is venomous and poses a threat.",
        examples=[True]
    )
    description: str = Field(
        ...,
        description="Detailed description of the snake physical characteristics, behaviors, and patterns.",
        examples=["The Indian cobra or spectacled cobra is a highly venomous species of the genus Naja found in the Indian subcontinent. It is characterized by its signature hood and spectacled mark on the back of its neck."]
    )
    habitat: str = Field(
        ...,
        description="Primary geographical locations and environments where this species is commonly found.",
        examples=["Forests, wetlands, grasslands, agricultural areas, and near human settlements."]
    )
    first_aid: str = Field(
        ...,
        description="Highly recommended emergency first-aid procedures to perform immediately in case of a bite.",
        examples=[
            "1. Keep the victim calm and reassured to slow venom circulation.\n"
            "2. Immobilize the bitten limb using a splint or loose bandage.\n"
            "3. Remove rings, bracelets, or tight clothing near the bite area.\n"
            "4. Transport the victim immediately to the nearest medical facility with anti-venom.\n"
            "5. DO NOT cut the bite site, apply a tourniquet, or try to suck out the venom."
        ]
    )

class PredictionResponse(BaseModel):
    species: str = Field(
        ...,
        description="The predicted class label of the snake species.",
        examples=["cobra"]
    )
    confidence: float = Field(
        ...,
        description="Confidence score of the prediction, representing probability between 0.0 and 1.0.",
        examples=[0.9724]
    )
    metadata: SnakeMetadata = Field(
        ...,
        description="Taxonomic and safety guidelines metadata corresponding to the predicted species."
    )
    inference_time_ms: float = Field(
        ...,
        description="The model inference duration in milliseconds.",
        examples=[45.23]
    )

class ErrorDetail(BaseModel):
    code: int = Field(
        ...,
        description="The HTTP status code corresponding to the error.",
        examples=[400]
    )
    message: str = Field(
        ...,
        description="A user-friendly detail message explaining why the request failed.",
        examples=["Bad Request: Invalid image format or content"]
    )

class ErrorResponse(BaseModel):
    error: ErrorDetail = Field(
        ...,
        description="Wrapper containing the detailed error response structure."
    )
