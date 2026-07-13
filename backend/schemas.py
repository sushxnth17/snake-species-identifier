from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Dict

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
    first_aid_steps: List[str] = Field(
        ...,
        description="Sequential list of first-aid actions to perform.",
        examples=[["Keep the victim calm.", "Immobilize the bitten limb."]]
    )
    avoid_actions: List[str] = Field(
        ...,
        description="Actions the user must avoid to prevent worsening the condition.",
        examples=[["DO NOT cut the bite site.", "DO NOT apply a tourniquet."]]
    )

class TopPrediction(BaseModel):
    species: str = Field(
        ...,
        description="Species name of the ranked prediction."
    )
    confidence: float = Field(
        ...,
        description="Confidence score of the ranked prediction."
    )

class SpeciesEnrichment(BaseModel):
    overview: str = Field(
        ...,
        description="A concise herpetological overview of the species, approximately 2 to 4 sentences.",
        examples=["The spectacled cobra is a venomous snake species found in India. It is easily recognized by its hood."]
    )
    habitats: List[str] = Field(
        ...,
        description="Common habitat types for this species.",
        examples=[["grasslands", "deciduous forests"]]
    )
    appearance: List[str] = Field(
        ...,
        description="Commonly described physical traits of the species (for educational purposes only).",
        examples=[["signature hood", "spectacled markings"]]
    )
    behavior: str = Field(
        ...,
        description="A concise description of the species' typical behavior.",
        examples=["Nocturnal, typically docile unless cornered."]
    )
    interesting_facts: List[str] = Field(
        ...,
        description="A list of interesting and educational herpetological facts.",
        examples=[["Known for its iconic defensive posture"]]
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
    confidence_level: str = Field(
        ...,
        description="The calibrated confidence level of the prediction.",
        examples=["High Confidence", "Medium Confidence", "Low Confidence"]
    )
    is_uncertain: bool = Field(
        default=False,
        description="Indicates whether the prediction is uncertain."
    )
    top_predictions: List[TopPrediction] = Field(
        ...,
        description="List of the top predictions, always populated."
    )
    confidence_interpretation: str = Field(
        ...,
        description="Interpretation of the prediction confidence."
    )
    prediction_reliability: str = Field(
        ...,
        description="Overall prediction reliability rating."
    )
    explanation_text: str = Field(
        ...,
        description="Concise explanation text for the prediction and its reliability."
    )
    uncertainty_reason: Optional[str] = Field(
        default=None,
        description="Reason explaining why the prediction is uncertain."
    )
    visualization_path: Optional[str] = Field(
        default=None,
        description="The relative URL path to the saved Grad-CAM visualization image, if generated.",
        examples=["predictions/gradcam_20260710_123456.png"]
    )
    metadata: SnakeMetadata = Field(
        ...,
        description="Taxonomic and safety guidelines metadata corresponding to the predicted species."
    )
    enrichment: Optional[SpeciesEnrichment] = Field(
        default=None,
        description="Structured educational herpetological information about the predicted species, if available."
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

class ModelInfoResponse(BaseModel):
    model_name: str = Field(
        ...,
        description="The filename of the classification model.",
        examples=["snake_classifier.keras"]
    )
    model_format: str = Field(
        ...,
        description="The serialization format of the model.",
        examples=["Keras"]
    )
    supported_classes: List[str] = Field(
        ...,
        description="List of target snake species classes the model was trained to predict.",
        examples=[["cobra", "krait"]]
    )
    image_size: Tuple[int, int] = Field(
        ...,
        description="Image dimensions height and width expected by the model.",
        examples=[(224, 224)]
    )
    confidence_threshold: float = Field(
        ...,
        description="The minimum confidence threshold required for predictions.",
        examples=[0.60]
    )
    model_loaded_status: bool = Field(
        ...,
        description="True if the model has been successfully initialized into system memory.",
        examples=[True]
    )

class MetricsResponse(BaseModel):
    total_predictions: int = Field(
        ...,
        description="Total number of prediction requests received since startup.",
        examples=[10]
    )
    average_inference_time_ms: float = Field(
        ...,
        description="Average execution time of model predictions in milliseconds.",
        examples=[24.52]
    )
    uptime_seconds: float = Field(
        ...,
        description="Application running duration in seconds.",
        examples=[3600.5]
    )
    successful_predictions: int = Field(
        ...,
        description="Number of prediction requests that completed successfully.",
        examples=[8]
    )
    failed_predictions: int = Field(
        ...,
        description="Number of prediction requests that encountered errors or exceptions.",
        examples=[2]
    )

class HealthResponse(BaseModel):
    api_status: str = Field(
        ...,
        description="Operational status of the FastAPI server.",
        examples=["healthy"]
    )
    model_status: str = Field(
        ...,
        description="Load status of the classification model.",
        examples=["loaded"]
    )
    version: str = Field(
        ...,
        description="The version of the API service.",
        examples=["1.0.0"]
    )
    uptime_seconds: float = Field(
        ...,
        description="Application running duration in seconds.",
        examples=[3600.5]
    )
    timestamp: str = Field(
        ...,
        description="Current UTC timestamp in ISO 8601 format.",
        examples=["2026-07-05T21:40:00Z"]
    )
    status: str = Field(
        ...,
        description="Legacy status string.",
        examples=["healthy"]
    )
    model_loaded: bool = Field(
        ...,
        description="Legacy model load status indicator.",
        examples=[True]
    )

class DiagnosticsResponse(BaseModel):
    confidence_distribution: Dict[str, int] = Field(
        ...,
        description="Counts of predictions within 10 equal-width confidence intervals from 0.0 to 1.0.",
        examples=[{"0.9-1.0": 5, "0.8-0.9": 2}]
    )
    confidence_level_counts: Dict[str, int] = Field(
        ...,
        description="Counts of predictions per calibrated confidence level.",
        examples=[{"High Confidence": 5, "Medium Confidence": 2, "Low Confidence": 1}]
    )
    prediction_frequency: Dict[str, int] = Field(
        ...,
        description="Counts of predictions per predicted species class, including 'Uncertain'.",
        examples=[{"cobra": 5, "krait": 2, "Uncertain": 1}]
    )
    predictions_per_minute: float = Field(
        ...,
        description="Average number of prediction requests processed per minute.",
        examples=[1.25]
    )
    average_confidence: float = Field(
        ...,
        description="Average confidence score across all successful predictions.",
        examples=[0.8724]
    )
    uncertain_prediction_rate: float = Field(
        ...,
        description="Ratio of low confidence / uncertain predictions to total successful predictions.",
        examples=[0.125]
    )
    total_predictions: int = Field(
        ...,
        description="Total number of prediction requests processed.",
        examples=[10]
    )
