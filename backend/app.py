from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import logging

from backend.config import APP_TITLE, APP_DESCRIPTION, APP_VERSION, ALLOWED_ORIGINS
from backend.schemas import PredictionResponse, SnakeMetadata, PredictionClassProbability
from backend.utils import setup_logging
from backend import model_loader
from backend import predictor
from backend.metadata import get_snake_metadata

# Setup Logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load the model and class names exactly once.
    # If the model does not exist or fails to load, log it and allow startup
    # so routes can still be served (especially for frontend/API debugging).
    logger.info("Initializing application resources during startup lifespan...")
    try:
        model_loader.load_model()
        model_loader.load_class_names()
        logger.info("Initialization completed successfully.")
    except Exception as e:
        logger.error(
            f"Startup failed to initialize TensorFlow model. "
            f"API endpoints will fall back to mock predictions. Details: {e}"
        )
    yield
    # Shutdown logic (none needed for now)
    logger.info("Cleaning up application resources during shutdown lifespan...")

app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify backend status and model readiness.
    """
    model_loaded = model_loader._model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_species(file: UploadFile = File(...)):
    """
    Accepts an uploaded image of a snake and predicts its species, returning
    taxonomic classification, safety warnings, and first-aid recommendations.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image.")

    start_time = time.perf_counter()
    
    try:
        # Read raw image bytes
        image_bytes = await file.read()
        
        # 1. Preprocess the image
        preprocessed = predictor.preprocess_image(image_bytes)
        
        # 2. Get the loaded model
        try:
            model = model_loader.get_model()
        except RuntimeError:
            logger.warning("Model was not loaded on startup. Attempting lazy load...")
            try:
                model = model_loader.load_model()
            except Exception:
                model = None
        
        # 3. Model Prediction
        raw_predictions = predictor.predict(model, preprocessed)
        
        # 4. Read class names
        try:
            class_names = model_loader.get_class_names()
        except Exception:
            try:
                class_names = model_loader.load_class_names()
            except Exception:
                class_names = ["cobra", "krait"]
            
        # 5. Extract top species and full probability breakdown
        predicted_species, confidence, probs = predictor.format_prediction_results(raw_predictions, class_names)
        
        # 6. Retrieve Safety Metadata
        meta_dict = get_snake_metadata(predicted_species)
        snake_meta = SnakeMetadata(**meta_dict)
        
        # Calculate latency
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Map probability dicts to Pydantic objects
        response_probs = [
            PredictionClassProbability(species=p["species"], probability=p["probability"])
            for p in probs
        ]
        
        return PredictionResponse(
            predicted_species=predicted_species,
            confidence=confidence,
            metadata=snake_meta,
            probabilities=response_probs,
            inference_time_ms=inference_time_ms
        )
        
    except Exception as e:
        logger.exception("Prediction endpoint encountered an error")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")