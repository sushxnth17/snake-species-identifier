from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import logging

from backend.config import APP_TITLE, APP_DESCRIPTION, APP_VERSION, ALLOWED_ORIGINS, ALLOWED_MIME_TYPES, MAX_UPLOAD_SIZE
from backend.schemas import PredictionResponse
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
    Accepts an uploaded image of a snake and predicts its species.
    Validates file existence, MIME type, and size limits before running inference.
    """
    # 1. Validate MIME type first (quick check on headers)
    if file.content_type not in ALLOWED_MIME_TYPES:
        logger.warning(f"Rejected upload with unsupported MIME type: {file.content_type}")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: '{file.content_type}'. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
        )
        
    try:
        # Read file content into memory
        image_bytes = await file.read()
        
        # 2. Validate file existence
        if not image_bytes or len(image_bytes) == 0:
            logger.warning("Rejected upload with empty file content.")
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
            
        # 3. Validate maximum upload size
        if len(image_bytes) > MAX_UPLOAD_SIZE:
            max_mb = MAX_UPLOAD_SIZE / (1024 * 1024)
            logger.warning(f"Rejected upload exceeding maximum size limit: {len(image_bytes)} bytes")
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds the maximum limit of {max_mb:.1f}MB."
            )
            
        # 4. Preprocess image
        preprocessed = predictor.preprocess_image(image_bytes)
        
        # 5. Run prediction
        try:
            model = model_loader.get_model()
        except RuntimeError as e:
            logger.warning("Model was not loaded on startup. Attempting lazy load...")
            try:
                model = model_loader.load_model()
            except Exception as lazy_err:
                logger.error(f"Lazy loading model failed: {lazy_err}")
                raise HTTPException(
                    status_code=503,
                    detail="Model is not available. Please try again later."
                )
                
        raw_predictions = predictor.predict(model, preprocessed)
        
        # 6. Read class names
        try:
            class_names = model_loader.get_class_names()
        except Exception:
            class_names = model_loader.load_class_names()
            
        # 7. Extract predicted class and confidence
        predicted_species, confidence = predictor.format_prediction_results(raw_predictions, class_names)
        
        # 8. Retrieve Safety and Taxonomic Metadata
        meta_dict = get_snake_metadata(predicted_species)
        
        return PredictionResponse(
            species=predicted_species,
            confidence=confidence,
            metadata=meta_dict
        )
        
    except HTTPException as he:
        raise he
    except ValueError as ve:
        logger.error(f"Value error during prediction pipeline: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Unexpected error in prediction pipeline")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")