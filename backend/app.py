from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
import logging

from backend.config import settings
from backend.schemas import PredictionResponse, ErrorResponse
from backend.utils import setup_logging
from backend import model_loader
from backend import predictor
from backend.metadata import get_snake_metadata

# Setup Logging
log_level = getattr(logging, settings.logging_level.upper(), logging.INFO)
setup_logging(level=log_level)
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
    title=settings.api_title,
    description=settings.app_description,
    version=settings.api_version,
    debug=settings.debug,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Exception Handlers ---

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTPException: status_code={exc.status_code}, detail={exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.status_code, "message": exc.detail}}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    logger.error(f"RequestValidationError: {errors}")
    error_msg = "; ".join([f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in errors])
    return JSONResponse(
        status_code=422,
        content={"error": {"code": 422, "message": f"Validation Error: {error_msg}"}}
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    logger.error(f"ValueError: {exc}", exc_info=True)
    return JSONResponse(
        status_code=400,
        content={"error": {"code": 400, "message": f"Bad Request: {str(exc)}"}}
    )

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    logger.error(f"FileNotFoundError: {exc}", exc_info=True)
    return JSONResponse(
        status_code=503,
        content={"error": {"code": 503, "message": f"Service Unavailable: Required resource missing ({str(exc)})"}}
    )

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    logger.error(f"RuntimeError: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": {"code": 500, "message": f"Internal Server Error: {str(exc)}"}}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": {"code": 500, "message": "An unexpected error occurred on the server."}}
    )


@app.get(
    "/health",
    summary="Check System Health",
    description="Verifies the operational status of the API service, CORS bindings, and checks whether the TensorFlow classification model has successfully loaded into memory.",
    response_description="A JSON status object indicating health and model load state.",
    responses={
        200: {
            "description": "API is healthy and running.",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "model_loaded": True
                    }
                }
            }
        }
    }
)
def health_check():
    """
    Health check endpoint to verify backend status and model readiness.
    """
    model_loaded = model_loader._model is not None
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Snake Species from Image",
    description=(
        "Uploads a picture of a snake and runs deep-learning classification.\n\n"
        "**Validations Executed**:\n"
        "- **File Existence**: Verifies the payload is not empty.\n"
        f"- **MIME Type**: Restricts uploads to standard images ({', '.join(settings.allowed_mime_types)}).\n"
        f"- **Max File Size**: Caps uploads at {settings.max_upload_size / (1024 * 1024):.1f}MB.\n\n"
        "**Inference Pipeline**:\n"
        "1. Decodes and scales input using MobileNetV2 preprocessing parameters.\n"
        "2. Performs prediction on the loaded neural network model.\n"
        "3. Fetches taxonomic description, danger class, and emergency first-aid guidelines."
    ),
    response_description="Enriched snake prediction detailing species, confidence, scientific name, and first-aid recommendations.",
    responses={
        200: {
            "model": PredictionResponse,
            "description": "Inference successfully completed."
        },
        400: {
            "model": ErrorResponse,
            "description": "Bad Request. Uploaded file is empty, corrupted, or has invalid data format."
        },
        413: {
            "model": ErrorResponse,
            "description": f"Payload Too Large. Uploaded image file size exceeds the {settings.max_upload_size / (1024 * 1024):.1f}MB limit."
        },
        415: {
            "model": ErrorResponse,
            "description": "Unsupported Media Type. Uploaded file format or MIME type is not allowed."
        },
        500: {
            "model": ErrorResponse,
            "description": "Internal Server Error. Encountered unhandled exception during processing or inference."
        },
        503: {
            "model": ErrorResponse,
            "description": "Service Unavailable. Model has not been initialized or required files are missing."
        }
    }
)
async def predict_species(file: UploadFile = File(..., description="The binary image of the snake to classify (JPEG, PNG, or WebP format).")):
    """
    Accepts an uploaded image of a snake and predicts its species.
    Validates file existence, MIME type, and size limits before running inference.
    """
    # 1. Validate MIME type first (quick check on headers)
    if file.content_type not in settings.allowed_mime_types:
        logger.warning(f"Rejected upload with unsupported MIME type: {file.content_type}")
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: '{file.content_type}'. Allowed types: {', '.join(settings.allowed_mime_types)}"
        )
        
    # Read file content into memory
    image_bytes = await file.read()
    
    # 2. Validate file existence
    if not image_bytes or len(image_bytes) == 0:
        logger.warning("Rejected upload with empty file content.")
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        
    # 3. Validate maximum upload size
    if len(image_bytes) > settings.max_upload_size:
        max_mb = settings.max_upload_size / (1024 * 1024)
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