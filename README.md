# 🐍 Snakify

Snakify is a snake species identification web application that analyzes uploaded snake images using a deep learning model and presents confidence-aware prediction results.

Along with species prediction, Snakify provides model explainability through Grad-CAM, structured educational species information through Groq-powered enrichment, and trusted safety guidance based on backend-maintained metadata.

> Snakify is intended for educational and informational use. Model predictions can be incorrect and should not be treated as a substitute for professional wildlife identification or emergency medical care.

---

## Project Overview

Snakify uses transfer learning with MobileNetV2 to classify snake images.

The machine learning pipeline performs image preprocessing and species classification, while the FastAPI backend handles image validation, inference, confidence interpretation, explainability, species enrichment, safety metadata, rate limiting, and API responses.

The React frontend provides an interactive workflow for uploading a snake image, viewing the prediction, understanding model confidence, inspecting Grad-CAM attention regions, reading educational species information, and reviewing relevant safety guidance.

### Application Flow

```text
Snake Image
    ↓
Image Validation
    ↓
MobileNetV2 Prediction
    ↓
Confidence Calibration
    ↓
Prediction Reliability Check
    ↓
├── Species Prediction
├── Confidence Information
├── Top Prediction Candidates
├── Grad-CAM Explainability
├── Trusted Venom & Safety Metadata
└── Groq Species Knowledge Enrichment
          ↓
    Pydantic Validation
          ↓
      Species Cache
          ↓
    React Prediction Interface
```

For uncertain predictions, Snakify avoids presenting a species as definitively identified and skips species enrichment to prevent additional information from making an uncertain result appear more reliable.

---

## Key Features

### 🐍 Image-Based Snake Classification

- Upload a snake image for species prediction.
- Uses a MobileNetV2 transfer learning model.
- Dynamically discovers species classes from the training dataset.
- Supports model fine-tuning.
- Returns the most likely species and prediction confidence.
- Provides top prediction candidates where available.

### 📊 Confidence-Aware Predictions

- Uses confidence calibration and reliability interpretation.
- Distinguishes confident and uncertain predictions.
- Avoids presenting low-confidence results as confirmed identifications.
- Preserves top prediction candidates for uncertain results.
- Keeps prediction uncertainty separate from API errors.

### 🔥 Grad-CAM Explainability

Snakify integrates Grad-CAM to help visualize which image regions influenced the model's prediction.

The frontend displays:

- The original uploaded image.
- The model attention visualization.
- A clear explanation of what highlighted regions represent.
- Cautious guidance explaining that Grad-CAM does not prove biological identification.

Grad-CAM is used as an interpretation tool and does not independently identify the snake.

### 🧠 Groq Species Knowledge Enrichment

For sufficiently reliable predictions, Snakify uses the Groq API to generate structured educational information about the predicted species.

The enrichment layer provides:

- Species overview.
- Common habitats.
- Physical appearance traits.
- Typical behavior.
- Interesting facts.

Groq receives only a validated species label produced by the existing prediction pipeline.

Uploaded images, image bytes, Grad-CAM visualizations, and arbitrary user prompts are not sent to Groq.

Groq does not:

- Identify the uploaded snake.
- Override TensorFlow predictions.
- Determine prediction confidence.
- Determine venom status.
- Generate first-aid instructions.
- Generate emergency medical guidance.

All generated species information is validated using Pydantic before being returned to the frontend.

### 💾 Species Enrichment Cache

Validated Groq species information is cached to avoid repeated provider requests for the same species.

The enrichment flow is:

```text
Predicted Species
       ↓
Validate Species Label
       ↓
Check Enrichment Cache
       ↓
   Cache Available?
      /       \
    Yes        No
     ↓          ↓
Validate     Groq API
Cached Data      ↓
     ↓       Validate Response
     │             ↓
     │        Cache Valid Data
     └──────┬──────┘
            ↓
    Species Information
```

Only successfully validated enrichment responses are cached.

Provider errors, malformed responses, timeouts, and validation failures are not cached.

Groq failures do not prevent snake predictions from succeeding.

### ⚠️ Trusted Safety Guidance

Safety-critical information is kept separate from Groq-generated educational content.

Trusted backend metadata is used for:

- Venomous status.
- Prediction-aware safety states.
- First-aid guidance.
- Actions to avoid.
- Emergency safety messaging.

Snakify supports separate safety states for:

- Venomous predictions.
- Non-venomous predictions.
- Uncertain predictions.

For uncertain predictions, the snake is treated as potentially dangerous.

The application does not use the top prediction candidate to reassure users when the overall prediction is uncertain.

### 🖼️ React Image Upload Experience

The frontend supports:

- Click-to-browse image selection.
- Drag-and-drop uploads.
- Client-side image validation.
- Image preview.
- File name and file size display.
- Image replacement.
- Image removal.
- Accessible upload controls.
- Object URL cleanup to prevent browser memory leaks.

Changing or removing an image clears stale prediction, Grad-CAM, enrichment, and safety state.

### ⚛️ React + Vite Frontend

The Snakify frontend is built using React and Vite.

The frontend includes:

- Component-based architecture.
- Real FastAPI integration.
- Environment-based API configuration.
- Request loading states.
- Request timeout handling.
- Normalized API errors.
- Confidence-aware prediction UI.
- Uncertain prediction presentation.
- Grad-CAM comparison.
- Species knowledge presentation.
- Trusted safety guidance.
- Responsive layouts.
- Accessibility improvements.

### 🚀 FastAPI Prediction Service

The backend provides:

- Multipart image upload handling.
- Image validation.
- TensorFlow model inference.
- Confidence-aware prediction responses.
- Uncertain prediction handling.
- Top prediction candidates.
- Grad-CAM integration.
- Species metadata.
- Groq enrichment integration.
- Trusted safety information.
- Structured API errors.
- Request ID support.
- Rate limiting.
- CORS configuration.
- Diagnostics and metrics.

---

## Tech Stack

### Machine Learning

- Python 3.10
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pillow
- Grad-CAM

### Backend

- FastAPI
- Uvicorn
- Pydantic
- Python Multipart
- Groq Python SDK

### Frontend

- React
- Vite
- JavaScript
- CSS3

### AI Enrichment

- Groq API
- Structured species knowledge generation
- Pydantic response validation
- Species enrichment caching

### Development

- Git
- GitHub
- Python virtual environments
- npm

---

## Prerequisites

Ensure the following are installed:

- Python 3.10
- Node.js and npm
- Git
- pip

Python 3.10 is recommended for compatibility with the TensorFlow environment used by this project.

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd snake
```

### 2. Create a Python Virtual Environment

#### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the Python Environment

```bash
python verify_environment.py
```

The verification script checks the required Python environment and core dependencies.

### 5. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

---

## Environment Configuration

Snakify uses environment variables for backend and frontend configuration.

### Backend Environment

Create a local `.env` file based on the project's environment example.

```env
GROQ_API_KEY=your_groq_api_key
```

The Groq API key is optional.

If `GROQ_API_KEY` is unavailable:

- The FastAPI backend still starts.
- Snake prediction still works.
- Confidence handling still works.
- Grad-CAM still works.
- Safety guidance still works.
- Species enrichment becomes unavailable gracefully.

> Never commit the real `.env` file or Groq API key to Git.

The Groq API key must remain backend-only.

Do not create:

```env
VITE_GROQ_API_KEY=...
```

### Frontend Environment

Configure the FastAPI base URL using:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

For production, replace the value with the deployed FastAPI service URL.

---

## Folder Structure

The exact repository structure may evolve as Snakify grows. The main architecture is organized as follows:

```text
snake/
├── backend/
│   ├── clients/                 # External service clients
│   ├── services/                # Prediction and enrichment services
│   ├── schemas/                 # Pydantic API schemas
│   ├── middleware/              # Request and security middleware
│   └── app.py                   # FastAPI application entry point
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/          # React UI components
│   │   ├── hooks/               # React hooks
│   │   ├── services/            # Frontend API services
│   │   ├── utils/               # Frontend utilities
│   │   ├── styles/              # Application styles
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── ml/                          # Machine learning pipeline
│   ├── train.py
│   ├── test.py
│   └── evaluation utilities
│
├── dataset/                     # Local training dataset
│   ├── cobra/
│   └── krait/
│
├── models/                      # Trained models and ML metadata
│   ├── snake_classifier.keras
│   └── class_names.json
│
├── data/
│   └── species_enrichment/      # Validated species enrichment cache
│
├── tests/                       # Backend and integration tests
├── requirements.txt
├── verify_environment.py
├── .env.example
├── .gitignore
└── README.md
```

Dataset images, trained model binaries, local environment files, and other generated artifacts may be Git ignored according to the repository configuration.

---

## Dataset Structure

The dataset follows the directory structure expected by Keras image data loaders.

```text
dataset/
├── cobra/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
│
└── krait/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

Each species is represented by a separate directory.

The training pipeline dynamically derives class names from the dataset folder structure.

Additional species can be introduced by adding new class directories and providing sufficient validated training images.

---

## Training the Model

Run the training pipeline from the project root:

```bash
python train.py
```

If the current training CLI supports configurable training phases, use:

```bash
python train.py --epochs 15 --fine_tune_epochs 10
```

This performs:

1. Initial transfer learning.
2. MobileNetV2 feature extraction.
3. Validation-based model evaluation.
4. Fine-tuning of selected model layers.
5. Model artifact generation.

Generated model artifacts are stored in the configured models directory.

---

## Local Model Testing

Use the local inference script to test a single image without starting FastAPI.

```bash
python test.py --image path/to/image.jpg
```

The script loads the trained model, preprocesses the image using the model's expected input pipeline, and prints prediction information to the terminal.

---

## How to Run Snakify

Snakify requires the FastAPI backend and React frontend to run simultaneously.

### Step 1: Activate the Python Environment

#### Windows

```powershell
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### Step 2: Start the FastAPI Backend

From the project root:

```bash
uvicorn backend.app:app --reload
```

The backend typically runs at:

```text
http://127.0.0.1:8000
```

### Step 3: Start the React Frontend

Open another terminal:

```bash
cd frontend
npm run dev
```

Vite will display the local development URL, typically:

```text
http://localhost:5173
```

Open the URL shown by Vite in your browser.

Do not open `frontend/index.html` directly.

The frontend is a Vite application and should run through the Vite development server.

---

## Prediction Workflow

The prediction process is:

```text
Select Snake Image
        ↓
Client-Side Validation
        ↓
Image Preview
        ↓
Analyze Snake
        ↓
Multipart Upload to FastAPI
        ↓
Backend Image Validation
        ↓
TensorFlow Preprocessing
        ↓
MobileNetV2 Inference
        ↓
Confidence Calibration
        ↓
Reliability Evaluation
        ↓
Grad-CAM Explainability
        ↓
Trusted Safety Metadata
        ↓
Reliable Prediction?
       /             \
     No               Yes
      ↓                 ↓
Uncertain Result   Species Enrichment
                         ↓
                     Cache Check
                         ↓
                      Groq API
                    when required
                         ↓
                  Pydantic Validation
                         ↓
                    React Results
```

---

## API Reliability and Error Handling

The frontend handles prediction request states and backend errors separately from model uncertainty.

An uncertain prediction is a successful API response where the model does not have enough confidence to present a species as reliably identified.

Supported API and client error handling includes relevant cases such as:

- Invalid uploads.
- Unsupported media types.
- Oversized images.
- Validation failures.
- Rate limits.
- Backend failures.
- Service unavailability.
- Network failures.
- Request timeouts.

The frontend avoids exposing internal Python errors, TensorFlow exceptions, provider errors, and internal file paths.

---

## Security and Rate Limiting

The FastAPI backend includes security controls for prediction requests.

### Upload Validation

Uploaded files are validated for:

- Maximum file size.
- Supported image format.
- Image file headers.
- Image dimensions.
- Invalid or corrupted image data.
- Decompression bomb risks.

Default upload size:

```text
5 MB
```

Default maximum image dimensions:

```text
4096 × 4096 pixels
```

### Rate Limiting

The backend uses per-client rate limiting to protect prediction resources.

Relevant configuration includes:

```env
RATE_LIMIT_ENABLED=True
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

Rate-limited requests return:

```text
HTTP 429 Too Many Requests
```

A `Retry-After` header may be included according to the backend implementation.

### CORS

Allowed frontend origins are configured through:

```env
CORS_ORIGINS=http://localhost:5173
```

Production deployments should configure explicit frontend domains.

Do not use wildcard origins with credential-enabled CORS configurations.

Example:

```env
CORS_ORIGINS=https://snakify.example.com
```

---

## Groq Integration and Security

Groq is used only for structured educational species enrichment.

The integration follows these rules:

- The API key remains in backend environment configuration.
- React never calls Groq directly.
- The Groq API key is never stored in a `VITE_*` variable.
- Uploaded images are not sent to Groq.
- Grad-CAM images are not sent to Groq.
- Arbitrary frontend prompts are not accepted.
- Only trusted species labels are used for enrichment.
- Unsupported species do not trigger enrichment requests.
- Uncertain predictions do not trigger enrichment.
- Groq failures do not fail valid snake predictions.
- Groq-generated data is validated before frontend presentation.

The species enrichment feature is an educational layer and is not part of the image classification model.

---

## Explainable AI

Snakify uses Grad-CAM to provide visual insight into model attention.

The Grad-CAM visualization highlights image regions that had a stronger influence on the model's prediction.

This feature helps inspect model behavior and may reveal whether the classifier is focusing on relevant snake regions or unrelated background features.

Grad-CAM does not confirm species identity and should not be interpreted as biological evidence.

---

## Safety Notice

Snakify is an educational and informational project.

Snake identification from images can be incorrect because of:

- Poor image quality.
- Similar-looking species.
- Partial visibility.
- Lighting conditions.
- Background bias.
- Limited training data.
- Model uncertainty.

Do not approach, handle, or attempt to capture a snake based on a Snakify prediction.

If a snake bite may have occurred, seek medical attention immediately.

Do not delay emergency care while waiting for an image identification result.

Groq-generated educational species information is not used as medical or first-aid guidance.

---

## Current Limitations

- The current model supports a limited number of trained snake species.
- Prediction quality depends heavily on the dataset.
- Visually similar species may be difficult to distinguish.
- Grad-CAM explains model attention but does not verify identification.
- Groq-generated educational information may contain inaccuracies.
- Species enrichment requires Groq availability unless validated cached data exists.
- The application is not a replacement for professional snake identification.
- The application is not a medical diagnostic system.

---

## Future Roadmap

- [ ] Expand the dataset to additional snake species.
- [ ] Improve class balance and dataset quality.
- [ ] Add Vipers, Pythons, and additional commonly encountered species.
- [ ] Perform broader real-world model evaluation.
- [ ] Compare MobileNetV2 with alternative image-classification backbones.
- [ ] Improve confidence calibration using larger validation datasets.
- [ ] Analyze Grad-CAM attention patterns for dataset bias.
- [ ] Add frontend automated testing infrastructure.
- [ ] Add model version tracking.
- [ ] Add enrichment cache versioning and refresh policies.
- [ ] Improve production deployment configuration.
- [ ] Deploy the React frontend and FastAPI backend.

---

## Disclaimer

Snakify predictions may be incorrect.

The application is intended for educational and informational purposes only and should not be used as the sole basis for approaching, handling, or making safety decisions about a snake.

If a snake bite may have occurred, seek professional medical attention immediately.

---

## License

This project is currently maintained as an educational and portfolio project.

Add a formal open-source license before distributing or accepting external contributions under specific licensing terms.
