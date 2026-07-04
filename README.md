# 🐍 AI Snake Identifier

An AI-powered web application that identifies snake species from uploaded images, classifies them, and provides essential safety information (e.g., distinguishing between venomous and non-venomous species).

---

## Project Overview
The AI Snake Identifier uses deep transfer learning (via MobileNetV2) to classify snake images. It features a lightweight FastAPI backend to serve model predictions and a responsive, simple frontend interface for user interaction.

### Key Features
* **Image-based Species Classification:** Identifies snake species from image inputs.
* **Safety Classification:** Labels species as venomous or non-venomous.
* **FastAPI Backend:** Lightweight API endpoints to handle image processing and predictions.
* **Interactive Frontend:** Easy-to-use webpage for uploading images and viewing results.

---

## Tech Stack
* **Language:** Python 3.8+
* **Deep Learning Framework:** TensorFlow 2.11+ / Keras
* **Image Processing:** Pillow (PIL), NumPy
* **Backend Framework:** FastAPI, Uvicorn
* **Frontend:** HTML5, CSS3, JavaScript

---

## Prerequisites
Ensure you have the following installed on your system:
* **Python 3.8+** (Python 3.13 recommended)
* **Git** (for version control)
* **Pip** (Python package installer)

---

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd snake
```

### 2. Virtual Environment Setup
It is highly recommended to use a virtual environment to manage dependencies.

**On Windows (PowerShell/CMD):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Dependency Installation
Install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Verify Environment Installation
Run the built-in verification script to ensure all packages (TensorFlow, FastAPI, Pillow, NumPy) are installed correctly:
```bash
python verify_environment.py
```

---

## Folder Structure
The repository is designed to cleanly separate code, dataset, and models to ensure future scalability:

```text
snake/
├── backend/                  # FastAPI Web Server Code
│   └── app.py                # Server entry point and prediction API
├── dataset/                  # Local training dataset (Git-ignored)
│   ├── cobra/                # Subfolder containing Cobra training images
│   └── krait/                # Subfolder containing Krait training images
├── frontend/                 # Client UI Application (Empty)
├── models/                   # Saved trained models and evaluation artifacts (Git-ignored)
│   ├── snake_classifier.keras # Final trained weights file
│   └── class_names.json      # Discovered classes mapping JSON
├── train.py                  # Script to train and compile the model
├── test.py                   # Script to run local inference tests
├── verify_environment.py     # Local environment diagnostic script
├── requirements.txt          # Python dependency list
├── .gitignore                # Git exclude list
└── README.md                 # Project documentation
```

---

## How to Run

### Step 1: Prepare the Dataset
Place your training images into their respective subfolders under `dataset/`:
* `dataset/cobra/` (Cobra images)
* `dataset/krait/` (Krait images)

*Note: For testing, you can run a dummy generator script to populate synthetic datasets if real images are not yet available.*

### Step 2: Train the Model
Train the MobileNetV2 classifier on the local dataset:
```bash
python train.py
```
This saves the trained model weights to `models/snake_classifier.keras` (with the best epoch saved to `models/best_snake_model.keras`).

### Step 3: Run the Backend
Start the FastAPI server:
```bash
uvicorn backend.app:app --reload
```

### Step 4: Access the Frontend
Open `frontend/index.html` in your favorite web browser to upload images and fetch predictions from the backend.

---

## Future Roadmap
- [ ] **Expand Species Dataset:** Add datasets for Vipers, Pythons, and other common snake species.
- [ ] **Model Optimizations:** Fine-tune hyperparameters and experiment with larger backbones (e.g., ResNet50) for higher accuracy.
- [ ] **Robust UI/UX:** Upgrade the frontend with drag-and-drop uploads, loading spinners, and rich safety information panels.
- [ ] **API Enhancements:** Implement detailed safety instructions, treatment guidelines, and geographical risk overlays in the backend response.
