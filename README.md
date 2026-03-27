# 🐍 AI Snake Identifier

An AI-powered web application that identifies snake species from images and provides safety information.

## Features

* Image-based snake classification
* Detects venomous vs non-venomous snakes
* FastAPI backend for predictions
* Simple frontend interface

## Tech Stack

* Python
* TensorFlow
* FastAPI
* HTML, CSS, JavaScript

## Dataset

Dataset is not included in this repository.
Images will be collected from Kaggle, Google Images.

## How to Run

1. Train the model:
   python train.py

2. Start backend:
   uvicorn backend.app:app --reload

3. Open frontend:
   Open index.html in browser
