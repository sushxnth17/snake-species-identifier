# Snake Image Dataset Specifications

This directory holds the training images used to train the snake classification model. The data pipeline utilizes TensorFlow's `image_dataset_from_directory` utility, which requires structured class-based directory separation.

---

## Expected Directory Layout
The dataset folder must contain a flat subdirectory for each target snake species. Do not nest species directories inside other class folders.

```text
dataset/
├── README.md               # Dataset specification documentation
├── cobra/                  # Subdirectory containing all cobra images
│   ├── cobra_0.jpg
│   ├── cobra_1.jpg
│   └── ...
└── krait/                  # Subdirectory containing all krait images
    ├── krait_0.jpg
    ├── krait_1.jpg
    └── ...
```

---

## Supported Image Formats
TensorFlow's image loading utilities natively support the following standard formats:
* `.jpg` / `.jpeg` (Recommended)
* `.png`
* `.bmp`
* `.gif` (Static frame only)

*Note: Ensure all images are valid, uncorrupted, and have 3 color channels (RGB) for training.*

---

## Naming Conventions
To keep the dataset structured and avoid namespace collisions, use the following lowercase snake case naming format:
`[species_name]_[index].[extension]`

**Examples:**
* `cobra_001.jpg`
* `krait_124.png`

---

## Minimum Recommended Size
For reliable fine-tuning of pretrained CNN backbones (like `MobileNetV2`):
* **Minimum per class:** 100 images
* **Recommended per class:** 500+ images for balanced feature representation
* **Ratio:** Maintain a roughly equal number of images per class (e.g. 50/50 balance) to prevent model classification bias.

---

## Scalability
Adding new snake species is straightforward:
1. Create a new subdirectory under `dataset/` named after the snake species (e.g., `dataset/viper/`).
2. Populate the folder with images of that species following the naming conventions.
3. The training pipeline `train.py` dynamically infers classes from these subdirectories using:
   ```python
   tf.keras.utils.image_dataset_from_directory("dataset", labels="inferred", label_mode="int")
   ```
   No modifications to the pipeline code are required to recognize the new species.
