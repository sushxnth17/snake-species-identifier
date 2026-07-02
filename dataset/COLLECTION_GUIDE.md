# Computer Vision Dataset Collection Guide: Snake Species Classifier

This guide provides instructions and standards for collecting, cleaning, and formatting image data to train our TensorFlow-based snake species classifier. Adhering to these guidelines ensures a robust dataset that minimizes model bias and maximizes generalization.

---

## 1. Recommended Image Sources
To ensure biological accuracy and variety, collect images from these sources:
*   **Scientific and Citizen Science Platforms (Highly Recommended):**
    *   [iNaturalist](https://www.inaturalist.org/) (Excellent for geo-located, verified wild observations)
    *   [HerpMapper](https://www.herpmapper.org/) (Global repository for reptile/amphibian data)
    *   [CalPhotos](https://calphotos.berkeley.edu/) (UC Berkeley repository of nature photos)
*   **Public Machine Learning Repositories:**
    *   [Kaggle](https://www.kaggle.com/) (Search for existing reptile or snake identification datasets)
*   **Search Engine Scraping (Use Caution):**
    *   Google Images, Flickr (Must be manually filtered to verify correct species identifications)

---

## 2. Target Number of Images per Class
For reliable transfer learning with architectures like `MobileNetV2` or `ResNet50`:
*   **Minimum Target:** 150 images per class (useful for baseline prototyping)
*   **Production Target:** 500 to 1,000+ images per class
*   **Class Balancing:** Keep the number of images per class within a 1.2x ratio (e.g., if Class A has 500 images, Class B should have between 420 and 600 images) to prevent the loss function from biasing towards the majority class.

---

## 3. Data Diversity Requirements
To train a model that works in real-world scenarios, the dataset must have high variance across several axes:

### Background Diversity
Snakes should be photographed on diverse substrates:
*   **Natural environments:** Leaves, dirt, grass, sand, rocks, water, tree branches.
*   **Artificial environments:** Concrete, asphalt, carpets, tiles, cages/glass enclosures.
*   **Context variation:** Images containing *only* snakes vs. images where the snake is camouflaged in vegetation.

### Lighting Diversity
Collect images taken under different lighting conditions:
*   Direct sunlight (high contrast, harsh shadows).
*   Overcast daylight (soft lighting, diffused shadows).
*   Dawn/dusk or low-light conditions.
*   Flash photography / artificial enclosure lighting.

### Camera Angle & Perspective Diversity
Capture different poses and view perspectives:
*   **Top-down view:** Showing the full dorsal pattern (critical for pattern matching).
*   **Eye-level / Side view:** Showing head shapes and scale arrangements.
*   **Macro close-ups:** Head, tail, or scale detail.
*   **Wide-angle shots:** Entire snake curled or moving across the ground.
*   **State variation:** Coiled defensive positions, stretched out, or climbing.

---

## 4. Image Quality & Resolution Recommendations
*   **Resolution:** Minimum resolution of **500x500 pixels**. Standard CNN backbones crop/resize inputs to `224x224` or `299x299` pixels. High-resolution images are acceptable but do not need to exceed 4K.
*   **Focus & Clarity:** The snake must be reasonably sharp and in focus. Avoid heavily blurred or shaky photos.
*   **Aspect Ratio:** Square or near-square aspect ratios are preferred, though TensorFlow handles resizing/aspect-ratio warping automatically.

---

## 5. Duplicate and Near-Duplicate Handling
Duplicate images cause data leakage (e.g. identical images in both train and validation splits), leading to overly optimistic metric scores.
*   **Exact Duplicates:** Eliminate using cryptographic hashes (e.g., MD5 check) on the raw image files.
*   **Near-Duplicates (Burst Mode Shots):** If an observer uploaded multiple photos of the same individual snake in the exact same position, **only keep the single best photo**. Including multiple near-identical frames artificially inflates training metrics and causes overfitting.

---

## 6. Common Mistakes to Avoid
*   **Species Misidentification:** The most critical error. Ensure images sourced from search engine scrapers are verified by a subject matter expert or sourced from platforms with community verification (e.g., iNaturalist Research Grade).
*   **Watermarks and Overlays:** Avoid images containing copyright text, timestamps, or logo overlays. The model may learn to associate the watermark pattern with a specific class rather than the snake itself.
*   **Human Hand Bias:** Avoid images showing people holding the snakes. The neural network can easily learn to detect human hands/skin tones rather than the reptile features.
*   **Specimen Bias:** Do not source all images of a species from a single individual or a single zoo enclosure. The model will overfit to that specific environment or specimen's pattern.
