/**
 * API Service for AI Snake Species Identifier.
 * This is a frontend service module placeholder for future FastAPI integration.
 * In a future phase, these stubs will be replaced by actual fetch requests.
 */

// Toggle mock delay (milliseconds)
const MOCK_DELAY = 1200;

// High-confidence Venomous prediction mock (Cobra)
const MOCK_SUCCESS_COBRA = {
  species: "cobra",
  confidence: 0.9682,
  confidence_level: "High Confidence",
  is_uncertain: false,
  top_predictions: [
    { species: "cobra", confidence: 0.9682 },
    { species: "krait", confidence: 0.0318 }
  ],
  confidence_interpretation: "Confidence exceeds the 90.0% accuracy threshold determined during validation calibration.",
  prediction_reliability: "High",
  explanation_text: "Predicted cobra with 96.82% confidence. The prediction is highly reliable.",
  uncertainty_reason: null,
  visualization_path: "C:\\Users\\susha\\OneDrive\\Desktop\\snake\\predictions\\gradcam_mock_cobra.png",
  metadata: {
    common_name: "Spectacled Cobra",
    scientific_name: "Naja naja",
    venomous: true,
    description: "The Indian cobra or spectacled cobra is a highly venomous species of the genus Naja found in the Indian subcontinent. It is characterized by its signature hood and spectacled mark on the back of its neck.",
    habitat: "Forests, wetlands, grasslands, agricultural areas, and near human settlements.",
    first_aid: "1. Keep the victim calm and reassured to slow venom circulation.\n2. Immobilize the bitten limb using a splint or loose bandage.\n3. Remove rings, bracelets, or tight clothing near the bite area.\n4. Transport the victim immediately to the nearest medical facility with anti-venom.\n5. DO NOT cut the bite site, apply a tourniquet, or try to suck out the venom."
  },
  inference_time_ms: 38.45
};

// High-confidence Non-venomous prediction mock (Garter Snake - simulated)
const MOCK_SUCCESS_GARTER = {
  species: "garter snake",
  confidence: 0.9245,
  confidence_level: "High Confidence",
  is_uncertain: false,
  top_predictions: [
    { species: "garter snake", confidence: 0.9245 },
    { species: "cobra", confidence: 0.0512 },
    { species: "krait", confidence: 0.0243 }
  ],
  confidence_interpretation: "Confidence exceeds the 90.0% accuracy threshold determined during validation calibration.",
  prediction_reliability: "High",
  explanation_text: "Predicted garter snake with 92.45% confidence. The prediction is highly reliable.",
  uncertainty_reason: null,
  visualization_path: "C:\\Users\\susha\\OneDrive\\Desktop\\snake\\predictions\\gradcam_mock_garter.png",
  metadata: {
    common_name: "Garter Snake",
    scientific_name: "Thamnophis sirtalis",
    venomous: false,
    description: "Garter snakes are small to medium-sized, harmless snakes common across North America. They are characterized by longitudinal stripes and are generally found near water sources or grassy fields.",
    habitat: "Fields, woodlands, lawns, wetlands, and water edges.",
    first_aid: "1. Clean the bite area thoroughly with warm water and soap.\n2. Apply a mild antiseptic ointment and cover with a sterile bandage.\n3. Monitor for minor localized irritation or secondary infection.\n4. Reassure the victim that this species is non-venomous."
  },
  inference_time_ms: 32.12
};

// Low-confidence Uncertain prediction mock
const MOCK_UNCERTAIN = {
  species: "Uncertain",
  confidence: 0.4512,
  confidence_level: "Low Confidence",
  is_uncertain: true,
  top_predictions: [
    { species: "krait", confidence: 0.4512 },
    { species: "cobra", confidence: 0.3876 }
  ],
  confidence_interpretation: "Confidence is below the threshold required for a reliable classification.",
  prediction_reliability: "Low",
  explanation_text: "The classification is uncertain. The highest confidence class was krait (45.12%), which is below the safety threshold.",
  uncertainty_reason: "Prediction confidence 45.12% is below the calibrated threshold of 70.00% required for medium confidence.",
  visualization_path: null,
  metadata: {
    common_name: "Unknown / Uncertain Species",
    scientific_name: "N/A",
    venomous: true, // Defaulting safety-first to venomous
    description: "The neural network was unable to confidently identify the snake from the provided image. This could be due to poor lighting, obstruction, or an unsupported species.",
    habitat: "Unknown",
    first_aid: "1. TREAT ALL UNCERTAIN IDENTIFICATIONS AS VENOMOUS.\n2. Keep the victim calm and completely immobilized.\n3. Transport immediately to the nearest medical emergency center with anti-venom options.\n4. Do NOT attempt to handle or capture the snake."
  },
  inference_time_ms: 41.05
};

export const apiService = {
  /**
   * Predict snake species from an uploaded image file (Simulated API call)
   * @param {File} file - The uploaded image file
   * @returns {Promise<Object>} - Promise resolving to PredictionResponse Pydantic structure
   */
  async predictImage(file) {
    return new Promise((resolve, reject) => {
      setTimeout(() => {
        // Simple error mock if no file
        if (!file) {
          reject({
            error: {
              code: 400,
              message: "Bad Request: No file uploaded or image is corrupted."
            }
          });
          return;
        }

        // Cycle through mock response patterns based on file name or random criteria
        const name = file.name.toLowerCase();
        if (name.includes("cobra")) {
          resolve(MOCK_SUCCESS_COBRA);
        } else if (name.includes("garter") || name.includes("safe")) {
          resolve(MOCK_SUCCESS_GARTER);
        } else if (name.includes("uncertain") || name.includes("unknown") || Math.random() < 0.3) {
          resolve(MOCK_UNCERTAIN);
        } else {
          // Default to Cobra mock for general preview
          resolve(MOCK_SUCCESS_COBRA);
        }
      }, MOCK_DELAY);
    });
  },

  /**
   * Health status check
   * @returns {Promise<Object>}
   */
  async checkHealth() {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          api_status: "healthy",
          model_status: "loaded",
          version: "1.0.0",
          uptime_seconds: 3600.25,
          timestamp: new Date().toISOString(),
          status: "healthy",
          model_loaded: true
        });
      }, 300);
    });
  },

  /**
   * Fetch active model information
   * @returns {Promise<Object>}
   */
  async getModelInfo() {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve({
          model_name: "snake_classifier.keras",
          model_format: "Keras",
          supported_classes: ["cobra", "krait", "garter snake"],
          image_size: [224, 224],
          confidence_threshold: 0.60,
          model_loaded_status: true
        });
      }, 300);
    });
  }
};
