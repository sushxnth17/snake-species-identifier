/**
 * API Service for Snakify.
 * Handles communications with the FastAPI backend.
 */

// Configurable request timeout constant (15 seconds)
const REQUEST_TIMEOUT_MS = 15000;

/**
 * Normalizes backend error schemas and network exceptions into a frontend-friendly error format.
 * Hides raw Python tracebacks, internal filepaths, and unformatted system errors.
 *
 * @param {Object} errorObj - The raw error wrapper.
 * @returns {Object} - { message: string, requestId: string|null, status: number }
 */
export function normalizeApiError(errorObj) {
  // 1. Handle timeouts or network failures
  if (errorObj.status === 0 || errorObj.isTimeout) {
    return {
      message: errorObj.message,
      requestId: errorObj.requestId || null,
      status: errorObj.status
    };
  }

  const status = errorObj.status;
  const errorData = errorObj.data;
  const requestId = errorObj.requestId;

  let message = "An unexpected error occurred. Please try again.";

  // 2. Parse standard FastAPI structured error: {"error": {"code": 400, "message": "..."}}
  if (errorData && errorData.error && errorData.error.message) {
    message = errorData.error.message;
  } else if (errorData && errorData.detail) {
    // 3. Fallback to standard FastAPI HTTPException schemas
    if (typeof errorData.detail === 'string') {
      message = errorData.detail;
    } else if (Array.isArray(errorData.detail)) {
      // 4. Handle Pydantic field validation lists
      message = errorData.detail
        .map(err => `${err.loc.join('.')}: ${err.msg}`)
        .join('; ');
    }
  }

  // 5. Apply status-code specific overrides for user friendliness
  if (status === 429) {
    message = "Too many requests. Wait a moment and try again.";
  } else if (status === 500) {
    message = "An unexpected server error occurred. Please try again later.";
  } else if (status === 503) {
    message = message || "Service Unavailable: Required prediction resources are missing.";
  }

  return {
    message,
    requestId,
    status
  };
}

export const apiService = {
  /**
   * Post snake image to the FastAPI /predict route.
   * Uses AbortController to enforce request timeouts.
   *
   * @param {File} file - Valid selected snake image file
   * @returns {Promise<Object>} - Resolves to success response payload
   */
  async predictImage(file) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

    try {
      // Get base API URL from Vite environment, fallback to the same-origin '/api' proxy
      const baseUrl = import.meta.env.VITE_API_BASE_URL || '/api';
      const sanitizedBaseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
      const url = `${sanitizedBaseUrl}/predict`;

      // Build multipart form data
      const formData = new FormData();
      formData.append('file', file);

      // Execute request with signal
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);

      // Extract Request ID header
      const requestId = response.headers.get("X-Request-ID");

      if (!response.ok) {
        let errorData = null;
        try {
          errorData = await response.json();
        } catch (e) {
          // Response is not JSON
        }
        
        throw {
          status: response.status,
          data: errorData,
          requestId
        };
      }

      const successData = await response.json();
      return successData;

    } catch (error) {
      clearTimeout(timeoutId);
      
      // Handle timeout abortion
      if (error.name === 'AbortError') {
        throw {
          status: 408,
          message: "The analysis took too long. Please try again.",
          isTimeout: true
        };
      }
      
      // If it is a normalized error we threw from response.ok check
      if (error.status !== undefined) {
        throw error;
      }

      // Handle general connection failures (backend offline)
      throw {
        status: 0,
        message: "Unable to connect to the prediction service. Make sure the backend is running and try again."
      };
    }
  },

  /**
   * Health status check stub
   * @returns {Promise<Object>}
   */
  async checkHealth() {
    try {
      const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
      const sanitizedBaseUrl = baseUrl.endsWith('/') ? baseUrl.slice(0, -1) : baseUrl;
      const response = await fetch(`${sanitizedBaseUrl}/health`);
      if (response.ok) return await response.json();
    } catch (e) {
      // Fallback
    }
    return { api_status: "unknown", model_loaded: false };
  }
};
