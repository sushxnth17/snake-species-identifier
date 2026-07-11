/**
 * Validation utilities mirroring FastAPI backend limits in backend/validation.py and backend/config.py
 */

export const ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
export const MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024; // 5MB

/**
 * Validates an uploaded file format and size
 * @param {File} file - Selected browser File object
 * @returns {Object} - { isValid: boolean, error: string | null }
 */
export function validateImageFile(file) {
  if (!file) {
    return { isValid: false, error: "No file selected." };
  }

  // Enforce supported formats
  if (!ALLOWED_MIME_TYPES.includes(file.type)) {
    return {
      isValid: false,
      error: "Please choose a JPG, PNG, or WebP image."
    };
  }

  // Enforce max size limit (5MB)
  if (file.size > MAX_UPLOAD_SIZE_BYTES) {
    return {
      isValid: false,
      error: "Image is too large. Maximum size is 5 MB."
    };
  }

  return { isValid: true, error: null };
}
