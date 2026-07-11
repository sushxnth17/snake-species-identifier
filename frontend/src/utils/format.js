/**
 * Formatting helpers
 */

/**
 * Formats a file size in bytes to a human-readable string (e.g. "2.4 MB")
 * @param {number} bytes - Raw file size in bytes
 * @param {number} decimals - Precision decimals
 * @returns {string}
 */
export function formatBytes(bytes, decimals = 1) {
  if (!bytes || bytes === 0) return '0 Bytes';
  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
