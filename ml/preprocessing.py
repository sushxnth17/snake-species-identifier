"""
Preprocessing operations for images and datasets.
Handles image resizing, conversion to tensor, and MobileNetV2 normalization.
"""

from typing import Any, Tuple
import tensorflow as tf

def convert_to_tensor(image: Any) -> tf.Tensor:
    """
    Converts an input image (numpy array, PIL Image, etc.) to a TensorFlow Tensor.
    
    Args:
        image: Input image data.
        
    Returns:
        A TensorFlow Tensor representation of the image.
    """
    if not isinstance(image, tf.Tensor):
        return tf.convert_to_tensor(image, dtype=tf.float32)
    return image

def resize_image(image: tf.Tensor, target_size: Tuple[int, int]) -> tf.Tensor:
    """
    Resizes an image tensor to the target height and width.
    
    Args:
        image: Input image tensor.
        target_size: A tuple of (height, width).
        
    Returns:
        The resized image tensor.
    """
    return tf.image.resize(image, target_size)

def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalizes image pixel values to the range [-1, 1] as expected by MobileNetV2.
    
    MobileNetV2 expects input pixel values in the range [-1, 1].
    The function tf.keras.applications.mobilenet_v2.preprocess_input performs
    this normalization by scaling the pixel values (originally in [0, 255])
    accordingly: (pixel / 127.5) - 1.0.

    Args:
        image: Input image tensor.
        
    Returns:
        The normalized image tensor.
    """
    return tf.keras.applications.mobilenet_v2.preprocess_input(image)

def preprocess_image_dataset(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Applies MobileNetV2 preprocessing/normalization to a batch of images in a tf.data.Dataset.
    
    Args:
        image: A tensor representing the batch of input images.
        label: A tensor representing the batch of corresponding labels.
        
    Returns:
        A tuple of (normalized_image, label).
    """
    return normalize_image(image), label
