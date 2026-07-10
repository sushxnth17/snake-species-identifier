"""
Grad-CAM (Gradient-weighted Class Activation Mapping) module.
Generates heatmaps highlighting the regions in an image that influenced the model prediction,
supporting MobileNetV2 architecture.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
import matplotlib.cm as cm

class GradCAM:
    def __init__(self, model: tf.keras.Model, candidate_layer_name: str = "out_relu"):
        """
        Initializes the GradCAM generator.
        
        Args:
            model: Loaded tf.keras.Model.
            candidate_layer_name: The name of the convolutional layer to compute gradients against.
                                  For MobileNetV2, the final activation layer is typically 'out_relu'.
        """
        self.model = model
        
        # 1. Locate MobileNetV2 base model inside the functional architecture
        self.base_model = None
        for layer in model.layers:
            if "mobilenet" in layer.name or isinstance(layer, tf.keras.Model):
                self.base_model = layer
                break
                
        if self.base_model is None:
            raise ValueError("Could not find MobileNetV2 base model in the classification model.")
            
        # 2. Locate candidate layer in the base model
        try:
            self.conv_layer = self.base_model.get_layer(candidate_layer_name)
        except ValueError:
            # Fallback: search for the last 4D conv/activation layer in the base model
            self.conv_layer = None
            for layer in reversed(self.base_model.layers):
                if len(layer.output_shape) == 4:
                    self.conv_layer = layer
                    break
            if self.conv_layer is None:
                raise ValueError(f"Could not locate a 4D convolutional layer in the base model.")
                
        # 3. Build base sub-model mapping base inputs to conv layer output and base output
        self.base_sub_model = tf.keras.Model(
            inputs=self.base_model.inputs,
            outputs=[self.conv_layer.output, self.base_model.output]
        )
        
        # 4. Build classification head sub-model
        base_index = model.layers.index(self.base_model)
        head_input = tf.keras.Input(shape=self.base_model.output_shape[1:])
        x = head_input
        for layer in model.layers[base_index + 1:]:
            x = layer(x)
        self.head_model = tf.keras.Model(inputs=head_input, outputs=x)

    def generate_heatmap(self, preprocessed_img: tf.Tensor, class_idx: int) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for a given class index.
        
        Args:
            preprocessed_img: Preprocessed input image tensor batch, shape (1, H, W, 3).
            class_idx: Index of the target class to compute the activation mapping for.
            
        Returns:
            A 2D NumPy array representing the normalized heatmap of shape (H_conv, W_conv).
        """
        if len(preprocessed_img.shape) == 3:
            preprocessed_img = tf.expand_dims(preprocessed_img, axis=0)
            
        with tf.GradientTape() as tape:
            conv_outputs, base_outputs = self.base_sub_model(preprocessed_img)
            tape.watch(conv_outputs)
            preds = self.head_model(base_outputs)
            class_output = preds[:, class_idx]
            
        # Gradients of the predicted class with respect to the conv layer outputs
        grads = tape.gradient(class_output, conv_outputs)
        
        # Channel-wise mean of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map by its gradient weight, and sum over channels
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Apply ReLU to only keep positive influence regions, and normalize
        heatmap = tf.maximum(heatmap, 0.0)
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0.0:
            heatmap = heatmap / max_val
            
        return heatmap.numpy()

    def overlay_heatmap(self, heatmap: np.ndarray, original_img: Image.Image, alpha: float = 0.4) -> Image.Image:
        """
        Overlays the normalized 2D heatmap onto the original PIL image using matplotlib and PIL.
        
        Args:
            heatmap: Normalized 2D Grad-CAM heatmap.
            original_img: PIL Image object representing the original image.
            alpha: Semi-transparency blending factor [0.0, 1.0].
            
        Returns:
            A PIL Image representing the blended Grad-CAM overlay.
        """
        # Convert heatmap to PIL Image and resize to original image size
        heatmap_pil = Image.fromarray(np.uint8(255 * heatmap))
        heatmap_resized = heatmap_pil.resize(original_img.size, Image.Resampling.BILINEAR)
        
        # Retrieve colormap from matplotlib
        if hasattr(matplotlib, "colormaps"):
            colormap = matplotlib.colormaps["jet"]
        else:
            colormap = cm.get_cmap("jet")
            
        # Convert the resized single-channel heatmap to an RGB colormap image
        heatmap_color = np.uint8(255 * colormap(np.array(heatmap_resized) / 255.0)[:, :, :3])
        heatmap_color_img = Image.fromarray(heatmap_color)
        
        # Blend the original image and colormapped heatmap
        overlayed = Image.blend(original_img.convert("RGB"), heatmap_color_img, alpha)
        return overlayed

    def save_visualization(self, heatmap: np.ndarray, original_img: Image.Image, save_path: str, alpha: float = 0.4) -> None:
        """
        Combines heatmap overlay generation and saves the image to a file.
        
        Args:
            heatmap: Normalized 2D Grad-CAM heatmap.
            original_img: PIL Image object of the original image.
            save_path: Absolute destination path for the saved image.
            alpha: Semi-transparency blending factor.
        """
        overlayed = self.overlay_heatmap(heatmap, original_img, alpha)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        overlayed.save(save_path)
