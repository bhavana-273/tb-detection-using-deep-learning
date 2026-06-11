import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

IMG_SIZE = 128
MODEL_PATH = "models/best_tb_model.keras"

model = load_model(MODEL_PATH)

# Get last conv layer name automatically
def get_last_conv_layer(m):
    for layer in reversed(m.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")

last_conv_layer_name = get_last_conv_layer(model)

# Grad-CAM
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

# Load & preprocess image
def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img / 255.0
    img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img, img_input

# Overlay heatmap
def overlay_heatmap(orig_img, heatmap):
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR), 0.6, heatmap, 0.4, 0)
    return overlay

# Test with an image path
img_path = "test.jpg"   # <-- put any X-ray here

orig, inp = load_image(img_path)
heatmap = make_gradcam_heatmap(inp, model, last_conv_layer_name)
overlay = overlay_heatmap(orig, heatmap)

plt.figure(figsize=(10,4))
plt.subplot(1,3,1); plt.title("Original"); plt.imshow(orig, cmap="gray"); plt.axis("off")
plt.subplot(1,3,2); plt.title("Heatmap"); plt.imshow(heatmap, cmap="jet"); plt.axis("off")
plt.subplot(1,3,3); plt.title("Grad-CAM"); plt.imshow(overlay); plt.axis("off")
plt.show()