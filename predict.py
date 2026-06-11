import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -------------------------------
# Config
# -------------------------------
IMG_SIZE = 128
MODEL_PATH = "models/best_tb_model.keras"

# Load model
model = load_model(MODEL_PATH)

# -------------------------------
# Prediction function
# -------------------------------
def predict_image(img_path, threshold=0.6):
    try:
        # Read image
        img = cv2.imread(img_path)

        if img is None:
            return "Error: Image not found or invalid format"

        # Preprocessing (same as training)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0
        img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        # Predict
        pred = model.predict(img)[0][0]

        print(f"Raw prediction value: {pred:.4f}")

        # -------------------------------
        # CORRECT LOGIC
        # -------------------------------
        # Model outputs probability of NORMAL (class 1)
        # Lower value → TB
        if pred < threshold:
            return f"⚠️ TB Detected (confidence: {1 - pred:.2f})"
        else:
            return f"✅ Normal (confidence: {pred:.2f})"

    except Exception as e:
        return f"Error: {str(e)}"


# -------------------------------
# Run from terminal
# -------------------------------
if __name__ == "__main__":
    img_path = input("Enter image path: ")
    result = predict_image(img_path)
    print(result)