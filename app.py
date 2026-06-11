import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 128

model = load_model("models/best_tb_model.keras")

st.title("TB Detection")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image")

    # PREPROCESS
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (IMG_SIZE, IMG_SIZE))
    img_resized = img_resized / 255.0
    img_resized = img_resized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # PREDICT
    pred = model.predict(img_resized)[0][0]

    st.write(f"Prediction value: {pred:.4f}")

    # CORRECT LOGIC
    if pred < 0.5:
        st.error("⚠️ TB Detected")
    else:
        st.success("✅ Normal")