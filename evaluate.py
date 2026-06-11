from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import kagglehub
import os
import cv2

IMG_SIZE = 128

model = load_model("models/best_tb_model.keras")

path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
base_path = os.path.join(path, "TB_Chest_Radiography_Database")

categories = ["Tuberculosis", "Normal"]

data, labels = [], []

for category in categories:
    folder = os.path.join(base_path, category)
    label = categories.index(category)

    for img in os.listdir(folder):
        try:
            img_path = os.path.join(folder, img)
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

            data.append(image)
            labels.append(label)
        except:
            pass

data = np.array(data) / 255.0
data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
labels = np.array(labels)

# 🔥 Lower threshold (IMPORTANT)
preds = (model.predict(data) > 0.4).astype("int32")

print("Classification Report:\n")
print(classification_report(labels, preds))

print("\nConfusion Matrix:\n")
print(confusion_matrix(labels, preds))