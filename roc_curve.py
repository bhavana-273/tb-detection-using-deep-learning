import numpy as np
import cv2
import os
import kagglehub
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.models import load_model

IMG_SIZE = 128

# Load model
model = load_model("models/best_tb_model.keras")

# Load dataset
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

# Probabilities
probs = model.predict(data).ravel()

# ROC
fpr, tpr, thresholds = roc_curve(labels, probs)
roc_auc = auc(fpr, tpr)

print("AUC:", roc_auc)

# Plot
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Optional: pick a better threshold (maximize TPR with reasonable FPR)
best_idx = (tpr - fpr).argmax()
print("Suggested threshold:", thresholds[best_idx])