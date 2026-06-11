import kagglehub
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from collections import Counter

IMG_SIZE = 128

# -------------------------------
# Download dataset
# -------------------------------
path = kagglehub.dataset_download("tawsifurrahman/tuberculosis-tb-chest-xray-dataset")
base_path = os.path.join(path, "TB_Chest_Radiography_Database")

print("Dataset path:", base_path)
print("Folders:", os.listdir(base_path))

categories = ["Tuberculosis", "Normal"]

data, labels = [], []

# -------------------------------
# Load data
# -------------------------------
for category in categories:
    folder = os.path.join(base_path, category)
    label = categories.index(category)

    print("Loading:", folder)

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

print("Total images:", len(data))
print("Class distribution:", Counter(labels))

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# -------------------------------
# Data augmentation
# -------------------------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

datagen.fit(X_train)

# -------------------------------
# Model
# -------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Callbacks
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "models/best_tb_model.keras",
    monitor='val_accuracy',
    save_best_only=True
)

# -------------------------------
# Class Weights (IMPORTANT FIX)
# -------------------------------
class_weights = {
    0: 2.0,   # Tuberculosis (increase importance)
    1: 1.0    # Normal
}

# -------------------------------
# Train
# -------------------------------
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    epochs=10,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint],
    class_weight=class_weights
)

# -------------------------------
# Save model
# -------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/tb_model.keras")

print("✅ Model saved!")