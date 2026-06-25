"""
signLanguageMainCode.py
-----------------------
Real-time sign language recognition using a webcam and a pre-trained
Teachable Machine (Keras) model.

The script:
    1. Captures frames from the webcam
    2. Detects a hand using cvzone's HandDetector
    3. Crops and normalizes the hand image (same pipeline as DataCollection.py)
    4. Feeds the image into the trained model for classification
    5. Overlays the predicted sign label on the live video feed

Requirements:
    - A trained model at:  Model/keras_model.h5
    - Label file at:       Model/labels.txt
    (Both exported from Google Teachable Machine)

Usage:
    python signLanguageMainCode.py
    Press 'q' or close the window to stop.
"""

import cv2
import numpy as np
import math
import os
from cvzone.HandTrackingModule import HandDetector

# Force TensorFlow to use the legacy Keras API (required for Teachable Machine .h5 models)
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # Must be set before importing tensorflow
import tensorflow as tf


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_teachable_machine_model(model_path, labels_path):
    """
    Loads a Keras model exported from Google Teachable Machine.

    Args:
        model_path  (str): Path to the .h5 model file.
        labels_path (str): Path to the labels.txt file.

    Returns:
        model  : Loaded Keras model.
        labels : List of class label strings (e.g. ['A', 'B', 'C', ...]).
    """
    model = tf.keras.models.load_model(model_path, compile=False)

    # Each line in labels.txt is formatted as "0 A", "1 B", etc.
    # We strip the index prefix and keep only the label name.
    with open(labels_path, "r") as f:
        labels = [line.strip().split(" ", 1)[-1] for line in f.readlines()]

    return model, labels


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(model, img, labels):
    """
    Runs inference on a single hand image and returns the predicted label index.

    Args:
        model  : Loaded Keras model.
        img    : Hand image (numpy array, any size — will be resized internally).
        labels : List of class label strings.

    Returns:
        prediction (np.ndarray): Raw softmax probabilities for each class.
        index      (int)       : Index of the class with the highest probability.
    """
    # Resize to 224x224 — the input size expected by Teachable Machine models
    img_resized = cv2.resize(img, (224, 224))

    # Convert to float32 and normalize to [-1, 1] (Teachable Machine standard)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0

    # Add batch dimension: (224, 224, 3) → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Run the model and get class probabilities
    prediction = model.predict(img_array, verbose=0)[0]
    index = np.argmax(prediction)   # Class with the highest probability

    return prediction, index


# ── Setup ─────────────────────────────────────────────────────────────────────

cap = cv2.VideoCapture(0)           # Open the default webcam
detector = HandDetector(maxHands=1) # Detect at most one hand at a time

# Load the trained Teachable Machine model and its class labels
model, labels = load_teachable_machine_model(
    "Model/keras_model.h5",
    "Model/labels.txt"
)

offset = 20      # Pixel padding around the detected hand bounding box
imgSize = 300    # Size of the square canvas used for preprocessing

# ── Main Loop ─────────────────────────────────────────────────────────────────

while True:
    success, img = cap.read()           # Read a frame from the webcam
    imgOutput = img.copy()              # Keep a clean copy for drawing the output overlay
    hands, img = detector.findHands(img)  # Detect hands (modifies img with landmarks)

    if hands:
        hand = hands[0]                     # Use the first detected hand
        x, y, w, h = hand['bbox']          # Bounding box: top-left (x,y), width, height

        # Blank white square canvas for the normalized hand image
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand from the frame with padding
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # ── Aspect Ratio Correction ──────────────────────────────────────────
        # Resize the cropped hand into the white square without distortion

        aspectRatio = h / w

        if aspectRatio > 1:
            # Taller than wide → scale by height, center horizontally
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            # Wider than tall → scale by width, center vertically
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # ── Prediction ───────────────────────────────────────────────────────

        prediction, index = predict(model, imgWhite, labels)

        # ── Overlay Results on the Output Frame ──────────────────────────────

        # Purple filled rectangle as background for the label text
        cv2.rectangle(imgOutput,
                      (x - offset, y - offset - 50),       # Top-left of label box
                      (x - offset + 90, y - offset),       # Bottom-right of label box
                      (255, 0, 255), cv2.FILLED)

        # Display the predicted sign label above the hand
        cv2.putText(imgOutput, labels[index],
                    (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)

        # Draw a purple bounding box around the detected hand
        cv2.rectangle(imgOutput,
                      (x - offset, y - offset),            # Top-left corner
                      (x + w + offset, y + h + offset),    # Bottom-right corner
                      (255, 0, 255), 4)

        # Show intermediate windows for debugging
        cv2.imshow("ImageCrop", imgCrop)    # Raw cropped hand region
        cv2.imshow("ImageWhite", imgWhite)  # Preprocessed input sent to the model

    cv2.imshow("Image", imgOutput)  # Main output window with prediction overlay
    cv2.waitKey(1)