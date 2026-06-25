"""
DataCollection.py
-----------------
Captures hand gesture images from a webcam and saves them to a local folder.
These images are later used to train a sign language classification model
via Google Teachable Machine.

Usage:
    - Run the script
    - Show a hand sign to the camera
    - Press 's' to save the current frame
    - Repeat until you have enough samples for each letter/sign
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# ── Camera & Detector Setup ──────────────────────────────────────────────────

cap = cv2.VideoCapture(0)           # Open the default webcam (index 0)
detector = HandDetector(maxHands=1) # Detect at most one hand at a time

# ── Configuration ────────────────────────────────────────────────────────────

offset = 20      # Pixel padding around the detected hand bounding box
imgSize = 300    # Size of the output square image (300x300 pixels)

folder = "Data/C"   # Destination folder — change "C" to the current sign/letter
counter = 0         # Counts how many images have been saved in this session

# ── Main Loop ────────────────────────────────────────────────────────────────

while True:
    success, img = cap.read()               # Read a frame from the webcam
    hands, img = detector.findHands(img)    # Detect hands and draw landmarks on img

    if hands:
        hand = hands[0]                         # Take the first (and only) detected hand
        x, y, w, h = hand['bbox']              # Bounding box: top-left (x,y), width, height

        # Create a blank white square canvas to place the hand image on
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region from the frame, with padding (offset) on all sides
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # ── Aspect Ratio Correction ──────────────────────────────────────────
        # Resize the cropped hand to fit inside the square canvas
        # while preserving its aspect ratio (avoid stretching)

        aspectRatio = h / w  # Compare hand height to width

        if aspectRatio > 1:
            # Hand is taller than it is wide → scale by height
            k = imgSize / h
            wCal = math.ceil(k * w)                         # Scaled width
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)         # Center horizontally
            imgWhite[:, wGap:wCal + wGap] = imgResize      # Paste onto white canvas
        else:
            # Hand is wider than it is tall → scale by width
            k = imgSize / w
            hCal = math.ceil(k * h)                         # Scaled height
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)         # Center vertically
            imgWhite[hGap:hCal + hGap, :] = imgResize      # Paste onto white canvas

        # Show intermediate windows for visual feedback
        cv2.imshow("ImageCrop", imgCrop)    # Raw cropped hand
        cv2.imshow("ImageWhite", imgWhite)  # Processed square image (what gets saved)

    cv2.imshow("Image", img)    # Live webcam feed with hand landmarks

    key = cv2.waitKey(1)

    # Press 's' to save the current processed image to the dataset folder
    if key == ord("s"):
        counter += 1
        folder = "folder"  # TODO: update this to match the sign you're collecting e.g. "Data/C"
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)  # Unique filename via timestamp
        print(f"Saved image #{counter}")