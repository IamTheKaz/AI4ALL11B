import streamlit as st
from camera_input_live import camera_input_live
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Hand Detection", layout="centered")
st.title("🖐️ MediaPipe Hand Detection")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Capture image from webcam
img_data = camera_input_live()

if img_data:
    # Convert to PIL image and force RGB mode
    image = Image.open(img_data).convert("RGB")

    # Convert to NumPy array
    image_np = np.array(image)

    # Debug info
    st.write(f"Image shape: {image_np.shape}")
    st.write(f"Image dtype: {image_np.dtype}")

    # Check for 3-channel RGB
    if image_np.ndim == 3 and image_np.shape[2] == 3:
        # Run MediaPipe hand detection
        results = hands.process(image_np)

        # Draw landmarks if detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image_np, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            st.success("Hand detected!")
        else:
            st.warning("No hand detected.")

        # Display result
        st.image(image_np, caption="Processed Frame", channels="RGB")
    else:
        st.error("Image is not 3-channel RGB. Cannot process.")