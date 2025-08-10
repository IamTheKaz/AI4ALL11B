import streamlit as st
from streamlit_camera_input_live import camera_input_live
import mediapipe as mp
import numpy as np
from PIL import Image
import cv2

st.title("🖐️ Hand Detection")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

img_data = camera_input_live()

if img_data:
    image = Image.open(img_data)
    image_np = np.array(image)

    results = hands.process(image_np)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image_np, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        st.success("Hand detected!")
    else:
        st.warning("No hand detected.")

    st.image(image_np, caption="Processed Frame", channels="RGB")