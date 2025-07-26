import streamlit as st
from streamlit_camera_input_live import camera_input_live
import numpy as np
import cv2
from PIL import Image
import io

st.title("Webcam Test — Streamlit Camera Input Live")

# Capture frame from webcam
frame = camera_input_live()

if frame is not None:
    st.success("Frame received successfully!")

    # Display raw image
    pil_img = Image.open(io.BytesIO(frame.getvalue()))
    st.image(pil_img, caption="Captured Image")

    # Convert to OpenCV format
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Resize to model-compatible dimensions (e.g., 224x224)
    resized = cv2.resize(cv_img, (224, 224))
    st.text(f"Resized shape: {resized.shape}")

else:
    st.warning("Webcam frame not available. Try refreshing or switching browsers.")