import streamlit as st
from camera_input_live import camera_input_live

st.title("🎬 Live Webcam Feed")

image = camera_input_live()

if image is not None:
    st.image(image, caption="Current Frame from Webcam")
else:
    st.warning("No image received. Is your webcam active?")