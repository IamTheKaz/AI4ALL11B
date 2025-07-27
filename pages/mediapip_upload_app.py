import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from gtts import gTTS
import base64
import io
import tensorflow as tf

# Load your trained TensorFlow model
model = tf.keras.models.load_model("asl_model.h5")

# Define class names: A-Z and 'blank' only
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 🔊 Speech functions
def speak_text_input(letter):
    return "No hand sign detected" if letter == "blank" else letter

def speak_text(text):
    tts = gTTS(text=text)
    with io.BytesIO() as f:
        tts.write_to_fp(f)
        f.seek(0)
        return f.read()

def get_audio_download_link(audio):
    b64 = base64.b64encode(audio).decode()
    return f'<audio autoplay src="data:audio/mp3;base64,{b64}"/>'

# 🧠 Prediction function
def predict_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)]

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Extract and reshape landmarks
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    input_array = np.array(landmarks).reshape(1, -1)
    prediction_probs = model.predict(input_array)[0]
    pred_index = np.argmax(prediction_probs)
    prediction = CLASS_NAMES[pred_index]
    confidence = round(prediction_probs[pred_index], 2)

    return prediction, confidence, list(zip(CLASS_NAMES, prediction_probs))

# 🎯 Streamlit UI
st.title("ASL Hand Sign Detection")

uploaded_file = st.file_uploader("Upload an image of a hand sign", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    letter, confidence, probs = predict_image(image)

    st.image(image, caption=f"Prediction: {letter}", channels="BGR")
    st.write(f"Confidence: {confidence}")
    st.write(f"Probabilities: {probs}")

    spoken_text = speak_text_input(letter)
    st.markdown(get_audio_download_link(speak_text(spoken_text)), unsafe_allow_html=True)

# --- Mode Switching Section ---
st.markdown("---")
st.markdown("#### 🧭 Try Alternate Input Modes:")
col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Snapshot Version"):
        st.switch_page("pages/app_snapshot.py")
with col2:
    if st.button("🤳 Live Webcam Version"):
        st.switch_page("app.py")
