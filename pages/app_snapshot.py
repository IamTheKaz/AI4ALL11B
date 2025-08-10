import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from gtts import gTTS
import base64
import io
import tensorflow as tf
from PIL import Image
import tempfile
from nltk.corpus import words
import nltk

# 🧼 Hide sidebar and set page config (must be first Streamlit command)
st.set_page_config(page_title="ASL Snapshot Detector", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebarContent"] { display: none; }
    .css-1d391kg { display: none; }
    </style>
""", unsafe_allow_html=True)

# 📦 Setup
@st.cache_data
def load_nltk_words():
    try:
        return set(words.words())
    except LookupError:
        nltk.download('words')
        return set(words.words())

nltk_words = load_nltk_words()

# 🖐️ MediaPipe setup
try:
    mp_hands = mp.solutions.hands
    mp_hands_instance = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    HAND_CONNECTIONS = getattr(mp_hands, 'HAND_CONNECTIONS', None)
    if HAND_CONNECTIONS is None:
        st.warning("HAND_CONNECTIONS not detected. Landmark drawing will be disabled.")
except Exception as e:
    st.error(f"MediaPipe initialization failed: {e}")
    st.stop()

IMG_SIZE = 224  # Reduced from potential higher default if needed

# 🧠 Load model and class names
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("asl_model.h5")
    except Exception as e:
        st.error(f"Failed to load model: {e}. Ensure 'asl_model.h5' is in the directory.")
        st.stop()

model = load_model()
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank', 'Could not identify hand sign']

# 🔊 Speech synthesis
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        with io.BytesIO() as f:
            tts.write_to_fp(f)
            f.seek(0)
            return f.read()
    except Exception as e:
        st.error(f"Speech synthesis failed: {e}")
        return b''

def get_audio_download_link(audio):
    b64 = base64.b64encode(audio).decode()
    return f'<audio autoplay src="data:audio/mp3;base64,{b64}"/>'

# 🧠 Prediction logic with preprocessing
def predict_image(image):
    try:
        # Resize and preprocess image
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Hand detection
        results = mp_hands_instance.process(image_rgb)
        if not results.multi_hand_landmarks:
            return "Could not identify hand sign", 0.0, [("Could not identify hand sign", 1.0)]

        hand_landmarks = results.multi_hand_landmarks[0]
        if HAND_CONNECTIONS is not None:
            mp_drawing.draw_landmarks(image, hand_landmarks, HAND_CONNECTIONS)

        # Extract and normalize landmarks
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
        ]).flatten()  # 63 features (21 points * 3 coords)

        # Validate input shape
        if landmarks.shape[0] != 63:
            return "Could not identify hand sign", 0.0, [("Could not identify hand sign", 1.0)]

        input_array = landmarks.reshape(1, -1)
        if 'last_prediction_time' not in st.session_state or (time.time() - st.session_state.last_prediction_time) > 1.0:
            with st.spinner("Predicting..."):
                prediction_probs = model.predict(input_array, verbose=0)[0]
            st.session_state.last_prediction_time = time.time()
        else:
            return st.session_state.get('last_letter', "Could not identify hand sign"), 0.0, [("Could not identify hand sign", 1.0)]

        if len(prediction_probs) != len(CLASS_NAMES) - 1:  # Exclude 'Could not identify hand sign'
            return "Could not identify hand sign", 0.0, [("Could not identify hand sign", 1.0)]

        pred_index = np.argmax(prediction_probs)
        letter = CLASS_NAMES[pred_index] if pred_index < len(CLASS_NAMES) - 1 else "Could not identify hand sign"
        confidence = prediction_probs[pred_index] if pred_index < len(CLASS_NAMES) - 1 else 0.0
        st.session_state.last_letter = letter

        return letter, confidence, [(letter, confidence)]  # Only top prediction
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return "Could not identify hand sign", 0.0, [("Could not identify hand sign", 1.0)]

import time  # Added for timing

# 🚀 Main app
def main():
    st.title("🤟 Snapshot ASL Detector")
    st.markdown("Capture a photo using your webcam to predict ASL letters. Try forming the phrase **HELLO WORLD**!")

    # ✅ Mode-switch buttons
    st.markdown("### 🧭 Switch Mode:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎬 Live Mode"):
            st.switch_page("app.py")
    with col2:
        if st.button("🖼️ Upload Mode"):
            st.switch_page("pages/app_upload.py")

    # 🧠 Session state
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
    if 'last_prediction_time' not in st.session_state:
        st.session_state.last_prediction_time = 0.0

    # 📷 Webcam input
    st.markdown("---")
    st.subheader("Capture ASL Letter")
    webcam_image = st.camera_input("Click below to capture")

    if webcam_image:
        # Use context manager for temporary file
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
            tmp_file.write(webcam_image.getvalue())
            image = cv2.imread(tmp_file.name)

        if image is None:
            st.error("Failed to load image. Please try again.")
            return

        letter, confidence, _ = predict_image(image)

        st.image(image, caption=f"🖼️ Prediction: `{letter.upper()}`", channels="BGR")
        st.markdown(f"### ✅ Letter: `{letter.upper()}` — Confidence: `{confidence:.2f}`")

        spoken_text = "No hand sign detected" if letter == "Could not identify hand sign" else letter
        audio_buffer = speak_text(spoken_text)
        st.markdown(get_audio_download_link(audio_buffer), unsafe_allow_html=True)

        if letter not in ["Could not identify hand sign"]:
            st.session_state.sequence.append(letter)
            if len(st.session_state.sequence) > 50:
                st.session_state.sequence = st.session_state.sequence[-50:]

        current = ''.join([l.upper() for l in st.session_state.sequence])
        longest_word = max((word for j in range(len(current), 1, -1) 
                          for word in [current[-j:]] if word in nltk_words), 
                          key=len, default='')

        if longest_word:
            st.markdown(f"🗣 Detected Word: **{longest_word}**")
            st.markdown(get_audio_download_link(speak_text(longest_word)), unsafe_allow_html=True)

        target_sequence = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']
        if len(st.session_state.sequence) >= len(target_sequence):
            recent = st.session_state.sequence[-len(target_sequence):]
            if all(r == t for r, t in zip(recent, target_sequence)):
                st.success("🎉 Phrase Detected: HELLO WORLD")
                st.markdown(get_audio_download_link(speak_text("Hello World")), unsafe_allow_html=True)
                st.session_state.sequence = []

# 🏁 Entry point
if __name__ == "__main__":
    main()