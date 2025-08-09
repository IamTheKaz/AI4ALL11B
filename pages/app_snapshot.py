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
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
IMG_SIZE = 224  # Match training image size

# 🧠 Load model and class names
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("asl_model.h5")

model = load_model()
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank', 'Could not identify hand sign']

# 🔊 Speech synthesis
def speak_text(text):
    tts = gTTS(text=text, lang='en')
    with io.BytesIO() as f:
        tts.write_to_fp(f)
        f.seek(0)
        return f.read()

def get_audio_download_link(audio):
    b64 = base64.b64encode(audio).decode()
    return f'<audio autoplay src="data:audio/mp3;base64,{b64}"/>'

# 🧠 Prediction logic with preprocessing
def predict_image(image):
    # Resize and preprocess image
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Hand detection
    results = hands.process(image_rgb)
    if not results.multi_hand_landmarks:
        return "Could not identify hand sign", 0.0, [("Could not identify hand sign", 1.0)]

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Extract and normalize landmarks
    landmarks = np.array([
        [lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark
    ]).flatten()  # 63 features (21 points * 3 coords)

    # Validate input shape
    if landmarks.shape[0] != 63:
        return "Could not identify hand sign",