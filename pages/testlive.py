import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from gtts import gTTS
import base64
import io
import tensorflow as tf
from camera_input_live import camera_input_live
from PIL import Image
from nltk.corpus import words
import nltk

nltk.download('words')
nltk_words = set(words.words())

# Load trained model
model = tf.keras.models.load_model("asl_model.h5")

# Define class names: A-Z + 'blank'
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
    if isinstance(image, Image.Image):
        image = np.array(image)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)]

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    x_vals = [lm.x for lm in hand_landmarks.landmark]
    y_vals = [lm.y for lm in hand_landmarks.landmark]
    z_vals = [lm.z for lm in hand_landmarks.landmark]
    input_array = np.array(x_vals + y_vals + z_vals).reshape(1, -1)

    prediction_probs = model.predict(input_array)[0]
    pred_index = np.argmax(prediction_probs)
    if pred_index >= len(CLASS_NAMES):
        return "blank", 0.0, [("blank", 1.0)]

    prediction = CLASS_NAMES[pred_index]
    confidence = prediction_probs[pred_index]
    top_3 = [(CLASS_NAMES[i], prediction_probs[i]) for i in np.argsort(prediction_probs)[-3:][::-1]]

    return prediction, confidence, top_3

# 🎯 Streamlit UI
st.title("🤳 Live ASL Detection")
st.write("Click the button below to capture a frame from your webcam and predict the ASL letter.")

if 'sequence' not in st.session_state:
    st.session_state.sequence = []

if st.button("📸 Capture Frame"):
    image = camera_input_live()

    if image is not None:
        st.image(image, caption="📷 Captured Frame", channels="RGB")

        letter, confidence, top_3 = predict_image(image)

        st.markdown(f"### ✅ Detected Letter: `{letter.upper()}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}`")

        st.markdown("#### 🔝 Top 3 Predictions:")
        for i, (char, conf) in enumerate(top_3, 1):
            st.write(f"{i}. `{char}` — `{conf:.2f}`")

        spoken_text = speak_text_input(letter)
        st.markdown(get_audio_download_link(speak_text(spoken_text)), unsafe_allow_html=True)

        if letter != "blank":
            st.session_state.sequence.append(letter)

        current = ''.join(st.session_state.sequence).upper()
        longest_word = ''
        for j in range(len(current), 1, -1):
            word = current[-j:]
            if word in nltk_words and len(word) > len(longest_word):
                longest_word = word

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
    else:
        st.warning("No image received. Is your webcam active?")