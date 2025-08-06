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
import gc

# 📦 Setup
try:
    nltk_words = set(words.words())
except LookupError:
    nltk.download('words')
    nltk_words = set(words.words())

# 🧼 Hide sidebar and set page config
st.set_page_config(page_title="ASL Snapshot Detector", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebarContent"] { display: none; }
    .css-1d391kg { display: none; }
    </style>
""", unsafe_allow_html=True)

# 🖐️ MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 🧠 Load model and class names
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("asl_model.h5")

model = load_model()
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank', 'fallback']

# 🔊 Speech synthesis
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

# 🧠 Prediction logic
def predict_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)]

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    input_array = np.array(
        [lm.x for lm in hand_landmarks.landmark] +
        [lm.y for lm in hand_landmarks.landmark] +
        [lm.z for lm in hand_landmarks.landmark]
    ).reshape(1, -1)

    prediction_probs = model.predict(input_array)[0]
    pred_index = np.argmax(prediction_probs)

    if pred_index >= len(CLASS_NAMES):
        return "fallback", 0.0, [("fallback", 1.0)]

    letter = CLASS_NAMES[pred_index]
    confidence = prediction_probs[pred_index]
    top_3 = [(CLASS_NAMES[i], prediction_probs[i]) for i in np.argsort(prediction_probs)[-3:][::-1]]

    return letter, confidence, top_3

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

    # 📷 Webcam input
    st.markdown("---")
    st.subheader("Capture ASL Letter")
    webcam_image = st.camera_input("Click below to capture")

    if webcam_image:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(webcam_image.getvalue())
            tmp_file_path = tmp_file.name

        image = cv2.imread(tmp_file_path)
        letter, confidence, top_3 = predict_image(image)

        st.image(image, caption=f"🖼️ Prediction: `{letter.upper()}`", channels="BGR")
        st.markdown(f"### ✅ Letter: `{letter.upper()}` — Confidence: `{confidence:.2f}`")

        st.markdown("#### 🔝 Top 3 Predictions:")
        for i, (char, conf) in enumerate(top_3, 1):
            st.write(f"{i}. `{char}` — `{conf:.2f}`")

        spoken_text = speak_text_input(letter)
        audio_buffer = speak_text(spoken_text)
        st.markdown(get_audio_download_link(audio_buffer), unsafe_allow_html=True)

        if letter not in ["blank", "fallback"]:
            st.session_state.sequence.append(letter)
            if len(st.session_state.sequence) > 50:
                st.session_state.sequence = st.session_state.sequence[-50:]

        current = ''.join([l.upper() for l in st.session_state.sequence])
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

        # 🧹 Clean up memory
        del image, letter, confidence, top_3, audio_buffer
        gc.collect()

# 🏁 Entry point
if __name__ == "__main__":
    main()