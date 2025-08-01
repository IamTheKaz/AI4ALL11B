import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from gtts import gTTS
import base64
import io
import tensorflow as tf
from camera_input_live import camera_input_live
from nltk.corpus import words
import nltk
import time

nltk.download('words')
nltk_words = set(words.words())

# Load trained model
model = tf.keras.models.load_model("asl_model.h5")

# Define class names: A-Z + 'blank'
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
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

# 🧠 Stability check
def is_stable(prev, curr, threshold=0.01):
    if prev is None or curr is None:
        return False
    diffs = np.abs(np.array(prev) - np.array(curr))
    return np.max(diffs) < threshold

# 🧠 Prediction function
def predict_landmarks(landmarks):
    input_array = np.array(landmarks).reshape(1, -1)
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
st.write("Use your webcam to detect ASL letters and build the phrase 'HELLO WORLD'.")

if 'sequence' not in st.session_state:
    st.session_state.sequence = []
if 'live_mode' not in st.session_state:
    st.session_state.live_mode = False
    st.session_state.prev_landmarks = None

col1, col2 = st.columns(2)
if col1.button("▶️ Start Live Detection"):
    st.session_state.live_mode = True
if col2.button("⏹️ Stop"):
    st.session_state.live_mode = False

frame_placeholder = st.empty()
info_placeholder = st.empty()

while st.session_state.live_mode:
    image = camera_input_live()
    if image is None:
        frame_placeholder.warning("No image received. Is your webcam active?")
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        x_vals = [lm.x for lm in hand_landmarks.landmark]
        y_vals = [lm.y for lm in hand_landmarks.landmark]
        z_vals = [lm.z for lm in hand_landmarks.landmark]
        curr_landmarks = x_vals + y_vals + z_vals

        if is_stable(st.session_state.prev_landmarks, curr_landmarks):
            letter, confidence, top_3 = predict_landmarks(curr_landmarks)
            frame_placeholder.image(image, caption=f"🖼️ Stable Sign: `{letter}`", channels="BGR")

            info_placeholder.markdown(f"### ✅ Detected Letter: `{letter.upper()}`")
            info_placeholder.markdown(f"**Confidence:** `{confidence:.2f}`")
            info_placeholder.markdown("#### 🔝 Top 3 Predictions:")
            for i, (char, conf) in enumerate(top_3, 1):
                info_placeholder.write(f"{i}. `{char}` — `{conf:.2f}`")

            spoken_text = speak_text_input(letter)
            info_placeholder.markdown(get_audio_download_link(speak_text(spoken_text)), unsafe_allow_html=True)

            if letter != "blank":
                st.session_state.sequence.append(letter)

            current = ''.join(st.session_state.sequence).upper()
            longest_word = ''
            for j in range(len(current), 1, -1):
                word = current[-j:]
                if word in nltk_words and len(word) > len(longest_word):
                    longest_word = word

            if longest_word:
                info_placeholder.markdown(f"🗣 Detected Word: **{longest_word}**")
                info_placeholder.markdown(get_audio_download_link(speak_text(longest_word)), unsafe_allow_html=True)

            target_sequence = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']
            if len(st.session_state.sequence) >= len(target_sequence):
                recent = st.session_state.sequence[-len(target_sequence):]
                if all(r == t for r, t in zip(recent, target_sequence)):
                    info_placeholder.success("🎉 Phrase Detected: HELLO WORLD")
                    info_placeholder.markdown(get_audio_download_link(speak_text("Hello World")), unsafe_allow_html=True)
                    st.session_state.sequence = []

        else:
            frame_placeholder.image(image, caption="🖼️ Waiting for stable sign...", channels="BGR")

        st.session_state.prev_landmarks = curr_landmarks
    else:
        frame_placeholder.image(image, caption="🖼️ No hand detected", channels="BGR")

    time.sleep(1.5)  # Delay to avoid rapid looping

# --- Mode Switching Section ---
st.markdown("---")
st.markdown("#### 🧭 Try Alternate Input Modes:")
col1, col2 = st.columns(2)
with col1:
    if st.button("📤 Upload Version"):
        st.switch_page("pages/mediapip_upload_app.py")
with col2:
    if st.button("📷 Snapshot Version"):
        st.switch_page("pages/app_snapshot.py")