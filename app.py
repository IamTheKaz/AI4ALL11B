import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import base64
import io
import time
from PIL import Image
import nltk
from nltk.corpus import words
import gc
from camera_input_live import camera_input_live

# 🧼 Hide sidebar and set page config
st.set_page_config(page_title="ASL Live Detector", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebarContent"] { display: none; }
    .css-1d391kg { display: none; }
    </style>
""", unsafe_allow_html=True)

# 📦 Ensure NLTK words are available
try:
    nltk_words = set(words.words())
except LookupError:
    nltk.download('words')
    nltk_words = set(words.words())

# 🖐️ MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 🧠 Load model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("asl_model.h5")

model = load_model()
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank', 'fallback']

# 🔊 Speech synthesis
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
def extract_landmark_array(hand_landmarks):
    return np.array([lm.x for lm in hand_landmarks.landmark] +
                    [lm.y for lm in hand_landmarks.landmark] +
                    [lm.z for lm in hand_landmarks.landmark])

def predict_image(image):
    try:
        image_np = np.array(image.convert("RGB"))
        image_np = image_np.astype(np.uint8)
        results = hands.process(image_np)

        if not results.multi_hand_landmarks:
            return "blank", 0.0, [("blank", 1.0)], None

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image_np, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        landmark_array = extract_landmark_array(hand_landmarks).reshape(1, -1)
        prediction_probs = model.predict(landmark_array)[0]

        if len(prediction_probs) != len(CLASS_NAMES):
            return "fallback", 0.0, [("fallback", 1.0)], extract_landmark_array(hand_landmarks)

        pred_index = np.argmax(prediction_probs)
        prediction = CLASS_NAMES[pred_index]
        confidence = prediction_probs[pred_index]
        top_3 = [(CLASS_NAMES[i], prediction_probs[i]) for i in np.argsort(prediction_probs)[-3:][::-1]]

        return prediction, confidence, top_3, extract_landmark_array(hand_landmarks)

    except Exception:
        gc.collect()
        return "fallback", 0.0, [("fallback", 1.0)], None

# 🧠 Stability check
def is_stable(current, previous, threshold=0.01):
    if previous is None:
        return False
    delta = np.linalg.norm(current - previous)
    return delta < threshold

# 🚀 Main app
def main():
    st.title("🤟 Auto-Capture ASL Detector")
    st.markdown("This app automatically captures and predicts ASL signs when your hand is stable.")

    if 'sequence' not in st.session_state:
        st.session_state.sequence = []

    st.markdown("### 🎬 Live Detection Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Live Predictions"):
            st.session_state.start_stream = True
    with col2:
        if st.button("⏹️ Stop Live Predictions"):
            st.session_state.start_stream = False
            st.session_state.sequence = []
            st.info("Live prediction stopped. Click 'Start' to resume.")
            st.empty().empty()

    image_placeholder = st.empty()
    status_placeholder = st.empty()
    prev_landmarks = None

    if st.session_state.get('start_stream', False):
        image = camera_input_live()

        if image is None:
            status_placeholder.warning("⚠️ No image received from camera. Please check your device.")
            return

        try:
            if isinstance(image, Image.Image):
                pil_image = image
            elif hasattr(image, "read"):
                pil_image = Image.open(image)
            else:
                status_placeholder.warning("⚠️ Unsupported image format.")
                return

            image_np = np.array(pil_image.convert("RGB"))

            if image_np.size == 0 or image_np.ndim != 3:
                status_placeholder.warning("⚠️ Invalid image data. Skipping frame.")
                return

            image_placeholder.image(image_np, caption="Live Preview", channels="RGB")

            letter, confidence, top_3, current_landmarks = predict_image(pil_image)

            if current_landmarks is not None:
                stable = is_stable(current_landmarks, prev_landmarks)
                prev_landmarks = current_landmarks.copy()

                if stable:
                    if letter not in ["blank", "fallback"]:
                        st.session_state.sequence.append(letter)
                        if len(st.session_state.sequence) > 50:
                            st.session_state.sequence = st.session_state.sequence[-50:]

                        status_placeholder.success(f"✋ Stable hand detected — predicted: `{letter}` ({confidence:.2f})")

                        st.markdown("#### 🔝 Top 3 Predictions:")
                        for i, (char, conf) in enumerate(top_3, 1):
                            st.write(f"{i}. `{char}` — `{conf:.2f}`")

                        st.markdown(get_audio_download_link(speak_text(letter)), unsafe_allow_html=True)

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
                        status_placeholder.warning("⚠️ Could not detect a hand sign. Try adjusting your hand position or lighting.")
                else:
                    status_placeholder.info("✋ Waiting for stable hand position...")
            else:
                status_placeholder.warning("⚠️ Could not detect a hand sign. Try adjusting your hand position or lighting.")

        except Exception:
            gc.collect()

    # ✅ Mode-switch buttons (moved below camera)
    st.markdown("---")
    st.markdown("### 🧭 Switch Mode:")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("📸 Snapshot Mode"):
            st.switch_page("pages/app_snapshot.py")
    with col4:
        if st.button("🖼️ Upload Mode"):
            st.switch_page("pages/app_upload.py")

# 🏁 Entry point
if __name__ == '__main__':
    main()