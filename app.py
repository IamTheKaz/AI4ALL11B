import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from gtts import gTTS
import base64
import io
from PIL import Image
import nltk
from nltk.corpus import words
import gc
from camera_input_live import camera_input_live

# 🧼 Page config and sidebar cleanup
st.set_page_config(page_title="ASL Live Detector", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    </style>
""", unsafe_allow_html=True)

# 📦 Load NLTK words once
@st.cache_resource
def load_nltk_words():
    try:
        return set(words.words())
    except LookupError:
        nltk.download('words')
        return set(words.words())

nltk_words = load_nltk_words()

# 🖐️ MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 🧠 Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("asl_model.h5")

model = load_model()
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank', 'fallback']

# 🔊 Text-to-speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    b64 = base64.b64encode(mp3_fp.read()).decode()
    audio_html = f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)

# 📊 Predict ASL letter from landmarks
def predict_letter(landmarks):
    input_data = np.array(landmarks).reshape(1, -1)
    prediction = model.predict(input_data, verbose=0)
    confidence = np.max(prediction)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    if confidence < 0.7:
        predicted_class = 'fallback'
    print(f"[DEBUG] Prediction: {predicted_class}, Confidence: {confidence:.2f}")
    return predicted_class

# 🔁 Stability filter
def get_stable_prediction(predictions, threshold=15):
    if len(predictions) < threshold:
        return None
    recent = predictions[-threshold:]
    most_common = max(set(recent), key=recent.count)
    if recent.count(most_common) > threshold * 0.8:
        return most_common
    return None

# 🧠 Session state setup
for key in ['predictions', 'sentence', 'start_stream']:
    if key not in st.session_state:
        st.session_state[key] = [] if key == 'predictions' else "" if key == 'sentence' else False

# 🎬 Live Detection Controls
st.markdown("### 🎬 Live Detection Controls")
col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start Live Predictions"):
        st.session_state.start_stream = True
with col2:
    if st.button("⏹️ Stop Live Predictions"):
        st.session_state.start_stream = False
        st.session_state.predictions.clear()
        st.info("Live prediction stopped. Click 'Start' to resume.")

# 🎥 Live prediction loop
if st.session_state.start_stream:
    frame = camera_input_live()
    if frame is not None:
        image_np = np.array(Image.fromarray(frame))
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                predicted_class = predict_letter(landmarks)
                st.session_state.predictions.append(predicted_class)

            stable_letter = get_stable_prediction(st.session_state.predictions)
            if stable_letter and stable_letter != 'fallback':
                if stable_letter == 'blank':
                    st.session_state.sentence += ' '
                else:
                    st.session_state.sentence += stable_letter
                st.session_state.predictions.clear()

# ✅ Mode-switch buttons (moved below live feed)
st.markdown("---")
st.markdown("### 🧭 Switch Mode:")
col3, col4 = st.columns(2)
with col3:
    if st.button("📸 Snapshot Mode"):
        st.switch_page("pages/app_snapshot.py")
with col4:
    if st.button("🖼️ Upload Mode"):
        st.switch_page("pages/app_upload.py")

# ✍️ Display sentence
st.markdown("### ✍️ Detected Sentence")
st.markdown(f"**{st.session_state.sentence}**")

# ✅ Word validation
words_in_sentence = st.session_state.sentence.strip().split()
valid_words = [word for word in words_in_sentence if word.lower() in nltk_words]
invalid_words = [word for word in words_in_sentence if word.lower() not in nltk_words]

st.markdown("### ✅ Valid Words")
st.write(valid_words if valid_words else "None")

st.markdown("### ❌ Invalid Words")
st.write(invalid_words if invalid_words else "None")

# 🔊 Speak sentence
if st.button("🔊 Speak Sentence"):
    text_to_speech(st.session_state.sentence)

# 🧹 Clear sentence
if st.button("🧹 Clear"):
    st.session_state.sentence = ""
    st.session_state.predictions.clear()
    gc.collect()

# 🪛 Optional debug panels
if st.checkbox("🪛 Show Raw Predictions (Debug)", value=False):
    st.write(st.session_state.predictions)

if st.checkbox("📜 Show Sentence History", value=False):
    st.write(st.session_state.sentence)

# 🏁 Entry point
if __name__ == '__main__':
    main()