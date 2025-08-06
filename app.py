import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
import joblib
from gtts import gTTS
import base64
import io
from PIL import Image
from camera_input_live import camera_input_live

# 🧼 UI setup
st.set_page_config(page_title="ASL Live Detector", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    </style>
""", unsafe_allow_html=True)

# 📦 Load words corpus once
if 'nltk_words' not in st.session_state:
    import nltk
    from nltk.corpus import words
    nltk.download('words')
    st.session_state.nltk_words = set(words.words())

# 🖐️ MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 🧠 Load model and scaler
model = tf.keras.models.load_model("asl_model.h5")
scaler = joblib.load("scaler.pkl")

CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank']  # A–Z + blank

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
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))
    else:
        image_np = np.array(image)
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]

    image_np = image_np.astype(np.uint8)
    results = hands.process(image_np)

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)], None

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(image_np, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    raw_landmarks = extract_landmark_array(hand_landmarks)

    if raw_landmarks.shape[0] != scaler.mean_.shape[0]:
        return "blank", 0.0, [("blank", 1.0)], raw_landmarks

    normalized = scaler.transform(raw_landmarks.reshape(1, -1))
    prediction_probs = model.predict(normalized)[0]
    pred_index = np.argmax(prediction_probs)
    prediction = CLASS_NAMES[pred_index]
    confidence = prediction_probs[pred_index]
    top_3 = [(CLASS_NAMES[i], prediction_probs[i]) for i in np.argsort(prediction_probs)[-3:][::-1]]

    return prediction, confidence, top_3, raw_landmarks

# Stability check
def is_stable(current, previous, threshold=0.01):
    return previous is not None and np.linalg.norm(current - previous) < threshold

# 🚀 Main app
def main():
    st.title("🤟 ASL Live Detector (Streamlined)")
    st.markdown("Detect ASL signs using live hand landmarks and consistent scaling.")

    # Session state setup
    for key in ['prev_landmarks', 'sequence', 'last_prediction', 'start_stream']:
        if key not in st.session_state:
            st.session_state[key] = None if key == 'prev_landmarks' else []

    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Live Predictions"):
            st.session_state.start_stream = True
    with col2:
        if st.button("⏹️ Stop Live Predictions"):
            st.session_state.start_stream = False
            st.session_state.prev_landmarks = None
            st.session_state.sequence = []
            st.session_state.last_prediction = None
            st.info("Live prediction stopped.")

    image_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.session_state.get('start_stream', False):
        image = camera_input_live()
        if image:
            image_placeholder.image(image, caption="Live Preview", channels="RGB")
            image_np = np.array(image)
            letter, confidence, top_3, current_landmarks = predict_image(image_np)

            if current_landmarks is not None and is_stable(current_landmarks, st.session_state.prev_landmarks):
                if letter != st.session_state.last_prediction:
                    st.session_state.last_prediction = letter
                    status_placeholder.success(f"✋ Predicted: `{letter}` ({confidence:.2f})")

                    st.markdown("#### 🔝 Top 3 Predictions:")
                    for i, (char, conf) in enumerate(top_3, 1):
                        st.write(f"{i}. `{char}` — `{conf:.2f}`")

                    st.markdown(get_audio_download_link(speak_text(letter if letter != "blank" else "No hand sign")))

                    if letter != "blank":
                        st.session_state.sequence.append(letter)

                    # 🔤 Word detection
                    current = ''.join(st.session_state.sequence).upper()
                    longest_word = ''
                    for j in range(len(current), 1, -1):
                        word = current[-j:]
                        if word in st.session_state.nltk_words and len(word) > len(longest_word):
                            longest_word = word

                    if longest_word:
                        st.markdown(f"🗣 Word detected: **{longest_word}**")
                        st.markdown(get_audio_download_link(speak_text(longest_word)))

                    # 🎉 Phrase detection
                    target_sequence = ['H','E','L','L','O','W','O','R','L','D']
                    if st.session_state.sequence[-10:] == target_sequence:
                        st.success("🎉 Phrase Detected: HELLO WORLD")
                        st.markdown(get_audio_download_link(speak_text("Hello World")))
                        st.session_state.sequence = []

            st.session_state.prev_landmarks = current_landmarks

    # 🔀 Mode-switch section (moved to bottom)
    st.markdown("---")
    st.markdown("### 🧭 Switch Mode:")
    col3, col4 = st.columns(2)
    with col3:
        if st.button("📸 Snapshot Mode"):
            st.switch_page("pages/app_snapshot.py")
    with col4:
        if st.button("🖼️ Upload Mode"):
            st.switch_page("pages/app_upload.py")

if __name__ == "__main__":
    main()
