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
from nltk.corpus import words
import nltk
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
nltk.download('words')
nltk_words = set(words.words())

# 🖐️ MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 🧠 Load model and class names
model = tf.keras.models.load_model("asl_model.h5")
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank']

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
    elif isinstance(image, np.ndarray):
        image_np = image
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]
    else:
        raise ValueError("Unsupported image format")

    image_np = image_np.astype(np.uint8)
    results = hands.process(image_np)

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)], None

    hand_landmarks = results.multi_hand_landmarks[0]
    annotated_image = image_np.copy()
    mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    landmark_array = extract_landmark_array(hand_landmarks).reshape(1, -1)
    prediction_probs = model.predict(landmark_array)[0]
    pred_index = np.argmax(prediction_probs)
    prediction = CLASS_NAMES[pred_index]
    confidence = prediction_probs[pred_index]
    top_3 = [(CLASS_NAMES[i], prediction_probs[i]) for i in np.argsort(prediction_probs)[-3:][::-1]]

    return prediction, confidence, top_3, extract_landmark_array(hand_landmarks)

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

    # 🧠 Session state setup
    for key in ['prev_landmarks', 'sequence', 'last_prediction', 'start_stream', 'last_frame_time']:
        if key not in st.session_state:
            st.session_state[key] = None if key == 'prev_landmarks' else [] if key == 'sequence' else False if key == 'start_stream' else 0.0

    # ▶️ Start/Stop buttons
    st.markdown("### 🎬 Live Detection Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start Live Predictions"):
            st.session_state.start_stream = True
            st.session_state.last_frame_time = time.time()
    with col2:
        if st.button("⏹️ Stop Live Predictions"):
            st.session_state.start_stream = False
            st.session_state.prev_landmarks = None
            st.session_state.sequence = []
            st.session_state.last_prediction = None
            st.info("Live prediction stopped. Click 'Start' to resume.")

    # 🎥 Live prediction loop
    image_placeholder = st.empty()
    status_placeholder = st.empty()

    if st.session_state.get('start_stream', False):
        now = time.time()
        if now - st.session_state.last_frame_time < 3:
            st.warning("📷 Warming up camera... Please hold still.")
            return

        frame = camera_input_live()
        if frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            image_placeholder.image(pil_image, caption="Live Preview", channels="RGB")

            # ⏱️ Throttle predictions to 1 per second
            if now - st.session_state.last_frame_time < 1.0:
                return
            st.session_state.last_frame_time = now

            letter, confidence, top_3, current_landmarks = predict_image(pil_image)

            print(f"[DEBUG] Letter: {letter}, Confidence: {confidence:.2f}")
            if current_landmarks is not None:
                delta = np.linalg.norm(current_landmarks - st.session_state.prev_landmarks) if st.session_state.prev_landmarks is not None else None
                print(f"[DEBUG] Stability delta: {delta:.4f}" if delta is not None else "[DEBUG] First frame")

            if current_landmarks is not None and is_stable(current_landmarks, st.session_state.prev_landmarks):
                if letter != st.session_state.last_prediction:
                    st.session_state.last_prediction = letter
                    status_placeholder.success(f"✋ Stable hand detected — predicted: `{letter}` ({confidence:.2f})")

                    st.markdown("#### 🔝 Top 3 Predictions:")
                    for i, (char, conf) in enumerate(top_3, 1):
                        st.write(f"{i}. `{char}` — `{conf:.2f}`")

                    spoken_text = letter if letter != "blank" else "No hand sign detected"
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

            st.session_state.prev_landmarks = current_landmarks

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

# 🏁 Entry point
if __name__ == '__main__':
    main()
