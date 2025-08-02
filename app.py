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

# 📦 Ensure NLTK words are available
nltk.download('words')
nltk_words = set(words.words())

# 🧠 Load model and class names
model = tf.keras.models.load_model("asl_model.h5")
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank']

# 🖐️ MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

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
    # Convert PIL image to NumPy array
    if isinstance(image, Image.Image):
        image_np = np.array(image.convert("RGB"))  # Ensure RGB mode

    elif isinstance(image, np.ndarray):
        image_np = image
        if image_np.shape[-1] == 4:
            image_np = image_np[:, :, :3]  # Drop alpha if present

    else:
        raise ValueError("Unsupported image format")

    # Ensure dtype is uint8
    image_np = image_np.astype(np.uint8)

    # Process with MediaPipe
    results = hands.process(image_np)

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)], None

    hand_landmarks = results.multi_hand_landmarks[0]

    # Optional: draw landmarks on a copy to avoid modifying original
    annotated_image = image_np.copy()
    mp_drawing.draw_landmarks(annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Prepare input for model
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

def main():
    # 🖼️ UI setup
    st.title("🖐️ Auto-Capture ASL Detector")
    st.markdown("Click below to start or stop live ASL detection from your webcam.")

    # 🧠 Session state initialization
    for key in ['prev_landmarks', 'sequence', 'last_prediction', 'start_stream']:
        if key not in st.session_state:
            st.session_state[key] = None if key == 'prev_landmarks' else []

    # 🎛️ Start/Stop buttons
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
            st.info("Live prediction stopped. Click 'Start' to resume.")

    # ⏱️ Refresh interval
    REFRESH_INTERVAL = 2

    # 🚀 Main live loop
    if st.session_state.get('start_stream', False):
        image = camera_input_live()
        if image:
            st.image(image, caption="Live Preview", channels="RGB")

            image_np = np.array(image)
            letter, confidence, top_3, current_landmarks = predict_image(image_np)

            if current_landmarks is not None and is_stable(current_landmarks, st.session_state.prev_landmarks):
                if letter != st.session_state.last_prediction:
                    st.session_state.last_prediction = letter
                    st.success(f"✋ Stable hand detected — predicted: `{letter}` ({confidence:.2f})")

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

            # ⏱️ Auto-refresh
            time.sleep(REFRESH_INTERVAL)
            st.experimental_rerun()
        else:
            st.warning("No image received. Is your webcam active?")

            # Buttons to switch to modes
            st.markdown("---")
            if st.button("Try the snapshot version"):
                st.switch_page("pages/app_snapshot.py")
            if st.button("Try the image upload version"):
                st.switch_page("pages/app_upload.py")

# 🏁 Entry point
if __name__ == '__main__':
    main()