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
nltk.download('words')
nltk_words = set(words.words())

# Load your trained TensorFlow model
model = tf.keras.models.load_model("asl_model.h5")

# Define class names: A-Z + 'blank' + fallback
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank', 'fallback']

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
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)]

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Match training format: x + y + z
    x_vals = [lm.x for lm in hand_landmarks.landmark]
    y_vals = [lm.y for lm in hand_landmarks.landmark]
    z_vals = [lm.z for lm in hand_landmarks.landmark]
    input_array = np.array(x_vals + y_vals + z_vals).reshape(1, -1)

    prediction_probs = model.predict(input_array)[0]
    pred_index = np.argmax(prediction_probs)

    if pred_index >= len(CLASS_NAMES):
        return "fallback", 0.0, [("fallback", 1.0)]

    letter = CLASS_NAMES[pred_index]
    confidence = prediction_probs[pred_index]
    top_3 = [(CLASS_NAMES[i], prediction_probs[i]) for i in np.argsort(prediction_probs)[-3:][::-1]]

    return letter, confidence, top_3

# 🎯 Main App
def main():
    st.title("🤟 ASL Letter Predictor")
    st.write("Use the webcam to capture ASL letters and form the phrase 'HELLO WORLD'. Alternatively, use the button below to upload images.")

    if 'sequence' not in st.session_state:
        st.session_state.sequence = []

    st.subheader("Use Your Webcam")
    webcam_image = st.camera_input("Capture an ASL letter")

    if webcam_image:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(webcam_image.getvalue())
            tmp_file_path = tmp_file.name

        image = cv2.imread(tmp_file_path)
        letter, confidence, top_3 = predict_image(image)

        st.image(image, caption=f"🖼️ Prediction: `{letter.upper()}`", channels="BGR")
        st.markdown(f"### ✅ Letter: `{letter.upper()}` — Confidence: `{confidence:.2f}`")
        st.write("🔝 Top 3 Predictions:")
        for i, (char, conf) in enumerate(top_3, 1):
            st.write(f"{i}. {char} — {conf:.2f}")

        spoken_text = speak_text_input(letter)
        audio_buffer = speak_text(spoken_text)
        st.markdown(get_audio_download_link(audio_buffer), unsafe_allow_html=True)

        if letter != "blank" and letter != "fallback":
            st.session_state.sequence.append(letter)

        current = ''.join([l.upper() for l in st.session_state.sequence])
        longest_word = ''
        for j in range(len(current), 1, -1):
            word = current[-j:]
            if word in nltk_words and len(word) > len(longest_word):
                longest_word = word

        if longest_word:
            st.markdown(f"🗣 Detected word: **{longest_word}**")
            audio_buffer = speak_text(longest_word)
            st.markdown(get_audio_download_link(audio_buffer), unsafe_allow_html=True)

        target_sequence = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']
        if len(st.session_state.sequence) >= len(target_sequence):
            recent = st.session_state.sequence[-len(target_sequence):]
            if all(r == t for r, t in zip(recent, target_sequence)):
                st.success("🎉 Phrase Detected: HELLO WORLD")
                audio_buffer = speak_text("Hello World")
                st.markdown(get_audio_download_link(audio_buffer), unsafe_allow_html=True)
                st.session_state.sequence = []

if __name__ == "__main__":
    main()