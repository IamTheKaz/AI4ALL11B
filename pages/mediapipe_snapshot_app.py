import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from gtts import gTTS
import base64
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO
from nltk.corpus import words
import nltk

# Ensure NLTK words corpus is available
nltk.download('words')
nltk_words = set(words.words())

# Load your trained TensorFlow model
model = tf.keras.models.load_model("asl_model.h5")

# Define image dimensions expected by the model
IMG_HEIGHT, IMG_WIDTH = 32, 32

# Define class names: A-Z plus 'blank'
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

def predict_image(image_file, model):
    img = load_img(image_file, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    letter = CLASS_NAMES[class_idx]
    confidence = np.max(predictions[0])
    top_3 = [(CLASS_NAMES[i], predictions[0][i]) for i in np.argsort(predictions[0])[-3:][::-1]]
    return letter, confidence, top_3

def main():
    st.title("🤟 ASL Letter Predictor")
    st.write("Use the webcam to capture ASL letters and form the phrase 'HELLO WORLD'. Alternatively, use the button below to upload images.")

    if 'sequence' not in st.session_state:
        st.session_state.sequence = []

    st.subheader("Use Your Webcam")
    webcam_image = st.camera_input("Capture an ASL letter")
    if webcam_image:
        image_buffer = BytesIO(webcam_image.getvalue())
        letter, confidence, top_3 = predict_image(image_buffer, model)

        st.markdown(f"### ✅ Letter: `{letter.upper()}` — Confidence: `{confidence:.2f}`")
        st.write("🔝 Top 3 Predictions:")
        for i, (char, conf) in enumerate(top_3, 1):
            st.write(f"{i}. {char} — {conf:.2f}")

        speak_text_input_value = speak_text_input(letter)
        audio_buffer = speak_text(speak_text_input_value)
        st.markdown(
            f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer).decode()}"></audio>',
            unsafe_allow_html=True
        )

        if letter != 'blank':
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
            st.markdown(
                f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer).decode()}"></audio>',
                unsafe_allow_html=True
            )

        target_sequence = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']
        if len(st.session_state.sequence) >= len(target_sequence):
            recent = st.session_state.sequence[-len(target_sequence):]
            if all(r == t for r, t in zip(recent, target_sequence)):
                st.success("🎉 Phrase Detected: HELLO WORLD")
                audio_buffer = speak_text("Hello World")
                st.markdown(
                    f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer).decode()}"></audio>',
                    unsafe_allow_html=True
                )
                st.session_state.sequence = []

if __name__ == "__main__":
    main()