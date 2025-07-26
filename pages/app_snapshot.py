import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gtts import gTTS
import nltk
from nltk.corpus import words
from io import BytesIO
import base64

# --- Page Configuration ---
st.set_page_config(page_title="ASL Letter Predictor", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

# --- Setup ---
nltk.download('words')
nltk_words = set(w.upper() for w in words.words())

IMG_HEIGHT, IMG_WIDTH = 32, 32
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']
MODEL_PATH = 'best_asl_model.h5'

def speak_text(text):
    tts = gTTS(text)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer

@st.cache_resource
def load_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
    ])
    model.load_weights(MODEL_PATH)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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

    model = load_model()

    st.subheader("Use Your Webcam")
    webcam_image = st.camera_input("Capture an ASL letter")
    if webcam_image:
        image_buffer = BytesIO(webcam_image.getvalue())
        letter, confidence, top_3 = predict_image(image_buffer, model)

        st.markdown(f"### ✅ Letter: `{letter.upper()}` — Confidence: `{confidence:.2f}`")
        st.write("🔝 Top 3 Predictions:")
        for i, (char, conf) in enumerate(top_3, 1):
            st.write(f"{i}. {char} — {conf:.2f}")

        speak_text_input = {'space': 'space', 'del': 'delete', 'nothing': 'no letter detected'}.get(letter, letter)
        audio_buffer = speak_text(speak_text_input)
        st.markdown(
            f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer.read()).decode()}"></audio>',
            unsafe_allow_html=True
        )

        st.session_state.sequence.append(letter)
        current = ''.join([l.upper() if l != 'space' else '' for l in st.session_state.sequence])

        longest_word = ''
        for j in range(len(current), 1, -1):
            word = current[-j:]
            if word in nltk_words and len(word) > len(longest_word):
                longest_word = word

        if longest_word:
            st.markdown(f"🗣 Detected word: **{longest_word}**")
            audio_buffer = speak_text(longest_word)
            st.markdown(
                f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer.read()).decode()}"></audio>',
                unsafe_allow_html=True
            )

        target_sequence = ['H', 'E', 'L', 'L', 'O', 'space', 'W', 'O', 'R', 'L', 'D']
        if len(st.session_state.sequence) >= len(target_sequence):
            recent = st.session_state.sequence[-len(target_sequence):]
            if all(r == t for r, t in zip(recent, target_sequence)):
                st.success("🎉 Phrase Detected: HELLO WORLD")
                audio_buffer = speak_text("Hello World")
                st.markdown(
                    f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer.read()).decode()}"></audio>',
                    unsafe_allow_html=True
                )
                st.session_state.sequence = []

    # --- Mode Switching Section ---
    st.markdown("---")
    st.markdown("#### 🧭 Try Alternate Input Modes:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🖼 Image Upload Version"):
            st.switch_page("pages/app_upload.py")
    with col2:
        if st.button("🤳 Live Webcam Version"):
            st.switch_page("app.py")

if __name__ == '__main__':
    main()