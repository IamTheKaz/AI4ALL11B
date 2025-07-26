import os
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_camera_input_live import camera_input_live
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from gtts import gTTS
import nltk
from nltk.corpus import words
from io import BytesIO
import base64

# Hide sidebar and set page config
st.set_page_config(page_title="ASL Letter Predictor - Live", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

# Setup
nltk.download('words', quiet=True)
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
    try:
        model.load_weights(MODEL_PATH)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        st.error(f"Failed to load model weights: {e}. Ensure 'best_asl_model.h5' is in the root directory.")
        st.stop()
    return model

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(img_array, model):
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    letter = CLASS_NAMES[class_idx]
    confidence = np.max(predictions[0])
    top_3 = [(CLASS_NAMES[i], predictions[0][i]) for i in np.argsort(predictions[0])[-3:][::-1]]
    return letter, confidence, top_3

def main():
    st.title("🤟 ASL Letter Predictor - Live Webcam")
    st.write("Stream live webcam feed to predict ASL letters and form the phrase 'HELLO WORLD'. If the feed doesn't start, check webcam permissions or try Chrome/Firefox.")

    # Initialize session state
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
    if 'last_letter' not in st.session_state:
        st.session_state.last_letter = None
    if 'last_confidence' not in st.session_state:
        st.session_state.last_confidence = 0.0

    model = load_model()

    # Webcam input
    frame = camera_input_live()
    if frame is not None:
        try:
            img = np.frombuffer(frame.getvalue(), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                st.warning("Invalid frame received from webcam.")
                return
            img_array = preprocess_frame(img)
            letter, confidence, top_3 = predict_image(img_array, model)

            # Display frame with prediction
            cv2.putText(img, f"{letter.upper()} ({confidence:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            st.image(img, channels="BGR", caption=f"Predicted: {letter.upper()} ({confidence:.2f})")

            # Update sequence if confidence is high and letter changes
            if confidence > 0.7 and letter != st.session_state.last_letter:
                st.session_state.sequence.append(letter)
                st.session_state.last_letter = letter
                st.session_state.last_confidence = confidence

            # Display predictions
            if st.session_state.sequence:
                st.markdown(f"### Current Sequence: {', '.join(st.session_state.sequence[-10:])}")
                current = ''.join([l.upper() if l != 'space' else '' for l in st.session_state.sequence])
                
                # Check for NLTK words
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

                # Check for HELLO WORLD sequence
                target_sequence = ['H', 'E', 'L', 'L', 'O', 'space', 'W', 'O', 'R', 'L', 'D']
                if len(st.session_state.sequence) >= len(target_sequence):
                    recent = st.session_state.sequence[-len(target_sequence):]
                    if all(r == t for r, t in zip(recent, target_sequence)):
                        st.success("🎉 Phrase Detected: HELLO WORLD")

    # Buttons to switch to modes
    st.markdown("---")
    if st.button("Try the snapshot version"):
        st.switch_page("pages/app_snapshot.py")
    if st.button("Try the image upload version"):
        st.switch_page("pages/app_upload.py")

if __name__ == '__main__':
    main()