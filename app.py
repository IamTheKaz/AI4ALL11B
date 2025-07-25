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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time
from collections import deque

# Hide sidebar and set page config
st.set_page_config(page_title="ASL Letter Predictor", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True
)

# Setup
nltk.download('words')
nltk_words = set(w.upper() for w in words.words())
nltk_words.update(['HELLO', 'WORLD'])  # Ensure HELLO and WORLD are recognized

IMG_HEIGHT, IMG_WIDTH = 32, 32
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['del', 'nothing']  # Removed 'space'
MODEL_PATH = 'best_asl_model.h5'
CONFIDENCE_THRESHOLD = 0.7
STABILITY_FRAMES = 5  # ~1 second at 2-5 FPS
TARGET_SEQUENCE = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']

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

def predict_image(image, model):
    if isinstance(image, av.video.frame.VideoFrame):
        img = image.to_ndarray(format="rgb24")
        img = load_img(BytesIO(cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[1].tobytes()), 
                       target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    else:
        img = load_img(image, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)
    class_idx = np.argmax(predictions[0])
    letter = CLASS_NAMES[class_idx]
    confidence = np.max(predictions[0])
    top_3 = [(CLASS_NAMES[i], predictions[0][i]) for i in np.argsort(predictions[0])[-3:][::-1]]
    return letter, confidence, top_3

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
        self.prediction_buffer = deque(maxlen=STABILITY_FRAMES)
        self.last_audio_time = 0
        self.audio_cooldown = 1.5  # seconds between audio plays

    def recv(self, frame):
        letter, confidence, top_3 = predict_image(frame, self.model)
        if confidence > CONFIDENCE_THRESHOLD:
            self.prediction_buffer.append((letter, confidence, top_3))
        
        # Check for stable prediction
        if len(self.prediction_buffer) == STABILITY_FRAMES:
            letters = [p[0] for p in self.prediction_buffer]
            if all(l == letters[0] for l in letters):  # All same letter
                return frame, letters[0], confidence, top_3
        return frame, None, None, None

def update_sequence(letter, placeholder_letter, placeholder_confidence, placeholder_top3, placeholder_sequence):
    if letter not in ['del', 'nothing']:
        st.session_state.sequence.append(letter)
        
        # Update UI
        placeholder_letter.markdown(f"### ✅ Letter: `{letter.upper()}` — Confidence: `{confidence:.2f}`")
        placeholder_top3.write("🔝 Top 3 Predictions:")
        for i, (char, conf) in enumerate(top_3, 1):
            placeholder_top3.write(f"{i}. {char} — {conf:.2f}")
        placeholder_sequence.markdown(f"📜 Sequence: `{' '.join(st.session_state.sequence)}`")

        # Speak letter
        audio_buffer = speak_text(letter)
        st.markdown(
            f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer.read()).decode()}"></audio>',
            unsafe_allow_html=True
        )

        # Check for words
        current = ''.join(st.session_state.sequence)
        longest_word = ''
        for j in range(len(current), 1, -1):
            word = current[-j:]
            if word in nltk_words and len(word) > len(longest_word):
                longest_word = word

        if longest_word:
            placeholder_sequence.markdown(f"🗣 Detected word: **{longest_word}**")
            audio_buffer = speak_text(longest_word)
            st.markdown(
                f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer.read()).decode()}"></audio>',
                unsafe_allow_html=True
            )

        # Check for HELLO WORLD
        if len(st.session_state.sequence) >= len(TARGET_SEQUENCE):
            recent = st.session_state.sequence[-len(TARGET_SEQUENCE):]
            if all(r == t for r, t in zip(recent, TARGET_SEQUENCE)):
                st.success("🎉 Phrase Detected: HELLO WORLD")
                audio_buffer = speak_text("Hello World")
                st.markdown(
                    f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer.read()).decode()}"></audio>',
                    unsafe_allow_html=True
                )
                st.session_state.sequence = []  # Reset sequence

def main():
    st.title("🤟 ASL Letter Predictor")
    st.write("Use the webcam to capture ASL letters and form the phrase 'HELLO WORLD'. Switch to live feed for real-time predictions or upload images via the button below.")

    # Initialize session state
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
    if 'camera_mode' not in st.session_state:
        st.session_state.camera_mode = "snapshot"

    model = load_model()

    # UI placeholders
    placeholder_letter = st.empty()
    placeholder_top3 = st.empty()
    placeholder_sequence = st.empty()

    # Handle camera modes
    if st.session_state.camera_mode == "snapshot":
        st.subheader("Snapshot Mode")
        webcam_image = st.camera_input("Capture an ASL letter")
        if webcam_image:
            letter, confidence, top_3 = predict_image(BytesIO(webcam_image.getvalue()), model)
            update_sequence(letter, placeholder_letter, placeholder_confidence, placeholder_top3, placeholder_sequence)

        if st.button("Switch to Live Feed"):
            st.session_state.camera_mode = "live"
            st.rerun()

        if st.button("Close Camera"):
            st.session_state.camera_mode = "paused"
            st.rerun()

    elif st.session_state.camera_mode == "live":
        st.subheader("Live Feed Mode")
        ctx = webrtc_streamer(
            key="live-feed",
            video_processor_factory=VideoProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            media_stream_constraints={"video": True, "audio": False}
        )

        if ctx.video_processor:
            frame, letter, confidence, top_3 = ctx.video_processor.recv()
            if letter and time.time() - ctx.video_processor.last_audio_time > ctx.video_processor.audio_cooldown:
                update_sequence(letter, placeholder_letter, placeholder_confidence, placeholder_top3, placeholder_sequence)
                ctx.video_processor.last_audio_time = time.time()

        if st.button("Stop Live Feed"):
            st.session_state.camera_mode = "paused"
            st.rerun()

    elif st.session_state.camera_mode == "paused":
        st.subheader("Camera Paused")
        st.write("Click below to reopen the camera in snapshot mode.")
        if st.button("Open Camera"):
            st.session_state.camera_mode = "snapshot"
            st.rerun()

    # Navigation to upload app
    st.markdown("---")
    if st.button("Try the image upload version"):
        st.switch_page("pages/app_upload.py")

if __name__ == '__main__':
    main()