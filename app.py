import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gtts import gTTS
import tempfile
from PIL import Image
import nltk
from nltk.corpus import words

# Setup
nltk.download('words')
nltk_words = set(w.upper() for w in words.words())

IMG_HEIGHT, IMG_WIDTH = 32, 32
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['space', 'del', 'nothing']
MODEL_PATH = 'best_asl_model.h5'

# Audio generator
def speak_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        return fp.name

# Load model with architecture
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

# Predict
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

# App
def main():
    st.title("🔤 ASL Letter Predictor with Word Detection")
    st.write("Upload an ASL image to see and hear the predicted letter, and listen to any full word that forms from the sequence.")

    if 'sequence' not in st.session_state:
        st.session_state.sequence = []

    model = load_model()
    uploaded_file = st.file_uploader("Upload one ASL image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        letter, confidence, top_3 = predict_image(uploaded_file, model)

        
        st.markdown(f"### ✅ Prediction: `{letter.upper()}` — Confidence: `{confidence:.2f}`")
        st.write("🔝 Top 3 Predictions:")
        for i, (char, conf) in enumerate(top_3, 1):
            st.write(f"{i}. {char} — {conf:.2f}")

        # Speak predicted letter
        spoken = {'space': 'space', 'del': 'delete', 'nothing': 'no letter detected'}.get(letter, letter)
        st.session_state.sequence.append(letter)

        audio_path = speak_text(spoken)
        st.audio(audio_path, format='audio/mp3')
        os.remove(audio_path)

        # Word detection on running sequence
        current = ''.join([l.upper() if l != 'space' else '' for l in st.session_state.sequence])
        longest_word = ''
        for j in range(len(current), 1, -1):
            word = current[-j:]
            if word in nltk_words and len(word) > len(longest_word):
                longest_word = word

        if longest_word:
            st.markdown(f"🧠 Recognized word: **{longest_word}**")
            word_audio = speak_text(longest_word)
            st.audio(word_audio, format='audio/mp3')
            os.remove(word_audio)

if __name__ == '__main__':
    main()


