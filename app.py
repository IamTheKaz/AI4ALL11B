import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gtts import gTTS
import tempfile
import nltk
from nltk.corpus import words

# Setup
nltk.download('words')
nltk_words = set(word.upper() for word in words.words())
IMG_HEIGHT, IMG_WIDTH = 32, 32
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['space', 'del', 'nothing']
MODEL_PATH = 'best_asl_model.h5'

# Speak text and return audio file path
def speak_text(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        gTTS(text).save(fp.name)
        return fp.name

# Predict a single image
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

# Load model with defined architecture
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

# Main Streamlit app
def main():
    st.title("ASL Letter Recognition & Spoken Feedback")
    st.markdown("Upload ASL hand sign images to predict and speak each letter. Detect complete words and phrases like **HELLO WORLD**.")

    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
        st.session_state.phrase = []
        st.session_state.hello_idx = 0
        st.session_state.phrase_detected = False

    model = load_model()

    uploaded_files = st.file_uploader("Upload ASL image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        hello_target = list('HELLO WORLD')

        for image_file in uploaded_files:
            letter, confidence, top_3 = predict_image(image_file, model)

            
            st.write(f"Letter: `{letter}` | Confidence: {confidence:.2f}")
            st.write("🔝 Top 3 predictions:")
            for i, (char, conf) in enumerate(top_3, 1):
                st.write(f"{i}. {char}: {conf:.2f}")

            st.session_state.sequence.append(letter)

            # Speak letter
            spoken = {'space': 'space', 'del': 'delete', 'nothing': 'no letter detected'}.get(letter, letter)
            audio_path = speak_text(spoken)
            st.audio(audio_path, format='audio/mp3')
            os.remove(audio_path)

            # Phrase detection
            expected = hello_target[st.session_state.hello_idx]
            match = (expected == ' ' and letter == 'space') or (letter == expected)

            if match:
                st.session_state.phrase.append(letter)
                st.session_state.hello_idx += 1
            else:
                st.session_state.phrase = []
                st.session_state.hello_idx = 0
                if letter == hello_target[0]:
                    st.session_state.phrase.append(letter)
                    st.session_state.hello_idx = 1

            phrase_str = ''.join([' ' if l == 'space' else l.upper() for l in st.session_state.phrase])

            if not st.session_state.phrase_detected and phrase_str.strip() == 'HELLO WORLD':
                st.success("🎉 Phrase Detected: HELLO WORLD")
                audio_path = speak_text("Hello World")
                st.audio(audio_path, format='audio/mp3')
                os.remove(audio_path)
                st.session_state.phrase_detected = True
                st.session_state.phrase = []
                st.session_state.hello_idx = 0

            # Word detection
            joined = ''.join([l.upper() if l != 'space' else '' for l in st.session_state.sequence])
            longest_word = ''
            for j in range(len(joined), 1, -1):
                word = joined[-j:]
                if word in nltk_words and len(word) > len(longest_word):
                    longest_word = word
            if longest_word:
                st.markdown(f"🧠 Recognized word: **{longest_word}**")
                audio_path = speak_text(longest_word)
                st.audio(audio_path, format='audio/mp3')
                os.remove(audio_path)

if __name__ == '__main__':
    main()
