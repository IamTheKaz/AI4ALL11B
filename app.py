import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gtts import gTTS
import streamlit as st
import tempfile
import nltk
nltk.download('words')
from nltk.corpus import words

# Paths
MODEL_DIR = '.'  # Root directory
MODEL_PATH = 'best_asl_model.h5'  # Directly in root
IMG_HEIGHT, IMG_WIDTH = 32, 32
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['space', 'del']

os.makedirs(MODEL_DIR, exist_ok=True)

# TTS function for Streamlit
def speak_text(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts = gTTS(text)
        tts.save(fp.name)
        return fp.name

# Load model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])
        model.load_weights(MODEL_PATH)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    else:
        st.error("No trained model found at {}. Please upload best_asl_model.h5 to models/.".format(MODEL_PATH))
        return None

# Predict image
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

# Streamlit app
def main():
    st.title("ASL Letter Recognition")
    st.write("Upload ASL images to predict letters, words, and 'HELLO WORLD'!")
    
    # Initialize session state
    if 'predictions_sequence' not in st.session_state:
        st.session_state.predictions_sequence = []
        st.session_state.phrase = []
        st.session_state.hello_world_idx = 0
        st.session_state.phrase_detected = False
    
    # Load NLTK words
    nltk_words = set(word.upper() for word in words.words())

    model = load_model()
    if model is None:
        return

    # File uploader
    uploaded_files = st.file_uploader("Choose ASL images", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Predict letter
            letter, confidence, top_3 = predict_image(uploaded_file, model)
            st.write(f"Image: {uploaded_file.name} -> Predicted: {letter}, Confidence: {confidence:.4f}")
            st.session_state.predictions_sequence.append(letter)

            # Speak letter
            if letter == 'None':
                audio_file = speak_text('space')
            elif letter == 'del':
                audio_file = speak_text('delete')
            else:
                audio_file = speak_text(letter)
            st.audio(audio_file, format='audio/mp3')
            os.remove(audio_file)

            # Accumulate phrase for HELLO WORLD
            hello_world = list('HELLO WORLD')
            if st.session_state.hello_world_idx < len(hello_world):
                expected = 'space' if hello_world[st.session_state.hello_world_idx] == ' ' else hello_world[st.session_state.hello_world_idx]
                if letter == expected:
                    if letter == 'space':
                        st.session_state.phrase.append(' ')
                    elif letter not in ['del', 'nothing']:
                        st.session_state.phrase.append(letter)
                    st.session_state.hello_world_idx += 1
                else:
                    st.session_state.phrase = []
                    st.session_state.hello_world_idx = 0
                    if letter == hello_world[0]:
                        st.session_state.phrase.append(letter)
                        st.session_state.hello_world_idx = 1

            # Check for HELLO WORLD
            phrase_str = ''.join(st.session_state.phrase)
            if not st.session_state.phrase_detected and phrase_str.strip().upper() == 'HELLO WORLD':
                st.write("Demo Phrase detected: Hello World")
                audio_file = speak_text('Hello World')
                st.audio(audio_file, format='audio/mp3')
                os.remove(audio_file)
                st.session_state.phrase_detected = True
                st.session_state.phrase = []
                st.session_state.hello_world_idx = 0

            # Check for NLTK words
            current_sequence = ''.join([' ' if p == 'space' else p.upper() for p in st.session_state.predictions_sequence]).replace(' ', '')
            longest_word = ''
            for j in range(len(current_sequence), 0, -1):
                word = current_sequence[-j:]
                if word in nltk_words and len(word) > 1 and len(word) > len(longest_word):
                    longest_word = word
            if longest_word:
                st.write(f"Recognized word: {longest_word}")
                audio_file = speak_text(longest_word)
                st.audio(audio_file, format='audio/mp3')
                os.remove(audio_file)

if __name__ == '__main__':
    main()