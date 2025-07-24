import streamlit as st
import tensorflow as tf
import numpy as np
import os
import tempfile
from gtts import gTTS
from nltk.corpus import words
from PIL import Image
import nltk

# Ensure NLTK corpus is available
nltk.download('words')
NLTK_WORDS = set(word.upper() for word in words.words())

CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['space', 'del', 'nothing']
MODEL_PATH = 'models/best_asl_model.h5'

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

def speak_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        st.audio(fp.name, format='audio/mp3', start_time=0)

def predict_image(file, model):
    img = Image.open(file).convert('L').resize((32, 32))
    img_array = np.array(img).reshape(1, 32, 32, 1) / 255.0
    preds = model.predict(img_array)
    idx = np.argmax(preds[0])
    return CLASS_NAMES[idx], np.max(preds[0])

# Load model once
model = load_model()

st.title("🤟 ASL Letter Detection")
st.markdown("Upload a single image to see and hear the predicted letter. Try spelling `HELLO WORLD`!")

uploaded_file = st.file_uploader("Upload ASL image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=200)
    letter, confidence = predict_image(uploaded_file, model)
    st.markdown(f"### 🧠 Predicted: **{letter.upper()}** ({confidence:.2f} confidence)")

    speak_text({
        'space': 'space',
        'del': 'delete',
        'nothing': 'no letter detected'
    }.get(letter, letter))

    # Recognize longest NLTK word ending at current letter sequence
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
    st.session_state.sequence.append(letter)

    current = ''.join([l.upper() if l != 'space' else '' for l in st.session_state.sequence])
    longest = ''
    for j in range(len(current), 1, -1):
        word = current[-j:]
        if word in NLTK_WORDS and len(word) > len(longest):
            longest = word
    if longest:
        st.markdown(f"🗣️ Detected word: **{longest}**")
        speak_text(longest)

    # "HELLO WORLD" detection
    phrase = ''.join([' ' if l == 'space' else l.upper() for l in st.session_state.sequence])
    if 'HELLO WORLD' in phrase and not st.session_state.get('phrase_detected'):
        st.success("🎉 HELLO WORLD detected!")
        speak_text("Hello World")
        st.session_state.phrase_detected = True
