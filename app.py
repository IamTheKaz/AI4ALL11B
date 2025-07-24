import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
from gtts import gTTS
from PIL import Image
import nltk
from nltk.corpus import words

# Download NLTK corpus (on first run only)
nltk.download('words')
NLTK_WORDS = set(w.upper() for w in words.words())

# Configuration
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['space', 'del', 'nothing']
MODEL_PATH = 'models/best_asl_model.h5'

# Load model with matching architecture
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

# Prediction function
def predict_image(image, model):
    img = Image.open(image).convert('L').resize((32, 32))
    array = np.array(img).reshape(1, 32, 32, 1) / 255.0
    preds = model.predict(array)
    idx = np.argmax(preds[0])
    return CLASS_NAMES[idx], np.max(preds[0])

# Audio function
def speak_text(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts.save(fp.name)
        st.audio(fp.name, format='audio/mp3', start_time=0)

# App starts here
model = load_model()
st.title("🤟 ASL Letter Predictor")
st.markdown("Upload an ASL hand sign image and hear it spoken. Try spelling **HELLO WORLD**!")

uploaded_file = st.file_uploader("Upload one ASL image", type=["jpg", "jpeg", "png"])

# Session setup
if 'sequence' not in st.session_state:
    st.session_state.sequence = []
if 'phrase_detected' not in st.session_state:
    st.session_state.phrase_detected = False

if uploaded_file:
    st.image(uploaded_file, width=200)
    letter, confidence = predict_image(uploaded_file, model)
    st.markdown(f"### 🧠 Prediction: **{letter.upper()}** ({confidence:.2f})")
    st.session_state.sequence.append(letter)

    # Speak the letter
    speak_text({
        'space': 'space',
        'del': 'delete',
        'nothing': 'no letter detected'
    }.get(letter, letter))

    # Word detection
    current = ''.join([l.upper() if l != 'space' else '' for l in st.session_state.sequence])
    longest = ''
    for j in range(len(current), 1, -1):
        word = current[-j:]
        if word in NLTK_WORDS and len(word) > len(longest):
            longest = word
    if longest:
        st.markdown(f"🗣️ Recognized word: **{longest}**")
        speak_text(longest)

    # HELLO WORLD detection
    phrase = ''.join([' ' if l == 'space' else l.upper() for l in st.session_state.sequence])
    if 'HELLO WORLD' in phrase and not st.session_state.phrase_detected:
        st.success("🎉 Phrase Detected: HELLO WORLD")
        speak_text("Hello World")
        st.session_state.phrase_detected = True
