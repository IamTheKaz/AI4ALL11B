import os
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from gtts import gTTS
import nltk
from nltk.corpus import words
import requests
from io import BytesIO
import base64

# Hide sidebar and set page config
st.set_page_config(page_title="ASL Letter Predictor (Image Upload)", initial_sidebar_state="collapsed")
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

IMG_HEIGHT, IMG_WIDTH = 32, 32
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']
MODEL_PATH = 'best_asl_model.h5'

def speak_text(text):
    # Generate audio in memory
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

def get_image_download_link(img_url, filename):
    response = requests.get(img_url)
    img_data = response.content
    b64_string = base64.b64encode(img_data).decode()
    href = f'data:image/jpeg;base64,{b64_string}'
    return href

def main():
    st.title("🤟 ASL Letter Predictor (Image Upload Version)")
    st.write("Click each sample image below to download it, then use the file uploader to predict the letter and form the phrase 'HELLO WORLD'.")

    # Initialize session state
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []

    model = load_model()

    # GitHub image URLs for HELLO WORLD sequence
    github_images = {
        'H': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/H_test.jpg',
        'E': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/E_test.jpg',
        'L': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/L_test.jpg',
        'O': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/O_test.jpg',
        'space': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/space_test.jpg',
        'W': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/W_test.jpg',
        'R': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/R_test.jpg',
        'D': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/D_test.jpg'
    }

    # Display sample images for HELLO WORLD
    st.subheader("Sample Images for 'HELLO WORLD'")
    st.write("Click each image to download it, then upload it below to predict the letter. Follow the sequence to build 'HELLO WORLD'.")

    # First row: HELLO space
    cols1 = st.columns(6)
    hello_space_keys = ['H', 'E', 'L', 'L', 'O', 'space']
    for idx, key in enumerate(hello_space_keys):
        with cols1[idx]:
            display_key = 'L' if key == 'L' else key
            st.markdown(
                f'<a href="{get_image_download_link(github_images[display_key], f"{display_key}_test.jpg")}" download="{display_key}_test.jpg">'
                f'<img src="{github_images[display_key]}" alt="{display_key}" style="cursor:pointer;"></a>',
                unsafe_allow_html=True
            )
            st.caption(display_key)

    # Second row: WORLD, centered
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    cols2 = st.columns([1, 5, 1])  # Middle column wider to center WORLD
    world_keys = ['W', 'O', 'R', 'L', 'D']
    with cols2[1]:  # Use middle column for centering
        world_cols = st.columns(5)
        for idx, key in enumerate(world_keys):
            with world_cols[idx]:
                display_key = 'L' if key == 'L' else key
                st.markdown(
                    f'<a href="{get_image_download_link(github_images[display_key], f"{display_key}_test.jpg")}" download="{display_key}_test.jpg">'
                    f'<img src="{github_images[display_key]}" alt="{display_key}" style="cursor:pointer;"></a>',
                    unsafe_allow_html=True
                )
                st.caption(display_key)
    st.markdown("</div>", unsafe_allow_html=True)

    # File uploader
    st.subheader("Upload Your Image")
    uploaded_file = st.file_uploader("Upload a single ASL image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        letter, confidence, top_3 = predict_image(uploaded_file, model)

        st.markdown(f"### ✅ Letter: `{letter.upper()}` — Confidence: `{confidence:.2f}`")
        st.write("🔝 Top 3 Predictions:")
        for i, (char, conf) in enumerate(top_3, 1):
            st.write(f"{i}. {char} — {conf:.2f}")

        # Speak letter
        speak_text_input = {'space': 'space', 'del': 'delete', 'nothing': 'no letter detected'}.get(letter, letter)
        audio_buffer = speak_text(speak_text_input)
        st.markdown(
            f'<audio autoplay="true" src="data:audio/mp3;base64,{base64.b64encode(audio_buffer.read()).decode()}"></audio>',
            unsafe_allow_html=True
        )

        # Update sequence and check for words
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

        # Check for HELLO WORLD sequence
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
                st.session_state.sequence = []  # reset so it can re-detect

   # Buttons to switch to modes

    st.markdown("---")
    #if st.button("Try the live webcam version", key="webcam_button"):
      #  st.switch_page("app.py")
    st.markdown("---")
    if st.button("Try the snapshot version"):
        st.switch_page("app.py")
    


if __name__ == '__main__':
    main()