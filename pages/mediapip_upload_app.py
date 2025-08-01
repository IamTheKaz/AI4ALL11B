import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from gtts import gTTS
import base64
import io
import tensorflow as tf

# Load your trained TensorFlow model
model = tf.keras.models.load_model("asl_model.h5")

# Define class names: A-Z and 'blank' only
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

# 🧠 Prediction function
def predict_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)]

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Extract and reshape landmarks
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    input_array = np.array(landmarks).reshape(1, -1)
    prediction_probs = model.predict(input_array)[0]
    pred_index = np.argmax(prediction_probs)
    prediction = CLASS_NAMES[pred_index]
    confidence = round(prediction_probs[pred_index], 2)

    top_3 = [(CLASS_NAMES[i], round(prediction_probs[i], 2)) for i in np.argsort(prediction_probs)[-3:][::-1]]
    return prediction, confidence, top_3

# 🎯 Streamlit UI
st.title("🤟 ASL Hand Sign Detection (Upload Version)")
st.write("Upload an image of a hand sign to predict the letter and build the phrase 'HELLO WORLD'.")

# 🧪 Sample Images for HELLO WORLD
st.subheader("🧪 Sample Images for 'HELLO WORLD'")
st.write("Click each image to download it, then upload it below to predict the letter.")

github_images = {
    'H': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/H_test.jpg',
    'E': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/E_test.jpg',
    'L': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/L_test.jpg',
    'O': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/O_test.jpg',
    'W': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/W_test.jpg',
    'R': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/R_test.jpg',
    'D': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/D_test.jpg'
}

def get_image_download_link(url, filename):
    return url  # Direct GitHub raw link

# First row: HELLO
cols1 = st.columns(5)
hello_keys = ['H', 'E', 'L', 'L', 'O']
for idx, key in enumerate(hello_keys):
    with cols1[idx]:
        st.markdown(
            f'<a href="{get_image_download_link(github_images[key], f"{key}_test.jpg")}" download="{key}_test.jpg">'
            f'<img src="{github_images[key]}" alt="{key}" style="cursor:pointer; width:100%;"></a>',
            unsafe_allow_html=True
        )
        st.caption(key)

# Second row: WORLD
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
cols2 = st.columns([1, 5, 1])
world_keys = ['W', 'O', 'R', 'L', 'D']
with cols2[1]:
    world_cols = st.columns(5)
    for idx, key in enumerate(world_keys):
        with world_cols[idx]:
            st.markdown(
                f'<a href="{get_image_download_link(github_images[key], f"{key}_test.jpg")}" download="{key}_test.jpg">'
                f'<img src="{github_images[key]}" alt="{key}" style="cursor:pointer; width:100%;"></a>',
                unsafe_allow_html=True
            )
            st.caption(key)
st.markdown("</div>", unsafe_allow_html=True)

# 🧠 Prediction Section
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if 'sequence' not in st.session_state:
    st.session_state.sequence = []

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    letter, confidence, top_3 = predict_image(image)

    st.image(image, caption=f"🖼️ Prediction: `{letter.upper()}`", channels="BGR")
    st.markdown(f"### ✅ Detected Letter: `{letter.upper()}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")

    st.markdown("#### 🔝 Top 3 Predictions:")
    for i, (char, conf) in enumerate(top_3, 1):
        st.write(f"{i}. `{char}` — `{conf:.2f}`")

    spoken_text = speak_text_input(letter)
    st.markdown(get_audio_download_link(speak_text(spoken_text)), unsafe_allow_html=True)

    if letter != "blank":
        st.session_state.sequence.append(letter)

    current = ''.join(st.session_state.sequence).upper()
    longest_word = ''
    from nltk.corpus import words
    import nltk
    nltk.download('words')
    nltk_words = set(words.words())

    for j in range(len(current), 1, -1):
        word = current[-j:]
        if word in nltk_words and len(word) > len(longest_word):
            longest_word = word

    if longest_word:
        st.markdown(f"🗣 Detected Word: **{longest_word}**")
        st.markdown(get_audio_download_link(speak_text(longest_word)), unsafe_allow_html=True)

    target_sequence = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']
    if len(st.session_state.sequence) >= len(target_sequence):
        recent = st.session_state.sequence[-len(target_sequence):]
        if all(r == t for r, t in zip(recent, target_sequence)):
            st.success("🎉 Phrase Detected: HELLO WORLD")
            st.markdown(get_audio_download_link(speak_text("Hello World")), unsafe_allow_html=True)
            st.session_state.sequence = []

# --- Mode Switching Section ---
st.markdown("---")
st.markdown("#### 🧭 Try Alternate Input Modes:")
col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Snapshot Version"):
        st.switch_page("pages/app_snapshot.py")
with col2:
    if st.button("🤳 Live Webcam Version"):
        st.switch_page("app.py")