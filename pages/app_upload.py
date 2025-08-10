import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from gtts import gTTS
import base64
import io
import tensorflow as tf
from nltk.corpus import words
import nltk
import zipfile
import io
import requests
import pandas as pd

# 📦 Setup
nltk.download('words')
nltk_words = set(words.words())

# 🧼 Hide sidebar and set page config
st.set_page_config(page_title="ASL Upload Detector", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebarContent"] { display: none; }
    .css-1d391kg { display: none; }
    </style>
""", unsafe_allow_html=True)

# 🖐️ MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 🧠 Load model and class names
model = tf.keras.models.load_model("asl_model.h5")
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['blank', 'fallback']

# 📦 Load zipped dataset from GitHub
@st.cache_data
def load_dataset():
    url = "https://github.com/IamTheKaz/AI4ALL11B/raw/main/hand_landmarks.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open("hand_landmarks.csv") as f:
            return pd.read_csv(f)

df_landmarks = load_dataset()
# 🔊 Speech synthesis
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

def normalize_landmarks(landmarks):
    WRIST_IDX = 0
    MIDDLE_MCP_IDX = 9

    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    origin = points[WRIST_IDX]
    points -= origin

    ref_point = points[MIDDLE_MCP_IDX]
    angle = np.arctan2(ref_point[1], ref_point[0])
    rot_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle), 0],
        [np.sin(-angle),  np.cos(-angle), 0],
        [0,               0,              1]
    ])
    points = points @ rot_matrix.T
    return points.flatten().reshape(1, -1)

def get_finger_spread(landmarks):
    x_vals = [landmarks[8].x, landmarks[12].x, landmarks[16].x]
    return max(x_vals) - min(x_vals)

# 🧠 Prediction logic
def predict_image(image):
    image = cv2.resize(image, (224, 224))  # Match training resolution
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)], np.zeros((1, 64))

    score = results.multi_handedness[0].classification[0].score
    if score < 0.75:
        return "blank", 0.0, [("blank", 1.0)], np.zeros((1, 64))
    

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)], np.zeros((1, 64))

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    normalized = normalize_landmarks(hand_landmarks.landmark)  # shape (1, 63)
    spread = get_finger_spread(hand_landmarks.landmark)        # scalar float
    spread_array = np.array([[spread]])                        # shape (1, 1)
    input_array = np.hstack((normalized, spread_array))        # shape (1, 64)

    st.write(f"🧪 MediaPipe result: `{results.multi_hand_landmarks}`")
    st.write(f"✅ Final input shape: {input_array.shape}")

    if results.multi_hand_landmarks:
        annotated = image.copy()
        mp_drawing.draw_landmarks(annotated, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
        st.image(annotated, caption="🖐️ MediaPipe Landmarks", channels="BGR")

    prediction_probs = model.predict(input_array)[0]
    pred_index = np.argmax(prediction_probs)

    if pred_index >= len(CLASS_NAMES):
        return "fallback", 0.0, [("fallback", 1.0)]

    prediction = CLASS_NAMES[pred_index]
    confidence = round(prediction_probs[pred_index], 2)
    top_3 = [(CLASS_NAMES[i], round(prediction_probs[i], 2)) for i in np.argsort(prediction_probs)[-3:][::-1]]

    return prediction, confidence, top_3, input_array

# 🚀 Main app
st.title("🤟 Upload ASL Detector")
st.markdown("Upload an image of a hand sign to predict the letter. Try forming the phrase **HELLO WORLD**!")

# 🧪 Sample Images
st.markdown("### 🧪 Sample Images for 'HELLO WORLD'")
github_images = {
    'H': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/H_test.jpg',
    'E': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/E_test.jpg',
    'L': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/L_test.jpg',
    'O': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/O_test.jpg',
    'W': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/W_test.jpg',
    'R': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/R_test.jpg',
    'D': 'https://raw.githubusercontent.com/IamTheKaz/AI4ALL11B/main/D_test.jpg'
}

# First row: HELLO
cols1 = st.columns(5)
hello_keys = ['H', 'E', 'L', 'L', 'O']
for idx, key in enumerate(hello_keys):
    with cols1[idx]:
        st.markdown(
            f'<a href="{github_images[key]}" download="{key}_test.jpg">'
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
                f'<a href="{github_images[key]}" download="{key}_test.jpg">'
                f'<img src="{github_images[key]}" alt="{key}" style="cursor:pointer; width:100%;"></a>',
                unsafe_allow_html=True
            )
            st.caption(key)
st.markdown("</div>", unsafe_allow_html=True)

# 📤 Upload section
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if 'sequence' not in st.session_state:
    st.session_state.sequence = []

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    letter, confidence, top_3, input_array = predict_image(image)

    st.image(image, caption=f"🖼️ Prediction: `{letter.upper()}`", channels="BGR")
    st.markdown(f"### ✅ Detected Letter: `{letter.upper()}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}`")

    st.markdown("#### 🔝 Top 3 Predictions:")
    for i, (char, conf) in enumerate(top_3, 1):
        st.write(f"{i}. `{char}` — `{conf:.2f}`")

    spoken_text = speak_text_input(letter)
    st.markdown(get_audio_download_link(speak_text(spoken_text)), unsafe_allow_html=True)

    if letter not in ["blank", "fallback"]:
        st.session_state.sequence.append(letter)

    current = ''.join(st.session_state.sequence).upper()
    longest_word = ''
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

    # Filter training samples for label 'L'
    l_samples = df_landmarks[df_landmarks["label"] == "L"].drop(columns=["label", "source"])
    l_vectors = l_samples.values

    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(input_array, l_vectors)

    # Show top match
    st.markdown(f"🔍 Cosine similarity to training 'L' samples: `{similarities.max():.4f}`")

    if letter != "blank":
        st.markdown("### 🧬 Input Vector (Normalized Landmarks + Finger Spread)")
        st.dataframe(pd.DataFrame(input_array, columns=[f"f{i}" for i in range(input_array.shape[1])]))
    else:
        st.caption("⚠️ No hand detected — input vector is zero-filled.")

# 🧭 Mode-switch buttons
st.markdown("---")
st.markdown("### 🧭 Try Alternate Input Modes:")
col1, col2 = st.columns(2)
with col1:
    if st.button("📷 Snapshot Mode"):
        st.switch_page("pages/app_snapshot.py")
with col2:
    if st.button("🎬 Live Mode"):
        st.switch_page("app.py")