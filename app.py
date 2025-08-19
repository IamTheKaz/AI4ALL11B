import cv2
import numpy as np
import mediapipe as mp
import streamlit as st
from gtts import gTTS
import base64
import io
import tensorflow as tf
from PIL import Image
import tempfile
from nltk.corpus import words
import nltk
import pickle


# 🧼 Hide sidebar and set page config
st.set_page_config(page_title="ASL Snapshot Detector", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebarContent"] { display: none; }
    .css-1d391kg { display: none; }
    </style>
""", unsafe_allow_html=True)

# 📦 Setup
@st.cache_data
def load_nltk_words():
    try:
        return set(words.words())
    except LookupError:
        nltk.download('words')
        return set(words.words())

nltk_words = load_nltk_words()

# 🖐️ MediaPipe setup
try:
    mp_hands = mp.solutions.hands
    mp_hands_instance = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    HAND_CONNECTIONS = getattr(mp_hands, 'HAND_CONNECTIONS', None)
    if HAND_CONNECTIONS is None:
        st.warning("HAND_CONNECTIONS not detected. Landmark drawing will be disabled.")
except Exception as e:
    st.error(f"MediaPipe initialization failed: {e}")
    st.stop()

IMG_SIZE = 224

# 🧠 Load model and class names
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("asl_model.h5")
    except Exception as e:
        st.error(f"Failed to load model: {e}. Ensure 'asl_model.h5' is in the directory.")
        st.stop()

model = load_model()
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['nothing']

# 🔊 Speech synthesis
def speak_text(text):
    try:
        tts = gTTS(text=text, lang='en')
        with io.BytesIO() as f:
            tts.write_to_fp(f)
            f.seek(0)
            return f.read()
    except Exception as e:
        st.error(f"Speech synthesis failed: {e}")
        return b''

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
    return points  # Return 21x3 array

def get_finger_spread(landmarks):
    # Index tip (8), middle tip (12), ring tip (16)
    x_vals = [landmark.x for landmark in [landmarks[8], landmarks[12], landmarks[16]]]
    return max(x_vals) - min(x_vals)

def get_angle(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)  # Angle in radians

def get_finger_curvature(points, finger_joints):
    # finger_joints: list of indices [base, joint1, joint2, tip]
    straight_dist = np.linalg.norm(points[finger_joints[3]] - points[finger_joints[0]])
    path_length = sum(np.linalg.norm(points[finger_joints[i+1]] - points[finger_joints[i]]) for i in range(3))
    return straight_dist / path_length if path_length > 0 else 1.0

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# 🧠 Prediction logic with normalization
def predict_image(image):
    try:
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = mp_hands_instance.process(image_rgb)
        if not results.multi_hand_landmarks:
            st.warning("🚫 No hand landmarks detected.")
            return "Could not identify hand sign", 0.0, [("Could not identify hand sign", 1.0)]

        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(image, hand_landmarks, HAND_CONNECTIONS)

        landmarks_raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        if landmarks_raw.shape != (21, 3):
            st.warning(f"🚫 Unexpected landmark shape: {landmarks_raw.shape}")
            return "Could not identify hand sign", 0.0, [("Could not identify hand sign", 1.0)]

        # ✅ Feature engineering (updated for 71 features)
        normalized_array = normalize_landmarks(hand_landmarks.landmark)  # 21x3 array
        normalized_flat = normalized_array.flatten()  # 63 values
        spread = get_finger_spread(hand_landmarks.landmark)  # scalar

        # Compute angles using normalized vectors (tips: thumb=4, index=8, middle=12)
        v_thumb = normalized_array[4]
        v_index = normalized_array[8]
        v_middle = normalized_array[12]
        angle_thumb_index = get_angle(v_thumb, v_index)
        angle_index_middle = get_angle(v_index, v_middle)

        # Compute curvatures
        curv_thumb = get_finger_curvature(normalized_array, [1,2,3,4])
        curv_index = get_finger_curvature(normalized_array, [5,6,7,8])
        curv_middle = get_finger_curvature(normalized_array, [9,10,11,12])
        curv_ring = get_finger_curvature(normalized_array, [13,14,15,16])
        curv_pinky = get_finger_curvature(normalized_array, [17,18,19,20])
        curvatures = [curv_thumb, curv_index, curv_middle, curv_ring, curv_pinky]

        # Combine all features
        features = np.concatenate([normalized_flat, [spread, angle_thumb_index, angle_index_middle], curvatures])
        input_array = features.reshape(1, -1)  # shape (1, 71)

        # Scale the input features
        input_array_scaled = scaler.transform(input_array)

        if input_array.shape[1] != 71:
            st.warning(f"🚫 Input shape mismatch: expected 71, got {input_array.shape[1]}")
            return "Could not identify hand sign", 0.0, [("Could not identify hand sign", 1.0)]

        prediction_probs = model.predict(input_array_scaled)[0]

        top_indices = prediction_probs.argsort()[-3:][::-1]
        top_preds = [(CLASS_NAMES[i], prediction_probs[i]) for i in top_indices]

        letter, confidence = top_preds[0]
        return letter, confidence, top_preds

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return "Could not identify hand sign", 0.0, [("Could not identify hand sign", 1.0)]

# 🚀 Main app
def main():
    st.title("🤟 Snapshot ASL Detector")
    st.markdown("Capture a photo using your webcam to predict ASL letters. Try forming the phrase **HELLO WORLD**!")

    # ✅ Upload-only navigation
    st.markdown("### 🖼️ Upload Mode")

    if st.button("Upload Mode"):
        st.switch_page("pages/app_upload.py")


    st.markdown("Tip: Use good lighting, hold hand steady, and position it clearly in the frame for better detection.")

    if 'sequence' not in st.session_state:
        st.session_state.sequence = []

    st.markdown("---")
    st.subheader("Capture ASL Letter")
    webcam_image = st.camera_input("Click below to capture")

    if webcam_image:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
            tmp_file.write(webcam_image.getvalue())
            image = cv2.imread(tmp_file.name)

            st.image(image, caption="📷 Raw Input Image", channels="BGR")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)
            image_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

            results = mp_hands_instance.process(image_rgb)
            st.write(f"🧪 MediaPipe result: `{results.multi_hand_landmarks}`")
            st.image(image_rgb, caption="🖼️ Enhanced Image for Detection", channels="RGB")

        if image is None:
            st.error("Failed to load image. Please try again.")
            return

        letter, confidence, top_preds = predict_image(image)
        
        if letter == 'nothing':
            letter = "No hand sign detected"

        st.image(image, caption="📷 Snapshot Image", channels="BGR", use_column_width=True)
        st.write(f"📐 Image shape: `{image.shape}`")
        st.write("🔍 Prediction Debug Info:")
        st.write(f"Predicted letter: `{letter}`")
        st.write(f"Confidence: `{confidence:.2f}`")

        if letter != "Could not identify hand sign":
            results = mp_hands_instance.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Compute full features instead of just flattened coords
                normalized_array = normalize_landmarks(hand_landmarks.landmark)
                normalized_flat = normalized_array.flatten()
                spread = get_finger_spread(hand_landmarks.landmark)
                v_thumb = normalized_array[4]
                v_index = normalized_array[8]
                v_middle = normalized_array[12]
                angle_thumb_index = get_angle(v_thumb, v_index)
                angle_index_middle = get_angle(v_index, v_middle)
                curv_thumb = get_finger_curvature(normalized_array, [1,2,3,4])
                curv_index = get_finger_curvature(normalized_array, [5,6,7,8])
                curv_middle = get_finger_curvature(normalized_array, [9,10,11,12])
                curv_ring = get_finger_curvature(normalized_array, [13,14,15,16])
                curv_pinky = get_finger_curvature(normalized_array, [17,18,19,20])
                full_features = np.concatenate([normalized_flat, [spread, angle_thumb_index, angle_index_middle], [curv_thumb, curv_index, curv_middle, curv_ring, curv_pinky]])
                st.write("🧠 Full input features (71 values):")
                st.write(full_features.tolist())

        st.image(image, caption=f"🖼️ Prediction: `{letter.upper()}`", channels="BGR")
        st.markdown(f"### ✅ Letter: `{letter.upper()}` — Confidence: `{confidence:.2f}`")
        st.markdown("🔝 **Top 3 Predictions:**")
        for label, conf in top_preds:
            st.write(f"- `{label}`: {conf:.2f}")

        spoken_text = "No hand sign detected" if letter == "Could not identify hand sign" else letter
        audio_buffer = speak_text(spoken_text)
        st.markdown(get_audio_download_link(audio_buffer), unsafe_allow_html=True)

        if letter not in ["Could not identify hand sign"]:
            st.session_state.sequence.append(letter)
            if len(st.session_state.sequence) > 50:
                st.session_state.sequence = st.session_state.sequence[-50:]

        current = ''.join([l.upper() for l in st.session_state.sequence])
        longest_word = max((word for j in range(len(current), 1, -1)
                          for word in [current[-j:]] if word in nltk_words),
                          key=len, default='')

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

# 🏁 Entry point
if __name__ == "__main__":
    main()