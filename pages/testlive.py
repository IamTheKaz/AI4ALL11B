import streamlit as st
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Dummy model and class names for illustration
model = ...  # Load your trained model here
CLASS_NAMES = [...]  # Your gesture class names

def extract_landmark_array(hand_landmarks):
    return np.array([lm.x for lm in hand_landmarks.landmark] +
                    [lm.y for lm in hand_landmarks.landmark] +
                    [lm.z for lm in hand_landmarks.landmark])

def predict_image(image_pil):
    # Ensure it's a PIL Image
    if not isinstance(image_pil, Image.Image):
        image_pil = Image.open(image_pil)

    # Convert to NumPy array
    image_np = np.array(image_pil)

    # Drop alpha channel if present
    if image_np.shape[-1] == 4:
        image_np = image_np[:, :, :3]

    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Convert back to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return "blank", 0.0, [("blank", 1.0)], None

    hand_landmarks = results.multi_hand_landmarks[0]
    mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    landmark_array = extract_landmark_array(hand_landmarks).reshape(1, -1)
    prediction_probs = model.predict(landmark_array)[0]
    pred_index = np.argmax(prediction_probs)
    prediction = CLASS_NAMES[pred_index]
    confidence = prediction_probs[pred_index]
    top_3 = [(CLASS_NAMES[i], prediction_probs[i]) for i in np.argsort(prediction_probs)[-3:][::-1]]

    return prediction, confidence, top_3, extract_landmark_array(hand_landmarks)

# Streamlit UI
st.title("Live Hand Gesture Recognition")

image = st.camera_input("Show your hand to the camera")

if image:
    image_pil = Image.open(image)
    st.image(image_pil, caption="Live Preview", channels="RGB")

    letter, confidence, top_3, current_landmarks = predict_image(image_pil)

    if current_landmarks is not None:
        if letter != st.session_state.get("last_prediction", ""):
            st.session_state.last_prediction = letter
            st.success(f"Predicted: {letter} ({confidence:.2f})")
            st.write("Top 3 Predictions:")
            for label, conf in top_3:
                st.write(f"- {label}: {conf:.2f}")
    else:
        st.warning("No hand detected. Try again.")