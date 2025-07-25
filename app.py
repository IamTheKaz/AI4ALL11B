import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from streamlit_camera_input_live import camera_input_live
import nltk
from gtts import gTTS
import os

# Load the ASL model
model = load_model('best_asl_model.h5')

# Define class names (adjust based on your model, excluding 'space')
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize NLTK
nltk.download('words', quiet=True)

def preprocess_frame(frame):
    # Resize to match model input (e.g., 64x64, adjust as needed)
    frame = cv2.resize(frame, (64, 64))
    # Convert to RGB (if model expects RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    frame = frame / 255.0
    # Add batch dimension
    frame = np.expand_dims(frame, axis=0)
    return frame

def predict_letter(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return CLASS_NAMES[predicted_class]

def main():
    st.title("ASL Letter Predictor - Live Feed")
    st.write("Use your webcam to predict ASL letters in real-time.")

    # Initialize session state for predicted letters
    if 'predicted_letters' not in st.session_state:
        st.session_state.predicted_letters = []

    # Get live webcam frame
    frame = camera_input_live()
    
    if frame is not None:
        # Convert BytesIO to OpenCV format
        frame_array = np.frombuffer(frame.getvalue(), np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        # Display the frame
        st.image(frame, channels="BGR", caption="Live Webcam Feed")
        
        # Predict the letter
        predicted_letter = predict_letter(frame)
        st.write(f"Predicted Letter: **{predicted_letter}**")
        
        # Append to session state
        st.session_state.predicted_letters.append(predicted_letter)
        
        # Display recent predictions
        st.write("Recent Predictions:", ", ".join(st.session_state.predicted_letters[-5:]))
        
        # Convert predictions to text and audio (example for "HELLO")
        if len(st.session_state.predicted_letters) >= 5:
            word = "".join(st.session_state.predicted_letters[-5:])
            if word.lower() in nltk.corpus.words.words():
                st.write(f"Recognized Word: **{word}**")
                tts = gTTS(word)
                tts.save("output.mp3")
                st.audio("output.mp3")
                # Clear predictions after forming a word
                st.session_state.predicted_letters = []

if __name__ == "__main__":
    main()