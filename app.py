import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from gtts import gTTS
from io import BytesIO

# Load the ASL model
model = load_model('best_asl_model.h5')

# Define class names (excluding 'space')
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize NLTK
nltk.download('words', quiet=True)

def preprocess_frame(frame):
    # Resize to match model input (e.g., 64x64, adjust if needed)
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
    st.title("ASL Letter Predictor - Snapshot Mode")
    st.write("Take a photo with your webcam to predict ASL letters.")

    # Initialize session state for predicted letters
    if 'predicted_letters' not in st.session_state:
        st.session_state.predicted_letters = []

    # Get snapshot
    frame = st.camera_input("Take a photo")

    if frame is not None:
        # Convert BytesIO to OpenCV format
        frame_array = np.frombuffer(frame.getvalue(), np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        
        # Display the frame
        st.image(frame, channels="BGR", caption="Captured Image")
        
        # Predict the letter
        predicted_letter = predict_letter(frame)
        st.write(f"Predicted Letter: **{predicted_letter}**")
        
        # Append to session state
        st.session_state.predicted_letters.append(predicted_letter)
        
        # Display recent predictions (last 10 for target sequence)
        st.write("Recent Predictions:", ", ".join(st.session_state.predicted_letters[-10:]))
        
        # Check for target sequence (e.g., HELLOWORLD)
        target_sequence = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']
        if len(st.session_state.predicted_letters) >= len(target_sequence):
            recent = st.session_state.predicted_letters[-len(target_sequence):]
            if recent == target_sequence:
                word = "".join(recent)
                st.write(f"Recognized Word: **{word}**")
                try:
                    # Stream audio to avoid file write issues
                    audio_buffer = BytesIO()
                    tts = gTTS(word)
                    tts.write_to_fp(audio_buffer)
                    audio_buffer.seek(0)
                    st.audio(audio_buffer, format="audio/mp3")
                    # Clear predictions after forming the target sequence
                    st.session_state.predicted_letters = []
                except Exception as e:
                    st.error(f"Audio generation failed: {e}")

if __name__ == "__main__":
    main()