import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import nltk
from gtts import gTTS
from io import BytesIO

# Load the ASL model with error handling
try:
    model = load_model('best_asl_model.h5')
except Exception as e:
    st.error(f"Failed to load model: {e}. Please ensure 'best_asl_model.h5' is compatible with TensorFlow 2.12.0.")
    st.stop()

# Define class names (excluding 'space')
CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Initialize NLTK
nltk.download('words', quiet=True)

def preprocess_frame(frame):
    # Resize to 32x32 (model input)
    frame = cv2.resize(frame, (32, 32))
    # Convert to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Normalize pixel values
    frame = frame / 255.0
    # Add batch and channel dimensions
    frame = np.expand_dims(frame, axis=0)
    frame = np.expand_dims(frame, axis=-1)  # Shape: (1, 32, 32, 1)
    return frame

def predict_letter(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    predicted_class = np.argmax(prediction, axis=1)[0]
    if predicted_class >= len(CLASS_NAMES):
        raise ValueError(f"Predicted class index {predicted_class} exceeds CLASS_NAMES length {len(CLASS_NAMES)}")
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
        
        # Validate frame
        if frame is not None and frame.size > 0:
            st.image(frame, channels="BGR", caption="Captured Image")
            
            # Predict the letter
            try:
                predicted_letter = predict_letter(frame)
                st.write(f"Predicted Letter: **{predicted_letter}**")
                
                # Append to session state
                st.session_state.predicted_letters.append(predicted_letter)
                
                # Display recent predictions (last 10 for HELLOWORLD)
                st.write("Recent Predictions:", ", ".join(st.session_state.predicted_letters[-10:]))
                
                # Check for target sequence
                target_sequence = ['H', 'E', 'L', 'L', 'O', 'W', 'O', 'R', 'L', 'D']
                if len(st.session_state.predicted_letters) >= len(target_sequence):
                    recent = st.session_state.predicted_letters[-len(target_sequence):]
                    if recent == target_sequence:
                        word = "".join(recent)
                        st.write(f"Recognized Word: **{word}**")
                        try:
                            audio_buffer = BytesIO()
                            tts = gTTS(word)
                            tts.write_to_fp(audio_buffer)
                            audio_buffer.seek(0)
                            st.audio(audio_buffer, format="audio/mp3")
                            st.session_state.predicted_letters = []
                        except Exception as e:
                            st.error(f"Audio generation failed: {e}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()