import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import base64
from tensorflow.keras.preprocessing.image import img_to_array
from gtts import gTTS
from io import BytesIO
from camera_input_live import camera_input_live

# Setup
IMG_HEIGHT, IMG_WIDTH = 32, 32
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['del', 'nothing', 'space']

def speak_text(text):
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
    model.load_weights("best_asl_model.h5")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.set_page_config(page_title="Live ASL Predictor", layout="centered")
    st.title("🖐 ASL Letter Predictor (Live Webcam)")
    st.markdown("Click below to begin live ASL detection from your webcam.")

    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
    if 'last_letter' not in st.session_state:
        st.session_state.last_letter = None
    if 'last_confidence' not in st.session_state:
        st.session_state.last_confidence = 0.0
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0

    start_stream = st.button("Start Live Predictions")
    if not start_stream:
        st.info("Webcam feed is inactive. Click the button above to begin.")
    else:
        model = load_model()
        image = camera_input_live()

        if image is not None:
            st.session_state.frame_count += 1
            if st.session_state.frame_count % 5 != 0:
                st.stop()

            bytes_data = image.getvalue()
            img_np = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if img_np is not None:
                processed = preprocess(img_np)
                predictions = model.predict(processed, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0])
                letter = CLASS_NAMES[predicted_idx]

                cv2.putText(img_np, f"{letter} ({confidence:.2f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                st.image(img_np, channels="BGR", caption=f"Predicted: {letter} ({confidence:.2f})")

                if confidence > 0.7:
                    repeat_count = sum(1 for i in range(1, min(3, len(st.session_state.sequence)+1))
                                       if st.session_state.sequence[-i] == letter)

                    if repeat_count < 2:
                        st.session_state.sequence.append(letter)
                        st.session_state.last_letter = letter
                        st.session_state.last_confidence = confidence

                        audio = speak_text(letter)
                        st.markdown(
                            f'<audio autoplay src="data:audio/mp3;base64,{base64.b64encode(audio.read()).decode()}"></audio>',
                            unsafe_allow_html=True
                        )

                st.markdown("### 🔡 Letter Sequence")
                st.write(" → " + " ".join(st.session_state.sequence[-15:]))
            else:
                st.warning("⚠️ Unable to decode webcam frame.")
        else:
            st.warning("No webcam input received.")

    # --- Mode Switching Section ---
    st.markdown("---")
    st.markdown("#### Try Alternate Input Modes:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📷 Snapshot Version"):
            st.switch_page("pages/app_snapshot.py")
    with col2:
        if st.button("🖼 Image Upload Version"):
            st.switch_page("pages/app_upload.py")

if __name__ == "__main__":
    main()