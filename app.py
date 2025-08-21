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
import joblib


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
    mp_hands_instance = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.7,  # Increased from 0.5 for better quality
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils
    HAND_CONNECTIONS = getattr(mp_hands, 'HAND_CONNECTIONS', None)
    if HAND_CONNECTIONS is None:
        st.warning("HAND_CONNECTIONS not detected. Landmark drawing will be disabled.")
except Exception as e:
    st.error(f"MediaPipe initialization failed: {e}")
    st.stop()

IMG_SIZE = 224

# 🧠 Load model, scaler, and label encoder
@st.cache_resource
def load_model_artifacts():
    try:
        model = tf.keras.models.load_model("asl_model.h5")
        scaler = joblib.load("scaler.pkl")
        
        # Load label encoder if available, otherwise create mapping
        try:
            with open("label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)
            class_names = label_encoder.classes_
        except FileNotFoundError:
            st.warning("Label encoder not found. Using default mapping.")
            class_names = [chr(i) for i in range(65, 91)] + ['nothing']
            label_encoder = None
        
        return model, scaler, label_encoder, class_names
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

model, scaler, label_encoder, CLASS_NAMES = load_model_artifacts()

# Validate model input shape
expected_features = 71
actual_input_shape = model.input_shape[1] if len(model.input_shape) > 1 else model.input_shape[0]
if actual_input_shape != expected_features:
    st.error(f"Model expects {actual_input_shape} features, but code generates {expected_features}")
    st.stop()

st.write(f"🧪 Model expects {actual_input_shape} features")

# 🎯 Confidence thresholds based on your evaluation
CONFIDENCE_THRESHOLDS = {
    'high': 0.85,      # Show result immediately
    'medium': 0.65,    # Show with uncertainty indicator  
    'low': 0.45        # Show "nothing" or ask to retry
}

# 🔊 Speech synthesis (improved)
@st.cache_data
def speak_text(text):
    """Cached TTS to avoid regenerating same audio"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        with io.BytesIO() as f:
            tts.write_to_fp(f)
            f.seek(0)
            return f.read()
    except Exception as e:
        st.error(f"Speech synthesis failed: {e}")
        return b''

def get_audio_player(audio_data):
    """Create HTML audio player"""
    if audio_data:
        b64 = base64.b64encode(audio_data).decode()
        return f'<audio controls><source src="data:audio/mp3;base64,{b64}" type="audio/mpeg"></audio>'
    return ""

# Feature extraction functions (same as training)
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
    return points

def get_finger_spread(landmarks):
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
    return np.arccos(cos_theta)

def get_finger_curvature(points, finger_joints):
    straight_dist = np.linalg.norm(points[finger_joints[3]] - points[finger_joints[0]])
    path_length = sum(np.linalg.norm(points[finger_joints[i+1]] - points[finger_joints[i]]) for i in range(3))
    return straight_dist / path_length if path_length > 0 else 1.0

def extract_features(hand_landmarks):
    """Extract all 71 features from hand landmarks"""
    try:
        normalized_array = normalize_landmarks(hand_landmarks.landmark)
        normalized_flat = normalized_array.flatten()  # 63 values
        
        # Additional features (8 values)
        spread = get_finger_spread(hand_landmarks.landmark)
        v_thumb = normalized_array[4]
        v_index = normalized_array[8]  
        v_middle = normalized_array[12]
        angle_thumb_index = get_angle(v_thumb, v_index)
        angle_index_middle = get_angle(v_index, v_middle)
        
        curvatures = [
            get_finger_curvature(normalized_array, [1,2,3,4]),   # thumb
            get_finger_curvature(normalized_array, [5,6,7,8]),   # index
            get_finger_curvature(normalized_array, [9,10,11,12]), # middle
            get_finger_curvature(normalized_array, [13,14,15,16]), # ring
            get_finger_curvature(normalized_array, [17,18,19,20])  # pinky
        ]
        
        # Combine all features (63 + 1 + 2 + 5 = 71)
        features = np.concatenate([
            normalized_flat, 
            [spread, angle_thumb_index, angle_index_middle], 
            curvatures
        ])
        
        return features
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

def deployment_prediction(image, results):
    """Production-ready prediction with confidence handling"""
    try:
        # Check if hand is detected
        if not results.multi_hand_landmarks:
            return {
                'prediction': 'nothing',
                'confidence': 1.0,
                'status': 'no_hand_detected',
                'top_predictions': [('nothing', 1.0)],
                'message': 'No hand detected'
            }

        # Check confidence score if available
        if results.multi_handedness:
            detection_confidence = results.multi_handedness[0].classification[0].score
            if detection_confidence < 0.6:
                return {
                    'prediction': 'nothing',
                    'confidence': detection_confidence,
                    'status': 'low_detection_confidence',
                    'top_predictions': [('nothing', detection_confidence)],
                    'message': f'Hand detection confidence too low: {detection_confidence:.2f}'
                }

        # Extract features
        hand_landmarks = results.multi_hand_landmarks[0]
        features = extract_features(hand_landmarks)
        
        if features is None or len(features) != expected_features:
            return {
                'prediction': 'nothing',
                'confidence': 0.0,
                'status': 'feature_extraction_failed',
                'top_predictions': [('nothing', 0.0)],
                'message': f'Feature extraction failed. Expected {expected_features}, got {len(features) if features is not None else 0}'
            }

        # Scale features and predict
        features_scaled = scaler.transform(features.reshape(1, -1))
        prediction_probs = model.predict(features_scaled, verbose=0)[0]
        
        # Get top 3 predictions
        top_indices = prediction_probs.argsort()[-3:][::-1]
        top_predictions = [(CLASS_NAMES[i], prediction_probs[i]) for i in top_indices]
        
        prediction = top_predictions[0][0]
        confidence = top_predictions[0][1]
        
        # Determine status based on confidence thresholds
        if confidence >= CONFIDENCE_THRESHOLDS['high']:
            status = 'high_confidence'
            message = f'Confident prediction: {prediction}'
        elif confidence >= CONFIDENCE_THRESHOLDS['medium']:
            status = 'medium_confidence'
            message = f'Somewhat confident: {prediction}'
        elif confidence >= CONFIDENCE_THRESHOLDS['low']:
            status = 'low_confidence'
            message = f'Low confidence: {prediction}'
        else:
            status = 'very_low_confidence'
            prediction = 'nothing'
            message = 'Unable to recognize clearly'
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'status': status,
            'top_predictions': top_predictions,
            'message': message
        }
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return {
            'prediction': 'nothing',
            'confidence': 0.0,
            'status': 'error',
            'top_predictions': [('nothing', 0.0)],
            'message': f'Prediction error: {str(e)}'
        }

def display_prediction_results(result, image_with_landmarks=None):
    """Display prediction results with appropriate styling"""
    
    status = result['status']
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Status-based styling
    if status == 'high_confidence':
        st.success(f"🎯 **{prediction.upper()}** (Confidence: {confidence:.1%})")
        st.markdown("✅ High confidence prediction")
    elif status == 'medium_confidence':
        st.warning(f"🤔 **{prediction.upper()}** (Confidence: {confidence:.1%})")
        st.markdown("⚠️ Medium confidence - try adjusting hand position")
    elif status in ['low_confidence', 'very_low_confidence']:
        st.error(f"❓ **{prediction.upper()}** (Confidence: {confidence:.1%})")
        st.markdown("🔄 Low confidence - please try again with better positioning")
    elif status == 'no_hand_detected':
        st.info("👋 No hand detected")
        st.markdown("💡 **Tip:** Make sure your hand is clearly visible in the frame")
    else:
        st.error(f"⚠️ {result['message']}")
    
    # Show top predictions
    if len(result['top_predictions']) > 1:
        st.markdown("**Top 3 Predictions:**")
        for i, (pred, conf) in enumerate(result['top_predictions'], 1):
            if i == 1:
                st.markdown(f"{i}. **{pred}** - {conf:.1%} 🏆")
            else:
                st.markdown(f"{i}. {pred} - {conf:.1%}")
    
    # Display enhanced image if available
    if image_with_landmarks is not None:
        st.image(image_with_landmarks, caption="📍 Hand landmarks detected", channels="RGB")
    
    return result

# 🚀 Main app
def main():
    st.title("🤟 Advanced ASL Detector")
    st.markdown("""
    **Capture photos to detect ASL letters with confidence scoring.**
    
    🎯 **Confidence Levels:**
    - 🟢 High (>85%): Immediate prediction
    - 🟡 Medium (65-85%): Good prediction, try adjusting if needed  
    - 🔴 Low (<65%): Please try again with better hand positioning
    """)

    # Navigation
    if st.button("📁 Upload Mode"):
        st.switch_page("pages/app_upload.py")

    # Tips section
    with st.expander("💡 Tips for Better Recognition"):
        st.markdown("""
        - **Lighting**: Use good, even lighting
        - **Background**: Plain, contrasting background works best
        - **Hand Position**: Center your hand in the frame
        - **Stability**: Keep your hand steady during capture
        - **Distance**: Keep hand at arm's length from camera
        """)

    # Initialize session state
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

    st.markdown("---")
    st.subheader("📸 Capture ASL Letter")
    
    webcam_image = st.camera_input("Click below to capture")

    if webcam_image:
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
            tmp_file.write(webcam_image.getvalue())
            image = cv2.imread(tmp_file.name)

        if image is None:
            st.error("Failed to load image. Please try again.")
            return

        # Process image
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_flipped = cv2.flip(image_rgb, 1)  # Mirror for better UX
        results = mp_hands_instance.process(image_flipped)
        
        # Create annotated image
        image_with_landmarks = image_flipped.copy()
        if results.multi_hand_landmarks and HAND_CONNECTIONS:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_with_landmarks, 
                    hand_landmarks, 
                    HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )

        # Make prediction
        result = deployment_prediction(image, results)
        
        # Display results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image_flipped, caption="📷 Captured Image", channels="RGB")
        
        with col2:
            display_prediction_results(result, image_with_landmarks)
        
        # Text-to-speech
        if result['prediction'] != 'nothing' and result['status'] in ['high_confidence', 'medium_confidence']:
            audio_data = speak_text(result['prediction'])
            if audio_data:
                st.markdown("🔊 **Audio:**")
                st.markdown(get_audio_player(audio_data), unsafe_allow_html=True)
        
        # Add to sequence if confident enough
        if result['status'] in ['high_confidence', 'medium_confidence'] and result['prediction'] != 'nothing':
            if st.button(f"➕ Add '{result['prediction'].upper()}' to sequence"):
                st.session_state.sequence.append(result['prediction'].upper())
                st.session_state.prediction_history.append(result)
                st.success(f"Added {result['prediction'].upper()} to sequence!")
        
        # Display current sequence
        if st.session_state.sequence:
            st.markdown("---")
            st.markdown("**📝 Current Sequence:**")
            sequence_str = " ".join(st.session_state.sequence)
            st.markdown(f"### {sequence_str}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🗑️ Clear Sequence"):
                    st.session_state.sequence = []
                    st.session_state.prediction_history = []
                    st.rerun()
            
            with col2:
                if st.button("⬅️ Remove Last"):
                    if st.session_state.sequence:
                        st.session_state.sequence.pop()
                        if st.session_state.prediction_history:
                            st.session_state.prediction_history.pop()
                        st.rerun()
            
            with col3:
                if st.button("🔊 Speak Sequence"):
                    audio_data = speak_text(sequence_str)
                    if audio_data:
                        st.markdown(get_audio_player(audio_data), unsafe_allow_html=True)

        # Debug info (collapsible)
        with st.expander("🔧 Debug Information"):
            st.json({
                'prediction': result['prediction'],
                'confidence': f"{result['confidence']:.3f}",
                'status': result['status'],
                'message': result['message'],
                'mediapipe_detected': bool(results.multi_hand_landmarks),
                'detection_confidence': results.multi_handedness[0].classification[0].score if results.multi_handedness else 'N/A'
            })

if __name__ == "__main__":
    main()