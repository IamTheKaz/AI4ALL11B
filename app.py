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
import pickle
import gc

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

# 📦 Setup - Optimized loading
@st.cache_data
def load_nltk_words():
    try:
        return set(words.words())
    except LookupError:
        nltk.download('words')
        return set(words.words())

# Only load NLTK words when actually needed
def get_nltk_words():
    if 'nltk_words' not in st.session_state:
        st.session_state.nltk_words = load_nltk_words()
    return st.session_state.nltk_words

# 🖐️ MediaPipe setup - More conservative settings
@st.cache_resource
def init_mediapipe():
    try:
        mp_hands = mp.solutions.hands
        mp_hands_instance = mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            min_detection_confidence=0.5,  # Lowered back for better detection
            min_tracking_confidence=0.5
        )
        mp_drawing = mp.solutions.drawing_utils
        HAND_CONNECTIONS = getattr(mp_hands, 'HAND_CONNECTIONS', None)
        return mp_hands_instance, mp_drawing, HAND_CONNECTIONS
    except Exception as e:
        st.error(f"MediaPipe initialization failed: {e}")
        st.stop()

mp_hands_instance, mp_drawing, HAND_CONNECTIONS = init_mediapipe()

IMG_SIZE = 224

# 🧠 Load model, scaler, and label encoder - Optimized
@st.cache_resource
def load_model_artifacts():
    try:
        # Load with memory optimization
        model = tf.keras.models.load_model("asl_model.h5", compile=False)
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
        
        # Force garbage collection after loading
        gc.collect()
        
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

# 🎯 Adjusted confidence thresholds for better accuracy
CONFIDENCE_THRESHOLDS = {
    'high': 0.75,      # Lowered from 0.85
    'medium': 0.55,    # Lowered from 0.65  
    'low': 0.35        # Lowered from 0.45
}

# 🔊 Speech synthesis - Cached and optimized
@st.cache_data(ttl=300)  # Cache for 5 minutes
def speak_text(text):
    """Cached TTS to avoid regenerating same audio"""
    try:
        if len(text) > 50:  # Limit text length
            text = text[:50]
        tts = gTTS(text=text, lang='en', slow=False)
        with io.BytesIO() as f:
            tts.write_to_fp(f)
            f.seek(0)
            audio_data = f.read()
        return audio_data
    except Exception as e:
        st.error(f"Speech synthesis failed: {e}")
        return b''

def get_audio_player(audio_data):
    """Create HTML audio player"""
    if audio_data:
        b64 = base64.b64encode(audio_data).decode()
        return f'<audio autoplay src="data:audio/mp3;base64,{b64}"></audio>'
    return ""

# Feature extraction functions - Optimized with error handling
def normalize_landmarks(landmarks):
    try:
        WRIST_IDX = 0
        MIDDLE_MCP_IDX = 9
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])  # Keep default float64
        origin = points[WRIST_IDX]
        points -= origin
        ref_point = points[MIDDLE_MCP_IDX]
        angle = np.arctan2(ref_point[1], ref_point[0])
        rot_matrix = np.array([
            [np.cos(-angle), -np.sin(-angle), 0],
            [np.sin(-angle),  np.cos(-angle), 0],
            [0,               0,              1]
        ], dtype=np.float64)  # Match training data type
        points = points @ rot_matrix.T
        return points
    except Exception as e:
        st.error(f"Landmark normalization failed: {e}")
        return None

def get_finger_spread(landmarks):
    try:
        x_vals = [landmarks[8].x, landmarks[12].x, landmarks[16].x]
        return max(x_vals) - min(x_vals)
    except:
        return 0.0

def get_angle(v1, v2):
    try:
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        return np.arccos(cos_theta)
    except:
        return 0.0

def get_finger_curvature(points, finger_joints):
    try:
        straight_dist = np.linalg.norm(points[finger_joints[3]] - points[finger_joints[0]])
        path_length = sum(np.linalg.norm(points[finger_joints[i+1]] - points[finger_joints[i]]) for i in range(3))
        return straight_dist / path_length if path_length > 0 else 1.0
    except:
        return 1.0

def extract_features(hand_landmarks):
    """Extract all 71 features from hand landmarks - Optimized"""
    try:
        normalized_array = normalize_landmarks(hand_landmarks.landmark)
        if normalized_array is None:
            return None
            
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
        
        return features  # Keep original dtype for model compatibility
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        return None

def deployment_prediction(image, results):
    """Production-ready prediction with confidence handling - Optimized"""
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

        # Extract features first
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
        top_predictions = [(CLASS_NAMES[i], float(prediction_probs[i])) for i in top_indices]
        
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
            'confidence': float(confidence),
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

def display_prediction_results(result, show_landmarks=False):
    """Display prediction results with appropriate styling - Simplified"""
    
    status = result['status']
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Status-based styling
    if status == 'high_confidence':
        st.success(f"🎯 **{prediction.upper()}** (Confidence: {confidence:.1%})")
    elif status == 'medium_confidence':
        st.warning(f"🤔 **{prediction.upper()}** (Confidence: {confidence:.1%})")
    elif status in ['low_confidence', 'very_low_confidence']:
        st.error(f"❌ **{prediction.upper()}** (Confidence: {confidence:.1%})")
    elif status == 'no_hand_detected':
        st.info("👋 No hand detected - make sure your hand is visible")
    else:
        st.error(f"⚠️ {result['message']}")
    
    # Show only top prediction unless debugging
    if show_landmarks and len(result['top_predictions']) > 1:
        st.markdown("**Top 3 Predictions:**")
        for i, (pred, conf) in enumerate(result['top_predictions'][:3], 1):
            if i == 1:
                st.markdown(f"{i}. **{pred}** - {conf:.1%} 🏆")
            else:
                st.markdown(f"{i}. {pred} - {conf:.1%}")
    
    return result

# 🚀 Main app - Streamlined
def main():
    st.title("🤟 ASL Letter Detector")
    st.markdown("**Capture photos to detect ASL letters**")

    # Initialize session state - minimal
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []

    # Tips section - collapsed by default
    with st.expander("💡 Tips for Better Recognition"):
        st.markdown("""
        - **Lighting**: Good, even lighting
        - **Background**: Plain background
        - **Position**: Center hand in frame
        - **Distance**: Arm's length from camera
        """)

    st.markdown("---")
    
    webcam_image = st.camera_input("📸 Capture ASL Letter")

    if webcam_image:
        # Process image with memory management
        try:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
                tmp_file.write(webcam_image.getvalue())
                image = cv2.imread(tmp_file.name)

            if image is None:
                st.error("Failed to load image. Please try again.")
                return

            # Process image - simplified
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_flipped = cv2.flip(image_rgb, 1)
            results = mp_hands_instance.process(image_flipped)
            
            # Make prediction
            result = deployment_prediction(image, results)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image_flipped, caption="📷 Captured Image", channels="RGB", width=300)
            
            with col2:
                display_prediction_results(result)
                
                # Show confidence meter
                if result['prediction'] != 'nothing':
                    st.progress(result['confidence'])
            
            # Text-to-speech for confident predictions only
            if result['status'] in ['high_confidence', 'medium_confidence'] and result['prediction'] != 'nothing':
                audio_data = speak_text(result['prediction'])
                if audio_data:
                    st.markdown(get_audio_player(audio_data), unsafe_allow_html=True)
            
            # Sequence management - simplified
            MAX_SEQUENCE = 15  # Reduced from 20
            if result['status'] in ['high_confidence', 'medium_confidence'] and result['prediction'] != 'nothing':
                if st.button(f"➕ Add '{result['prediction'].upper()}' to sequence"):
                    st.session_state.sequence.append(result['prediction'].upper())
                    if len(st.session_state.sequence) > MAX_SEQUENCE:
                        st.session_state.sequence = st.session_state.sequence[-MAX_SEQUENCE:]
                    st.success(f"Added {result['prediction'].upper()}!")
            
            # Display sequence
            if st.session_state.sequence:
                st.markdown("---")
                sequence_str = " ".join(st.session_state.sequence)
                st.markdown(f"**📝 Sequence:** {sequence_str}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear"):
                        st.session_state.sequence = []
                        st.rerun()
                
                with col2:
                    if st.button("⬅️ Remove Last"):
                        if st.session_state.sequence:
                            st.session_state.sequence.pop()
                            st.rerun()
                
                # Word detection - only for longer sequences
                if len(st.session_state.sequence) >= 3:
                    nltk_words = get_nltk_words()
                    current = ''.join(st.session_state.sequence).upper()
                    longest_word = ""
                    
                    # Check last 10 characters only for performance
                    check_length = min(10, len(current))
                    for j in range(check_length, 2, -1):
                        word = current[-j:]
                        if word.lower() in nltk_words and len(word) > len(longest_word):
                            longest_word = word
                            break
                    
                    if longest_word and len(longest_word) >= 3:
                        st.success(f"🗣 Word Detected: **{longest_word}**")
                        if 'last_word' not in st.session_state or st.session_state.last_word != longest_word:
                            audio_data = speak_text(longest_word)
                            if audio_data:
                                st.markdown(get_audio_player(audio_data), unsafe_allow_html=True)
                                st.session_state.last_word = longest_word

            # Force cleanup
            del image, image_rgb, image_flipped, results
            gc.collect()
            
        except Exception as e:
            st.error(f"Processing error: {e}")
            gc.collect()

if __name__ == "__main__":
    main()