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

# Hide sidebar and set page config
st.set_page_config(page_title="Enhanced ASL Snapshot Detector", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    [data-testid="stSidebar"] { display: none; }
    [data-testid="stSidebarNav"] { display: none; }
    [data-testid="stSidebarContent"] { display: none; }
    .css-1d391kg { display: none; }
    </style>
""", unsafe_allow_html=True)

# Setup - Optimized loading
@st.cache_data
def load_nltk_words():
    try:
        return set(words.words())
    except LookupError:
        nltk.download('words')
        return set(words.words())

def get_nltk_words():
    if 'nltk_words' not in st.session_state:
        st.session_state.nltk_words = load_nltk_words()
    return st.session_state.nltk_words

# MediaPipe setup
@st.cache_resource
def init_mediapipe():
    try:
        mp_hands = mp.solutions.hands
        mp_hands_instance = mp_hands.Hands(
            static_image_mode=True, 
            max_num_hands=1, 
            min_detection_confidence=0.5,
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

# Load model artifacts - Updated for enhanced features
@st.cache_resource
def load_model_artifacts():
    try:
        model = tf.keras.models.load_model("asl_model_improved.h5", compile=False)  # Updated model name
        scaler = joblib.load("scaler.pkl")
        
        try:
            with open("label_encoder.pkl", "rb") as f:
                label_encoder = pickle.load(f)
            class_names = label_encoder.classes_
        except FileNotFoundError:
            st.warning("Label encoder not found. Using default mapping.")
            class_names = [chr(i) for i in range(65, 91)] + ['nothing']
            label_encoder = None
        
        gc.collect()
        return model, scaler, label_encoder, class_names
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()

model, scaler, label_encoder, CLASS_NAMES = load_model_artifacts()

# Validate model input shape for enhanced features
expected_features = 155  # Fixed to match your training data
actual_input_shape = model.input_shape[1] if len(model.input_shape) > 1 else model.input_shape[0]
if actual_input_shape != expected_features:
    st.error(f"Model expects {actual_input_shape} features, but code generates {expected_features}")
    st.stop()
else:
    st.info(f"Model expects {expected_features} features (enhanced feature set)")

# Adjusted confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.75,
    'medium': 0.55,
    'low': 0.40
}

# Speech synthesis - Cached and optimized
@st.cache_data(ttl=300)
def speak_text(text):
    try:
        if len(text) > 50:
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
    if audio_data:
        b64 = base64.b64encode(audio_data).decode()
        return f'<audio autoplay src="data:audio/mp3;base64,{b64}"></audio>'
    return ""

# ENHANCED FEATURE EXTRACTION - Updated to match training pipeline
def normalize_landmarks_3d(landmarks):
    """Improved 3D normalization preserving depth information"""
    points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
    
    # Center on wrist
    wrist = points[0]
    points_centered = points - wrist
    
    # Scale based on hand span (wrist to middle finger tip)
    hand_span = np.linalg.norm(points_centered[12] - points_centered[0])
    if hand_span > 0:
        points_centered = points_centered / hand_span
    
    # Align to consistent orientation using wrist-middle MCP vector
    ref_vector = points_centered[9]  # Middle MCP
    if np.linalg.norm(ref_vector[:2]) > 0:
        angle = np.arctan2(ref_vector[1], ref_vector[0])
        cos_a, sin_a = np.cos(-angle), np.sin(-angle)
        
        # 2D rotation matrix for x,y coordinates
        rotation_2d = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        points_centered[:, :2] = points_centered[:, :2] @ rotation_2d.T
    
    return points_centered

def extract_finger_crossing_features(points_3d):
    """Enhanced features to detect finger crossings (crucial for H, R, etc.)"""
    features = []
    
    # Finger tip indices
    tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
    
    # Check all pairs of fingers for crossing/overlapping (10 pairs * 3 features = 30)
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            tip_i, tip_j = points_3d[tips[i]], points_3d[tips[j]]
            
            # Distance between fingertips
            distance = np.linalg.norm(tip_i - tip_j)
            features.append(distance)
            
            # Relative positioning (which finger is in front/behind)
            z_diff = tip_i[2] - tip_j[2]  # depth difference
            features.append(z_diff)
            
            # X-axis crossing detection
            x_diff = tip_i[0] - tip_j[0]
            features.append(x_diff)
    
    # Specific features for common crossing patterns (15 additional features)
    index_tip, middle_tip = points_3d[8], points_3d[12]
    index_pip, middle_pip = points_3d[6], points_3d[10]
    index_mcp, middle_mcp = points_3d[5], points_3d[9]
    
    # Check crossings at different levels
    features.append((index_pip[0] - middle_pip[0]) * (index_tip[0] - middle_tip[0]))
    features.append((index_mcp[0] - middle_mcp[0]) * (index_pip[0] - middle_pip[0]))
    features.append(np.linalg.norm(index_pip - middle_pip))
    
    # Ring-Middle crossing
    ring_tip, ring_pip = points_3d[16], points_3d[14]
    ring_mcp = points_3d[13]
    features.append((ring_pip[0] - middle_pip[0]) * (ring_tip[0] - middle_tip[0]))
    features.append(np.linalg.norm(ring_pip - middle_pip))
    
    # Pinky-Ring crossing
    pinky_tip, pinky_pip = points_3d[20], points_3d[18]
    features.append((pinky_pip[0] - ring_pip[0]) * (pinky_tip[0] - ring_tip[0]))
    features.append(np.linalg.norm(pinky_pip - ring_pip))
    
    # Thumb interactions with other fingers
    thumb_tip, thumb_ip = points_3d[4], points_3d[3]
    for finger_tip in [8, 12, 16, 20]:  # index, middle, ring, pinky
        finger_pos = points_3d[finger_tip]
        # Distance from thumb tip to finger
        features.append(np.linalg.norm(thumb_tip - finger_pos))
        # Whether thumb is "over" or "under" finger
        features.append(thumb_tip[2] - finger_pos[2])
    
    return features

def extract_enhanced_shape_features(points_3d):
    """Enhanced shape features for better letter discrimination"""
    features = []
    
    # Finger extension levels with more precision (5 fingers * 3 features = 15)
    finger_joints = [
        [1, 2, 3, 4],    # thumb
        [5, 6, 7, 8],    # index
        [9, 10, 11, 12], # middle
        [13, 14, 15, 16], # ring
        [17, 18, 19, 20] # pinky
    ]
    
    for joints in finger_joints:
        # Extension ratio
        full_length = np.linalg.norm(points_3d[joints[3]] - points_3d[joints[0]])
        joint_lengths = [np.linalg.norm(points_3d[joints[i+1]] - points_3d[joints[i]]) 
                        for i in range(3)]
        total_joint_length = sum(joint_lengths)
        extension_ratio = full_length / total_joint_length if total_joint_length > 0 else 1.0
        features.append(extension_ratio)
        
        # Curvature at PIP joint
        v1 = points_3d[joints[1]] - points_3d[joints[0]]  # MCP to PIP
        v2 = points_3d[joints[2]] - points_3d[joints[1]]  # PIP to DIP
        angle = calculate_angle(v1, v2)
        features.append(angle)
        
        # Overall finger straightness
        straight_dist = np.linalg.norm(points_3d[joints[3]] - points_3d[joints[0]])
        path_length = sum([np.linalg.norm(points_3d[joints[i+1]] - points_3d[joints[i]]) 
                          for i in range(3)])
        straightness = straight_dist / path_length if path_length > 0 else 1.0
        features.append(straightness)
    
    # Hand compactness measures (5 features)
    fingertips = [points_3d[i] for i in [4, 8, 12, 16, 20]]
    palm_center = np.mean([points_3d[i] for i in [0, 5, 9, 13, 17]], axis=0)
    
    tip_distances = [np.linalg.norm(tip - palm_center) for tip in fingertips]
    features.append(np.mean(tip_distances))  # Average distance
    features.append(np.std(tip_distances))   # Spread of distances
    features.append(np.max(tip_distances))   # Maximum reach
    features.append(np.min(tip_distances))   # Minimum distance (tucked fingers)
    
    # Overall hand span
    x_coords = [point[0] for point in fingertips]
    hand_width = max(x_coords) - min(x_coords)
    features.append(hand_width)
    
    return features

def extract_palm_relative_features(points_3d):
    """Features describing finger positions relative to palm landmarks"""
    features = []
    
    # Define palm landmarks
    wrist = points_3d[0]
    palm_base = np.mean([points_3d[i] for i in [5, 9, 13, 17]], axis=0)  # MCP joints
    
    # Fingertip positions relative to palm (5 fingers * 3 measures = 15)
    fingertips = [4, 8, 12, 16, 20]
    for tip_idx in fingertips:
        tip = points_3d[tip_idx]
        
        # Distance from palm base
        palm_distance = np.linalg.norm(tip - palm_base)
        features.append(palm_distance)
        
        # Height above palm (y-axis)
        height_above_palm = tip[1] - palm_base[1]
        features.append(height_above_palm)
        
        # Side-to-side position relative to palm center
        lateral_position = tip[0] - palm_base[0]
        features.append(lateral_position)
    
    return features

def extract_comprehensive_angles(points_3d):
    """Comprehensive angle features for better shape discrimination"""
    features = []
    
    # Angles between adjacent fingers
    fingertips = [4, 8, 12, 16, 20]
    palm_center = np.mean([points_3d[i] for i in [5, 9, 13, 17]], axis=0)
    
    # Vectors from palm center to each fingertip
    finger_vectors = [points_3d[tip] - palm_center for tip in fingertips]
    
    # Angles between adjacent finger vectors (4 angles)
    for i in range(len(finger_vectors) - 1):
        angle = calculate_angle(finger_vectors[i], finger_vectors[i + 1])
        features.append(angle)
    
    # Specific critical angles (8 more features)
    thumb_index_angle = calculate_angle(finger_vectors[0], finger_vectors[1])
    features.append(thumb_index_angle)
    
    index_middle_angle = calculate_angle(finger_vectors[1], finger_vectors[2])
    features.append(index_middle_angle)
    
    middle_ring_angle = calculate_angle(finger_vectors[2], finger_vectors[3])
    features.append(middle_ring_angle)
    
    ring_pinky_angle = calculate_angle(finger_vectors[3], finger_vectors[4])
    features.append(ring_pinky_angle)
    
    # Overall finger spread measures
    max_angle = max([calculate_angle(finger_vectors[i], finger_vectors[j]) 
                     for i in range(len(finger_vectors)) 
                     for j in range(i+1, len(finger_vectors))])
    features.append(max_angle)
    
    min_angle = min([calculate_angle(finger_vectors[i], finger_vectors[j]) 
                     for i in range(len(finger_vectors)) 
                     for j in range(i+1, len(finger_vectors))])
    features.append(min_angle)
    
    # Thumb specific angles
    thumb_to_index_mcp = calculate_angle(finger_vectors[0], points_3d[5] - palm_center)
    features.append(thumb_to_index_mcp)
    
    # Average finger spread
    all_angles = [calculate_angle(finger_vectors[i], finger_vectors[j]) 
                  for i in range(len(finger_vectors)) 
                  for j in range(i+1, len(finger_vectors))]
    features.append(np.mean(all_angles))
    
    return features

def calculate_angle(v1, v2):
    """Calculate angle between two vectors in radians"""
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def extract_enhanced_landmarks(hand_landmarks):
    """Enhanced landmark extraction matching training pipeline - returns ~160 features"""
    try:
        landmarks = hand_landmarks.landmark
        
        # Get normalized 3D points
        points_3d = normalize_landmarks_3d(landmarks)
        
        # Extract comprehensive features
        features = []
        
        # 1. Basic normalized coordinates (21 * 3 = 63 features)
        features.extend(points_3d.flatten())
        
        # 2. Enhanced finger relationships for crossed finger detection
        finger_cross_features = extract_finger_crossing_features(points_3d)
        features.extend(finger_cross_features)
        
        # 3. Improved hand shape descriptors
        shape_features = extract_enhanced_shape_features(points_3d)
        features.extend(shape_features)
        
        # 4. Finger positions relative to palm
        palm_relative_features = extract_palm_relative_features(points_3d)
        features.extend(palm_relative_features)
        
        # 5. Angle-based features for better letter discrimination
        angle_features = extract_comprehensive_angles(points_3d)
        features.extend(angle_features)
        
        return np.array(features)
        
    except Exception as e:
        st.error(f"Enhanced feature extraction failed: {e}")
        return None

def deployment_prediction(image, results):
    """Enhanced prediction with improved features"""
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

        # Extract enhanced features
        hand_landmarks = results.multi_hand_landmarks[0]
        features = extract_enhanced_landmarks(hand_landmarks)
        
        if features is None:
            return {
                'prediction': 'nothing',
                'confidence': 0.0,
                'status': 'feature_extraction_failed',
                'top_predictions': [('nothing', 0.0)],
                'message': 'Enhanced feature extraction failed'
            }
        
        # Validate feature count
        if len(features) != expected_features:
            return {
                'prediction': 'nothing',
                'confidence': 0.0,
                'status': 'feature_count_mismatch',
                'top_predictions': [('nothing', 0.0)],
                'message': f'Feature count mismatch. Expected {expected_features}, got {len(features)}'
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
        st.error(f"Enhanced prediction failed: {e}")
        return {
            'prediction': 'nothing',
            'confidence': 0.0,
            'status': 'error',
            'top_predictions': [('nothing', 0.0)],
            'message': f'Prediction error: {str(e)}'
        }

def display_prediction_results(result, show_landmarks=False, debug_mode=False, expected_letter=None):
    """Display prediction results with appropriate styling"""
    
    status = result['status']
    prediction = result['prediction']
    confidence = result['confidence']
    
    # Debug mode: Color coding based on correctness
    if debug_mode and expected_letter:
        is_correct = (prediction.upper() == expected_letter.upper())
        if status == 'no_hand_detected':
            st.error(f"⌘ **NO HAND DETECTED** (Expected: {expected_letter})")
        elif prediction == 'nothing':
            st.error(f"⌘ **NOTHING** (Expected: {expected_letter})")
        elif is_correct:
            st.success(f"✅ **{prediction.upper()}** (Confidence: {confidence:.1%}) - CORRECT!")
        else:
            st.error(f"⌘ **{prediction.upper()}** (Confidence: {confidence:.1%}) - Expected: {expected_letter}")
    else:
        # Normal mode: Status-based styling
        if status == 'high_confidence':
            st.success(f"🎯 **{prediction.upper()}** (Confidence: {confidence:.1%})")
        elif status == 'medium_confidence':
            st.warning(f"🤔 **{prediction.upper()}** (Confidence: {confidence:.1%})")
        elif status in ['low_confidence', 'very_low_confidence']:
            st.error(f"🔴 **{prediction.upper()}** (Confidence: {confidence:.1%}) - LOW CONFIDENCE")
        elif status == 'no_hand_detected':
            st.info("👋 No hand detected - make sure your hand is visible")
        else:
            st.error(f"⚠️ {result['message']}")
    
    # Show top predictions if requested
    if show_landmarks and len(result['top_predictions']) > 1:
        st.markdown("**Top 3 Predictions:**")
        for i, (pred, conf) in enumerate(result['top_predictions'][:3], 1):
            if i == 1:
                st.markdown(f"{i}. **{pred}** - {conf:.1%} 🏆")
            else:
                st.markdown(f"{i}. {pred} - {conf:.1%}")
    
    return result

# Main app - Updated for enhanced features
def main():
    st.title("🤟 Enhanced ASL Letter Detector")
    st.markdown("**Capture photos to detect ASL letters - Now with enhanced feature recognition!**")
    st.info(f"✨ Using enhanced feature set with {expected_features} features for improved accuracy")

    # Initialize session state
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
    if 'alphabet_test' not in st.session_state:
        st.session_state.alphabet_test = {
            'expected_letter': 'A',
            'current_index': 0,
            'results': {},
            'alphabet': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        }

    # Debug mode toggle
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("")
    with col2:
        debug_toggle = st.checkbox("🔬 Alphabet Test Mode", value=st.session_state.debug_mode)
        if debug_toggle != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_toggle
            st.rerun()

    # Tips section
    with st.expander("💡 Tips for Better Recognition"):
        st.markdown("""
        - **Lighting**: Good, even lighting
        - **Background**: Plain background  
        - **Position**: Center hand in frame
        - **Distance**: Arm's length from camera
        - **Enhanced Features**: Better detection of finger crossings (H, R, K) and complex shapes
        """)

    st.markdown("---")
    
    st.info("💡 **For best results:** Use your non-dominant hand to click 'Take Photo' so you can sign with your dominant right hand")
    
    webcam_image = st.camera_input("📸 Capture ASL Letter")

    if webcam_image:
        try:
            with tempfile.NamedTemporaryFile(delete=True, suffix=".jpg") as tmp_file:
                tmp_file.write(webcam_image.getvalue())
                image = cv2.imread(tmp_file.name)

            if image is None:
                st.error("Failed to load image. Please try again.")
                return

            # Process image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = mp_hands_instance.process(image_rgb)
            
            # Make prediction with enhanced features
            result = deployment_prediction(image, results)
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image_rgb, caption="📷 Captured Image", channels="RGB", width=300)
            
            with col2:
                # Debug mode: Show expected letter
                if st.session_state.debug_mode:
                    expected = st.session_state.alphabet_test['expected_letter']
                    display_prediction_results(result, debug_mode=True, expected_letter=expected)
                else:
                    display_prediction_results(result)
                
                # Show confidence meter
                if result['prediction'] != 'nothing':
                    st.progress(result['confidence'])
            
            # Text-to-speech for confident predictions
            if (result['prediction'] != 'nothing' and 
                result['confidence'] >= 0.40 and 
                result['status'] in ['high_confidence', 'medium_confidence', 'low_confidence']):
                audio_data = speak_text(result['prediction'])
                if audio_data:
                    st.markdown(get_audio_player(audio_data), unsafe_allow_html=True)
            
            # Debug Mode: Alphabet Testing
            if st.session_state.debug_mode:
                st.markdown("---")
                test = st.session_state.alphabet_test
                expected = test['expected_letter']
                current_idx = test['current_index']
                
                # Display current target and controls
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    if current_idx < 26:
                        st.markdown(f"### 🎯 **Sign Letter: {expected}** ({current_idx + 1}/26)")
                    else:
                        st.markdown("### 🎉 **Alphabet Test Complete!**")
                with col2:
                    if st.button("⭐ Skip") and current_idx < 25:
                        test['current_index'] += 1
                        test['expected_letter'] = test['alphabet'][test['current_index']]
                        st.rerun()
                with col3:
                    if st.button("🔄 Reset Test"):
                        st.session_state.alphabet_test = {
                            'expected_letter': 'A',
                            'current_index': 0,
                            'results': {},
                            'alphabet': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                        }
                        if 'processed_results' in st.session_state:
                            del st.session_state.processed_results
                        st.rerun()
                
                # Process and record results for alphabet test
                if current_idx < 26:
                    if result['prediction'] != 'nothing' or result['status'] == 'no_hand_detected':
                        result_key = f"{expected}_{current_idx}"
                        if result_key not in st.session_state.get('processed_results', set()):
                            if 'processed_results' not in st.session_state:
                                st.session_state.processed_results = set()
                            
                            test['results'][expected] = {
                                'predicted': result['prediction'],
                                'confidence': result['confidence'],
                                'status': result['status'],
                                'correct': result['prediction'].upper() == expected.upper()
                            }
                            
                            st.session_state.processed_results.add(result_key)
                            
                            if result['prediction'].upper() == expected.upper():
                                st.success(f"✅ Correct: {expected}")
                            elif result['status'] == 'no_hand_detected':
                                st.warning(f"⚠️ No hand detected for {expected}")
                            else:
                                st.error(f"❌ Got {result['prediction']} for {expected}")
                            
                            if current_idx < 25:
                                test['current_index'] += 1
                                test['expected_letter'] = test['alphabet'][test['current_index']]
                
                # Display progress
                st.markdown("**📊 Alphabet Progress:**")
                alphabet_display = []
                for i, letter in enumerate(test['alphabet']):
                    if letter in test['results']:
                        if test['results'][letter]['correct']:
                            alphabet_display.append(f"🟢**{letter}**")
                        else:
                            alphabet_display.append(f"🔴**{letter}**")
                    elif i == current_idx:
                        alphabet_display.append(f"🟡**{letter}**")
                    else:
                        alphabet_display.append(f"⚪{letter}")
                
                row1 = alphabet_display[:13]
                row2 = alphabet_display[13:]
                st.markdown(" ".join(row1))
                st.markdown(" ".join(row2))
                
                # Show detailed results
                if test['results']:
                    with st.expander("📋 Detailed Results"):
                        for letter in test['alphabet']:
                            if letter in test['results']:
                                r = test['results'][letter]
                                status_icon = "✅" if r['correct'] else "❌"
                                st.markdown(f"{status_icon} **{letter}**: {r['predicted']} ({r['confidence']:.1%}, {r['status']})")
                
                # Summary stats
                if test['results']:
                    total_tested = len(test['results'])
                    correct = sum(1 for r in test['results'].values() if r['correct'])
                    accuracy = (correct / total_tested) * 100 if total_tested > 0 else 0
                    st.markdown(f"**Accuracy: {accuracy:.1f}% ({correct}/{total_tested})**")
            
            else:
                # Normal Mode: Sequence building
                MAX_SEQUENCE = 15
                DUPLICATE_ALLOWED = {'L', 'S', 'T', 'E', 'F', 'O', 'R', 'M', 'N', 'P'}
                
                if result['status'] in ['high_confidence', 'medium_confidence', 'low_confidence'] and result['prediction'] != 'nothing':
                    current_letter = result['prediction'].upper()
                    should_add = True
                    
                    if ('last_prediction' in st.session_state and 
                        st.session_state.last_prediction == current_letter and 
                        current_letter not in DUPLICATE_ALLOWED):
                        should_add = False
                        st.info(f"Duplicate prevented: {current_letter}")
                    
                    if should_add:
                        st.session_state.sequence.append(current_letter)
                        if len(st.session_state.sequence) > MAX_SEQUENCE:
                            st.session_state.sequence = st.session_state.sequence[-MAX_SEQUENCE:]
                        st.session_state.last_prediction = current_letter
                        st.success(f"Added: **{current_letter}**")
                
                # Display sequence
                if st.session_state.sequence:
                    sequence_str = "".join(st.session_state.sequence)
                    st.markdown("---")
                    st.markdown(f"### Current Sequence: **{sequence_str}**")
                    spaced_sequence = " ".join(st.session_state.sequence)
                    st.markdown(f"*({spaced_sequence})*")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Clear"):
                            st.session_state.sequence = []
                            if 'last_prediction' in st.session_state:
                                del st.session_state.last_prediction
                            if 'last_word' in st.session_state:
                                del st.session_state.last_word
                            st.rerun()
                    
                    with col2:
                        if st.button("Remove Last"):
                            if st.session_state.sequence:
                                st.session_state.sequence.pop()
                                if 'last_prediction' in st.session_state:
                                    del st.session_state.last_prediction
                            st.rerun()
                    
                    # Word detection
                    if len(st.session_state.sequence) >= 3:
                        nltk_words = get_nltk_words()
                        current = ''.join(st.session_state.sequence).upper()
                        longest_word = ""
                        
                        check_length = min(10, len(current))
                        for j in range(check_length, 2, -1):
                            word = current[-j:]
                            if word.lower() in nltk_words and len(word) > len(longest_word):
                                longest_word = word
                                break
                        
                        if longest_word and len(longest_word) >= 3:
                            st.success(f"Word Detected: **{longest_word}**")
                            if 'last_word' not in st.session_state or st.session_state.last_word != longest_word:
                                audio_data = speak_text(longest_word)
                                if audio_data:
                                    st.markdown(get_audio_player(audio_data), unsafe_allow_html=True)
                                    st.session_state.last_word = longest_word

            # Cleanup
            del image, image_rgb, results
            gc.collect()
            
        except Exception as e:
            st.error(f"Processing error: {e}")
            gc.collect()

if __name__ == "__main__":
    main()