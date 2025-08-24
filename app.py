def main():
    st.title("🤟 ASL Letter Detector")
    st.markdown("**Capture photos to detect ASL letters**")

    # Initialize session state FIRST - before any UI elements
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
    if 'debug_mode' not in st.session_state:
        st.session_state.debug_mode = False
    if 'alphabet_test' not in st.session_state:
        st.session_state.alphabet_test = {
            'expected_letter': 'A',
            'current_index': 0,
            'results': {},
            'alphabet': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        }

    # Debug mode toggle - AFTER state initialization
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("")
    with col2:
        # Use the session state value directly, don't reference it in value parameter initially
        debug_toggle = st.checkbox("🔬 Alphabet Test Mode", key="debug_checkbox")
        
        # Update session state when checkbox changes
        if debug_toggle != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_toggle
            if debug_toggle:
                # Reset alphabet test when entering debug mode
                st.session_state.alphabet_test = {
                    'expected_letter': 'A',
                    'current_index': 0,
                    'results': {},
                    'alphabet': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                }
                st.success("🔬 Debug mode activated!")
            else:
                st.info("📸 Normal mode activated!")
            st.rerun()

    # Debug Mode: Show alphabet test interface
    if st.session_state.debug_mode:
        st.markdown("### 🔬 Alphabet Test Mode")
        test = st.session_state.alphabet_test
        expected = test['expected_letter']
        current_idx = test['current_index']
        
        # Display current target with better visibility
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"### 🎯 **Sign Letter: {expected}** ({current_idx + 1}/26)")
            st.markdown(f"*Show the ASL sign for the letter **{expected}***")
        with col2:
            if st.button("⭐ Skip", help="Skip this letter"):
                if current_idx < 25:
                    test['current_index'] += 1
                    test['expected_letter'] = test['alphabet'][test['current_index']]
                    st.rerun()
        with col3:
            if st.button("🔄 Reset Test", help="Start over from A"):
                st.session_state.alphabet_test = {
                    'expected_letter': 'A',
                    'current_index': 0,
                    'results': {},
                    'alphabet': list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                }
                st.rerun()
        
        # Display alphabet progress with color coding
        st.markdown("**📊 Alphabet Progress:**")
        alphabet_display = []
        for i, letter in enumerate(test['alphabet']):
            if letter in test['results']:
                if test['results'][letter]['correct']:
                    alphabet_display.append(f"🟢**{letter}**")
                else:
                    alphabet_display.append(f"🔴**{letter}**")
            elif i == current_idx:
                alphabet_display.append(f"🟡**{letter}**")  # Current
            else:
                alphabet_display.append(f"⚪{letter}")  # Not tested yet
        
        # Display in rows of 13 letters each for better formatting
        row1 = alphabet_display[:13]
        row2 = alphabet_display[13:]
        st.markdown(" ".join(row1))
        st.markdown(" ".join(row2))
        
        # Show summary stats prominently
        if test['results']:
            total_tested = len(test['results'])
            correct = sum(1 for r in test['results'].values() if r['correct'])
            accuracy = (correct / total_tested) * 100 if total_tested > 0 else 0
            st.markdown(f"### **Accuracy: {accuracy:.1f}% ({correct}/{total_tested})**")
            
            # Show detailed results in expandable section
            with st.expander("📋 Detailed Results"):
                for letter in test['alphabet']:
                    if letter in test['results']:
                        r = test['results'][letter]
                        status_icon = "✅" if r['correct'] else "❌"
                        st.markdown(f"{status_icon} **{letter}**: {r['predicted']} ({r['confidence']:.1%}, {r['status']})")
        
        st.markdown("---")

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
                # Debug mode: Show expected letter and pass to display function
                if st.session_state.debug_mode:
                    expected = st.session_state.alphabet_test['expected_letter']
                    display_prediction_results(result, debug_mode=True, expected_letter=expected)
                else:
                    display_prediction_results(result)
                
                # Show confidence meter
                if result['prediction'] != 'nothing':
                    st.progress(result['confidence'])
            
            # Text-to-speech for confident predictions (lowered threshold to 40%)
            if (result['prediction'] != 'nothing' and 
                result['confidence'] >= 0.40 and  # Speak at 40%+ confidence
                result['status'] in ['high_confidence', 'medium_confidence', 'low_confidence']):
                audio_data = speak_text(result['prediction'])
                if audio_data:
                    st.markdown(get_audio_player(audio_data), unsafe_allow_html=True)
            
            # Debug Mode: Update alphabet test results when photo is taken
            if st.session_state.debug_mode:
                test = st.session_state.alphabet_test
                expected = test['expected_letter']
                current_idx = test['current_index']
                
                # Update test results ONLY if we have a real prediction
                if result['prediction'] != 'nothing' or result['status'] == 'no_hand_detected':
                    # Always record results for duplicate-allowed letters like O
                    can_duplicate = expected in {'L', 'S', 'T', 'E', 'F', 'O', 'R', 'M', 'N', 'P'}
                    
                    if expected not in test['results'] or can_duplicate:
                        test['results'][expected] = {
                            'predicted': result['prediction'],
                            'confidence': result['confidence'],
                            'status': result['status'],
                            'correct': result['prediction'].upper() == expected.upper()
                        }
                        
                        # Show immediate feedback
                        if result['prediction'].upper() == expected.upper():
                            st.success(f"✅ Correct! Moving to next letter...")
                        else:
                            st.error(f"❌ Expected {expected}, got {result['prediction']}")
                        
                        # Auto-advance to next letter after a short delay
                        if current_idx < 25:  # 0-25 for A-Z
                            test['current_index'] += 1
                            test['expected_letter'] = test['alphabet'][test['current_index']]
                            st.balloons() if result['prediction'].upper() == expected.upper() else None
                            st.rerun()
                        else:
                            st.success("🎉 Alphabet test complete!")
            
            else:
                # Normal Mode: Automatic sequence addition
                MAX_SEQUENCE = 15
                
                # Letters that are commonly duplicated in words - allow immediate duplicates
                DUPLICATE_ALLOWED = {'L', 'S', 'T', 'E', 'F', 'O', 'R', 'M', 'N', 'P'}
                
                if result['status'] in ['high_confidence', 'medium_confidence', 'low_confidence'] and result['prediction'] != 'nothing':
                    current_letter = result['prediction'].upper()
                    should_add = True
                    
                    # Check if we should prevent duplicate (skip this check if in debug mode)
                    if (not st.session_state.debug_mode and 
                        'last_prediction' in st.session_state and 
                        st.session_state.last_prediction == current_letter and 
                        current_letter not in DUPLICATE_ALLOWED):
                        should_add = False
                        st.info(f"Duplicate prevented: {current_letter}")
                    
                    if should_add:
                        st.session_state.sequence.append(current_letter)
                        if len(st.session_state.sequence) > MAX_SEQUENCE:
                            st.session_state.sequence = st.session_state.sequence[-MAX_SEQUENCE:]
                        st.session_state.last_prediction = current_letter
                        
                        # Print the letter that was added
                        st.success(f"✅ Added: **{current_letter}**")
                        
                # Always display current sequence prominently
                if st.session_state.sequence:
                    sequence_str = "".join(st.session_state.sequence)  # No spaces between letters
                    st.markdown("---")
                    st.markdown(f"### 📝 Current Sequence: **{sequence_str}**")
                    
                    # Also show with spaces for readability
                    spaced_sequence = " ".join(st.session_state.sequence)
                    st.markdown(f"*({spaced_sequence})*")
            
            # Display sequence (only in normal mode)
            if st.session_state.sequence and not st.session_state.debug_mode:
                st.markdown("---")
                sequence_str = " ".join(st.session_state.sequence)
                st.markdown(f"**📝 Sequence:** {sequence_str}")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🗑️ Clear"):
                        st.session_state.sequence = []
                        if 'last_prediction' in st.session_state:
                            del st.session_state.last_prediction
                        if 'last_word' in st.session_state:
                            del st.session_state.last_word
                        if 'duplicate_count' in st.session_state:
                            del st.session_state.duplicate_count
                        st.rerun()
                
                with col2:
                    if st.button("⬅️ Remove Last"):
                        if st.session_state.sequence:
                            st.session_state.sequence.pop()
                            # Reset duplicate tracking
                            if 'last_prediction' in st.session_state:
                                del st.session_state.last_prediction
                            if 'duplicate_count' in st.session_state:
                                del st.session_state.duplicate_count
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