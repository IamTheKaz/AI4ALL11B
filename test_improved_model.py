# -*- coding: utf-8 -*-
"""
Test script for the improved ASL recognition model
"""

import tensorflow as tf
import numpy as np
import os
from improved_asl_model import predict_asl_letter

# Configuration
IMG_HEIGHT, IMG_WIDTH = 128, 128
BASE_DIR = '/content/data/'
MODEL_PATH = os.path.join(BASE_DIR, 'improved_asl_model.h5')

def load_model_and_predict(image_path):
    """Load the improved model and make predictions"""
    try:
        # Load the trained model
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        
        # Class names
        class_names = [chr(i) for i in range(65, 91)] + ['space', 'del', 'nothing']
        
        # Make prediction
        predicted_letter, confidence, top_3 = predict_asl_letter(image_path, model, class_names)
        
        return predicted_letter, confidence, top_3
        
    except Exception as e:
        print(f"Error loading model or making prediction: {e}")
        return None, None, None

# Example usage
if __name__ == "__main__":
    # Example: predict from a test image
    test_image_path = "/content/data/asl_alphabet_test/A_test.jpg"
    
    if os.path.exists(test_image_path):
        print("Testing with sample image...")
        predicted, conf, top_3 = load_model_and_predict(test_image_path)
        
        if predicted:
            print(f"\nFinal Result: {predicted} with {conf:.2%} confidence")
    else:
        print("Test image not found. Please provide a valid image path.")
        print("Usage: load_model_and_predict('path/to/your/image.jpg')") 