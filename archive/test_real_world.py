import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Model settings
IMG_HEIGHT, IMG_WIDTH = 32, 32  # Match model size
NUM_CLASSES = 29
class_names = [chr(i) for i in range(65, 91)] + ['space', 'del', 'nothing']

def resize_and_preprocess_image(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Resize and preprocess image for model input"""
    try:
        # Load image with PIL for better resizing
        pil_img = Image.open(image_path)
        
        # Get original dimensions
        original_size = pil_img.size
        print(f"Original image size: {original_size[0]}x{original_size[1]}")
        
        # Resize image
        resized_img = pil_img.resize(target_size, Image.Resampling.LANCZOS)
        print(f"Resized to: {target_size[0]}x{target_size[1]}")
        
        # Convert to numpy array and normalize
        img_array = np.array(resized_img) / 255.0
        
        # Ensure 3 channels (RGB)
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]
        
        return img_array, original_size
        
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None, None

def load_and_predict(image_path, model):
    """Load image and get detailed prediction analysis"""
    try:
        # Resize and preprocess image
        img_array, original_size = resize_and_preprocess_image(image_path)
        
        if img_array is None:
            return None, None, None
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions
        predictions = model.predict(img_array, verbose=0)[0]
        
        # Get top predictions
        top_5_idx = np.argsort(predictions)[-5:][::-1]
        top_5_predictions = [(class_names[idx], predictions[idx]) for idx in top_5_idx]
        
        # Analysis
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions)
        
        print(f"\n=== PREDICTION ANALYSIS ===")
        print(f"Image: {os.path.basename(image_path)}")
        print(f"Original size: {original_size[0]}x{original_size[1]}")
        print(f"Model input size: {IMG_HEIGHT}x{IMG_WIDTH}")
        print(f"Predicted: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Entropy: {-np.sum(predictions * np.log(predictions + 1e-10)):.4f}")
        
        print(f"\nTop 5 predictions:")
        for i, (letter, conf) in enumerate(top_5_predictions, 1):
            print(f"  {i}. {letter}: {conf:.4f}")
        
        # Check if prediction is "confident" (high confidence) or "uncertain" (low confidence)
        if confidence > 0.8:
            print(f"\n✅ High confidence prediction")
        elif confidence > 0.5:
            print(f"\n⚠️  Medium confidence prediction")
        else:
            print(f"\n❌ Low confidence prediction - model is uncertain")
        
        return predicted_class, confidence, top_5_predictions
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None, None

def display_resized_image(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH)):
    """Display the original and resized image for verification"""
    try:
        # Load original image
        original_img = Image.open(image_path)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Display original
        ax1.imshow(original_img)
        ax1.set_title(f'Original ({original_img.size[0]}x{original_img.size[1]})')
        ax1.axis('off')
        
        # Display resized
        resized_img = original_img.resize(target_size, Image.Resampling.LANCZOS)
        ax2.imshow(resized_img)
        ax2.set_title(f'Resized ({target_size[0]}x{target_size[1]})')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error displaying image: {e}")

def test_multiple_images():
    """Test multiple images and analyze performance"""
    print("Loading model...")
    model = load_model('./models/improved_asl_model.h5')
    print("Model loaded!")
    
    # Open file dialog for multiple files
    Tk().withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select ASL images to test",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
    )
    
    if not file_paths:
        print("No files selected.")
        return
    
    print(f"\nTesting {len(file_paths)} images...")
    
    results = []
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n--- Image {i}/{len(file_paths)} ---")
        
        # Ask if user wants to see the resized image
        show_image = input(f"Show resized image for {os.path.basename(file_path)}? (y/n): ").lower().strip()
        if show_image == 'y':
            display_resized_image(file_path)
        
        predicted, confidence, top_5 = load_and_predict(file_path, model)
        if predicted:
            results.append({
                'file': os.path.basename(file_path),
                'predicted': predicted,
                'confidence': confidence,
                'top_5': top_5
            })
    
    # Summary
    if results:
        confidences = [r['confidence'] for r in results]
        avg_confidence = np.mean(confidences)
        print(f"\n=== SUMMARY ===")
        print(f"Average confidence: {avg_confidence:.4f}")
        print(f"High confidence predictions: {sum(1 for c in confidences if c > 0.8)}/{len(confidences)}")
        print(f"Low confidence predictions: {sum(1 for c in confidences if c < 0.5)}/{len(confidences)}")
        
        if avg_confidence < 0.6:
            print(f"\n⚠️  WARNING: Low average confidence suggests overfitting!")
            print(f"   The model may not generalize well to real-world images.")
        elif avg_confidence > 0.9:
            print(f"\n✅ High average confidence - but beware of overfitting!")

def quick_test_single_image():
    """Quick test for a single image without asking for display"""
    print("Loading model...")
    model = load_model('./models/improved_asl_model.h5')
    print("Model loaded!")
    
    # Open file dialog for single file
    Tk().withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an ASL image to test",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")]
    )
    
    if file_path:
        print(f"\nTesting: {os.path.basename(file_path)}")
        predicted, confidence, top_5 = load_and_predict(file_path, model)
        
        if predicted:
            print(f"\n✅ Prediction complete!")
        else:
            print(f"\n❌ Failed to process image")
    else:
        print("No file selected.")

if __name__ == "__main__":
    print("ASL Real-World Image Tester")
    print("1. Quick test (single image)")
    print("2. Full test (multiple images with analysis)")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    if choice == "1":
        quick_test_single_image()
    elif choice == "2":
        test_multiple_images()
    else:
        print("Invalid choice. Running quick test...")
        quick_test_single_image() 