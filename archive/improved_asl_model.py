print('=== improved_asl_model.py is running! ===')
# -*- coding: utf-8 -*-
"""
Improved ASL Recognition Model
Enhanced version with better architecture, data augmentation, and training practices
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import gc
from datetime import datetime

print("Starting ASL model script...")

# CPU Optimizations for local training
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
os.environ['OMP_NUM_THREADS'] = '8'  # Use all CPU cores
os.environ['MKL_NUM_THREADS'] = '8'

# Disable GPU usage for CPU-only training
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print("CPU-optimized TensorFlow configuration loaded!")
print(f"TensorFlow version: {tf.__version__}")
print(f"Available CPU cores: {os.cpu_count()}")

# Configuration - Optimized for local CPU training
IMG_HEIGHT, IMG_WIDTH = 200, 200  # Match actual training data size
BATCH_SIZE = 32  # Reduced batch size for larger images
EPOCHS = 50  # More epochs for better performance
NUM_CLASSES = 29

# Use relative archive directory (universal path)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(SCRIPT_DIR, 'archive')
TRAIN_DATA_DIR = os.path.join(ARCHIVE_DIR, 'asl_alphabet_train', 'asl_alphabet_train')
TEST_DATA_DIR = os.path.join(ARCHIVE_DIR, 'asl_alphabet_test', 'asl_alphabet_test')

print(f"Archive directory: {ARCHIVE_DIR}")
print(f"Train data directory: {TRAIN_DATA_DIR}")
print(f"Test data directory: {TEST_DATA_DIR}")

# Create output directories
os.makedirs('./models/', exist_ok=True)
os.makedirs('./logs/', exist_ok=True)

print(f"Configuration:")
print(f"- Image size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"- Batch size: {BATCH_SIZE}")
print(f"- Epochs: {EPOCHS}")
print(f"- Color mode: GRAYSCALE (reduced complexity)")
print(f"- Archive directory: {ARCHIVE_DIR}")
print(f"- Training directory: {TRAIN_DATA_DIR}")
print(f"- Test directory: {TEST_DATA_DIR}")

# Check if directories exist
if not os.path.exists(TRAIN_DATA_DIR):
    print(f"⚠ Warning: Training directory not found at {TRAIN_DATA_DIR}")
    print("Please ensure the ASL dataset is in the archive folder")
else:
    print(f"✓ Training directory found: {TRAIN_DATA_DIR}")

if not os.path.exists(TEST_DATA_DIR):
    print(f"⚠ Warning: Test directory not found at {TEST_DATA_DIR}")
else:
    print(f"✓ Test directory found: {TEST_DATA_DIR}")

print("Starting data augmentation setup...")

# Data Augmentation for Training - Enhanced for real-world performance
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,  # More rotation for better generalization
    width_shift_range=0.4,  # More shift
    height_shift_range=0.4,  # More shift
    shear_range=0.4,  # More shear
    zoom_range=0.4,  # More zoom
    horizontal_flip=False,  # Don't flip ASL signs horizontally
    fill_mode='nearest',
    validation_split=0.2,  # 20% for validation
    brightness_range=[0.5, 1.5],  # More brightness variation
    channel_shift_range=100.0  # More color variation
)

# Data Generator for Training - Now using GRAYSCALE
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',  # Changed to grayscale
    subset='training'
)

# Data Generator for Validation - Now using GRAYSCALE
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    color_mode='grayscale',  # Changed to grayscale
    subset='validation'
)

# Improved Model Architecture - Enhanced with L2 Regularization
def create_improved_model():
    # L2 regularization factor
    l2_reg = 0.01  # Moderate regularization
    
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                     input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),  # Changed to 1 channel for grayscale
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Increased dropout
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Increased dropout
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Increased dropout
        
        # Fourth Convolutional Block (added for better feature extraction)
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),  # Higher dropout for deeper layers
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

# Create and compile model
model = create_improved_model()

# Compile with better optimizer settings
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks for better training
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,  # Reduced from 15
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,  # Reduced from 10
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        './models/best_asl_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train the model with progress tracking
print("Training the improved model...")
start_time = time.time()

try:
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        workers=4,  # Optimize for CPU
        use_multiprocessing=False  # Disable for stability
    )
    
    training_time = time.time() - start_time
    hours = training_time // 3600
    minutes = (training_time % 3600) // 60
    
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Training time: {int(hours)}h {int(minutes)}m")
    print(f"Epochs completed: {len(history.history['accuracy'])}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
    
except Exception as e:
    print(f"Training error: {e}")
    print("Trying with reduced batch size...")
    
    # Retry with smaller batch size
    BATCH_SIZE = 16
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='validation'
    )
    
    model = create_improved_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        workers=4,
        use_multiprocessing=False
    )
    
    training_time = time.time() - start_time
    hours = training_time // 3600
    minutes = (training_time % 3600) // 60
    
    print(f"\n=== TRAINING COMPLETED ===")
    print(f"Training time: {int(hours)}h {int(minutes)}m")
    print(f"Epochs completed: {len(history.history['accuracy'])}")
    print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")

# Save the final model
model.save('./models/improved_asl_model.h5')

# Plot training history
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)

# Load and prepare test data
def load_test_data():
    test_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('_test.jpg')]
    images = []
    labels = []
    class_names = [chr(i) for i in range(65, 91)] + ['space', 'del', 'nothing']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    for f in test_files:
        try:
            img = tf.keras.preprocessing.image.load_img(
                os.path.join(TEST_DATA_DIR, f), 
                target_size=(IMG_HEIGHT, IMG_WIDTH), 
                color_mode='grayscale'  # Changed to grayscale
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img_array)
            label = f.split('_')[0]
            labels.append(class_to_idx[label])
        except Exception as e:
            print(f"Failed to load {f}: {e}")
    
    return np.array(images), np.array(labels), class_names

# Evaluate on test data
test_data, test_labels, class_names = load_test_data()

if len(test_data) > 0:
    # Convert labels to one-hot encoding
    test_labels_one_hot = tf.keras.utils.to_categorical(test_labels, NUM_CLASSES)
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_data, test_labels_one_hot, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Get predictions
    predictions = model.predict(test_data)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predicted_classes, target_names=class_names))
    
    # Plot confusion matrix
    cm = confusion_matrix(test_labels, predicted_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

# Prediction function for new images
def predict_asl_letter(image_path, model, class_names):
    """Predict ASL letter from image file"""
    try:
        img = tf.keras.preprocessing.image.load_img(
            image_path, 
            target_size=(IMG_HEIGHT, IMG_WIDTH), 
            color_mode='grayscale'  # Changed to grayscale
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        predictions = model.predict(img_array, verbose=0)
        class_idx = np.argmax(predictions[0])
        predicted_letter = class_names[class_idx]
        confidence = np.max(predictions[0])
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [(class_names[idx], predictions[0][idx]) for idx in top_3_idx]
        
        print(f"Predicted ASL Letter: {predicted_letter} (Confidence: {confidence:.4f})")
        print("Top 3 predictions:")
        for i, (letter, conf) in enumerate(top_3_predictions, 1):
            print(f"  {i}. {letter}: {conf:.4f}")
        
        return predicted_letter, confidence, top_3_predictions
        
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None, None

print("\nModel training and evaluation complete!")
print("Use predict_asl_letter() function to predict new images.")

print("\n=== IMPROVEMENTS IMPLEMENTED ===")
print("1. ✅ Grayscale conversion: Reduced complexity from 3 channels to 1")
print("2. ✅ L2 Regularization: Added to all Conv2D and Dense layers (l2_reg=0.01)")
print("3. ✅ Enhanced Dropout: Increased rates (0.25→0.3, 0.25→0.4 for deeper layers)")
print("4. ✅ Batch Normalization: Already present throughout the model")
print("5. ✅ Additional Conv Block: Added 4th block with 256 filters for better feature extraction")
print("\nThese changes should significantly reduce overfitting and improve generalization!") 