import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from gtts import gTTS
import pygame
import tempfile
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Paths
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_asl_model.h5')
TRAIN_DIR = 'train_images'
TEST_DIR = 'test_images'
LOG_DIR = 'logs'

IMG_HEIGHT, IMG_WIDTH = 32, 32
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 29
CLASS_NAMES = [chr(i) for i in range(65, 91)] + ['space', 'del']

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# TTS function using pygame
def speak_text(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
        tts = gTTS(text)
        tts.save(fp.name)
        temp_name = fp.name
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(temp_name)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    finally:
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        os.remove(temp_name)

# Custom Progress Callback for Detailed Training Stats
class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.start_time = None
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"🚀 EPOCH {epoch + 1}/{self.params['epochs']}")
        print(f"{'='*60}")
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.start_time
        self.epoch_times.append(epoch_time)
        self.history['accuracy'].append(logs.get('accuracy', 0))
        self.history['val_accuracy'].append(logs.get('val_accuracy', 0))
        self.history['loss'].append(logs.get('loss', 0))
        self.history['val_loss'].append(logs.get('val_loss', 0))
        train_acc = logs.get('accuracy', 0)
        train_loss = logs.get('loss', 0)
        val_acc = logs.get('val_accuracy', 0)
        val_loss = logs.get('val_loss', 0)
        if epoch > 0:
            acc_improvement = train_acc - self.history['accuracy'][-2]
            val_acc_improvement = val_acc - self.history['val_accuracy'][-2]
        else:
            acc_improvement = 0
            val_acc_improvement = 0
        print(f"\n📊 EPOCH {epoch + 1} RESULTS:")
        print(f"⏱️  Time: {epoch_time:.1f}s")
        print(f"📈 Training - Acc: {train_acc:.4f} ({acc_improvement:+.4f}) | Loss: {train_loss:.4f}")
        print(f"🎯 Validation - Acc: {val_acc:.4f} ({val_acc_improvement:+.4f}) | Loss: {val_loss:.4f}")
        progress = (epoch + 1) / self.params['epochs'] * 100
        bar_length = 30
        filled_length = int(bar_length * (epoch + 1) / self.params['epochs'])
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f"\n📊 OVERALL PROGRESS:")
        print(f"[{bar}] {progress:.1f}% Complete")
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        remaining_epochs = self.params['epochs'] - (epoch + 1)
        eta = remaining_epochs * avg_epoch_time
        if eta > 0:
            eta_minutes = eta / 60
            if eta_minutes > 60:
                eta_hours = eta_minutes / 60
                print(f"⏰ Estimated time remaining: {eta_hours:.1f} hours")
            else:
                print(f"⏰ Estimated time remaining: {eta_minutes:.1f} minutes")
        print(f"{'='*60}")

def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.6),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

def train_model():
    print("Setting up data generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20.0
    )
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='grayscale',
        subset='validation'
    )
    print("Building model...")
    model = create_model()
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        TrainingProgressCallback(),
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=4, min_lr=1e-8, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    print("Training model...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Model trained and saved to {MODEL_PATH}")
    # Save training history plot
    plot_training_history(history)
    return model

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()

def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        print(f"Loading model from {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Model loaded.")
    else:
        print("No trained model found. Training a new model...")
        model = train_model()
    return model

def predict_image(image_path, model):
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions[0])
    letter = CLASS_NAMES[class_idx]
    return letter

def interactive_prediction(model):
    print("\n🚀 ASL PREDICTION STARTING...")
    print("📁 File dialog will open automatically")
    print("🎯 Select an image to get instant ASL prediction!")
    print("="*50)
    phrase = []
    hello_world = list('HELLO WORLD')
    hello_world_idx = 0
    phrase_detected = False
    root = tk.Tk()
    root.withdraw()
    while True:
        print("\nPlease select your ASL image files (multiple allowed, or press cancel to stop)...")
        file_paths = filedialog.askopenfilenames(title="Select ASL Images", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not file_paths:
            print("No files selected. Exiting loop.")
            break
        for image_path in file_paths:
            filename = os.path.basename(image_path)
            letter = predict_image(image_path, model)
            print(f"Image: {filename} -> Predicted: {letter}")
            if letter == 'space':
                speak_text('space')
            elif letter == 'del':
                speak_text('delete')
            else:
                speak_text(letter)
            if hello_world_idx < len(hello_world):
                expected = 'space' if hello_world[hello_world_idx] == ' ' else hello_world[hello_world_idx]
                if letter == expected:
                    if letter == 'space':
                        phrase.append(' ')
                    elif letter not in ['del']:
                        phrase.append(letter)
                    hello_world_idx += 1
            phrase_str = ''.join(phrase)
            if not phrase_detected and phrase_str.strip().upper() == 'HELLO WORLD':
                print("\nDemo Phrase detected: Hello World")
                speak_text('Hello World')
                phrase_detected = True

if __name__ == '__main__':
    model = load_or_train_model()
    interactive_prediction(model) 