# Install mediapipe
!pip install mediapipe==0.10.14 --no-cache-dir

import os
import shutil
import cv2
import mediapipe as mp
import tqdm
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# Dataset path in Kaggle
file_path = '/kaggle/input/synthetic-asl-alphabet/'  # Path to Synthetic ASL Alphabet dataset

IMG_HEIGHT, IMG_WIDTH = 32, 32
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 29

TRAIN_DATA_DIR = '/kaggle/input/synthetic-asl-alphabet/Train_Alphabet'
TEST_DATA_DIR = '/kaggle/input/synthetic-asl-alphabet/Test_Alphabet'

# Enable mixed precision and GPU memory growth
tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Enable mixed precision and GPU memory growth
tf.keras.mixed_precision.set_global_policy('mixed_float16')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

print("✅ Setup complete.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

# Parameters
IMG_SIZE = 224
MAX_SAMPLES_PER_CLASS = 900  

data = []
label = []

# Wrap outer loop with tqdm
from tqdm import tqdm

folders = os.listdir(TRAIN_DATA_DIR)

import sys
import random

REAL_DATA_DIR = "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
REAL_LABELS = [chr(i) for i in range(ord('A'), ord('Z') + 1)] + ["nothing"]

MAX_SAMPLES_PER_CLASS = 750
OVERDRAW_LIMIT = 3000

for label_name in tqdm(REAL_LABELS, desc="📥 Collecting real samples"):
    folder_path = os.path.join(REAL_DATA_DIR, label_name)

    # Special case for 'nothing' → skip detection, insert synthetic rows
    if label_name == "nothing":
        placeholder_row = [0.0] * (21 * 3)  # 21 landmarks × (x, y, z)
        for _ in range(MAX_SAMPLES_PER_CLASS):
            data.append(placeholder_row)
            label.append("blank")  # Normalize label
        tqdm.write(f"🧪 'nothing' → Injected {MAX_SAMPLES_PER_CLASS} synthetic 'blank' samples")
        continue

    if not os.path.exists(folder_path):
        tqdm.write(f"⚠️ Missing folder: {label_name}")
        continue

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)
    files = files[:OVERDRAW_LIMIT]

    valid_samples = []
    sample_count = 0

    sys.stdout.write(f"\r▶️ Processing '{label_name}'... 0 samples collected")
    sys.stdout.flush()

    for file in files:
        if sample_count >= OVERDRAW_LIMIT:
            break

        img_path = os.path.join(folder_path, file)
        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_row = [lm.x for lm in hand_landmarks.landmark] + \
                               [lm.y for lm in hand_landmarks.landmark] + \
                               [lm.z for lm in hand_landmarks.landmark]
                valid_samples.append((landmark_row, label_name))
                sample_count += 1

                sys.stdout.write(f"\r▶️ Processing '{label_name}'... {sample_count} valid samples")
                sys.stdout.flush()

        except Exception as e:
            tqdm.write(f"⚠️ Error in '{file}': {e}")
            continue

    valid_samples = valid_samples[:MAX_SAMPLES_PER_CLASS]
    for row, label_value in valid_samples:
        data.append(row)
        label.append(label_value)

    sys.stdout.write("\n")
    tqdm.write(f"✅ {label_name:<10} finalized with {len(valid_samples)} samples")
    
# Outer tqdm for folders
for labels in tqdm(folders, desc="📦 Overall class progress"):
    letter_folders = os.path.join(TRAIN_DATA_DIR, labels)
    sample_count = 0
    img_files = os.listdir(letter_folders)

    sys.stdout.write(f"\r▶️ Processing '{labels}'... 0 samples collected")
    sys.stdout.flush()

    for img_file in img_files:
        if sample_count >= MAX_SAMPLES_PER_CLASS:
            break

        img_path = os.path.join(letter_folders, img_file)

        try:
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmark_row = [lm.x for lm in hand_landmarks.landmark] + \
                               [lm.y for lm in hand_landmarks.landmark] + \
                               [lm.z for lm in hand_landmarks.landmark]
                data.append(landmark_row)
                label.append(labels)
                sample_count += 1

                # Update sample count inline
                sys.stdout.write(f"\r▶️ Processing '{labels}'... {sample_count} samples collected")
                sys.stdout.flush()

        except Exception as e:
            tqdm.write(f"⚠️ Error: {e}")
            continue
   

    sys.stdout.write("\n")  # Finish the line cleanly after folder is done
    # Update with sample count in the same line
    tqdm.write(f"✅ {labels:<10} collected {sample_count} samples")
    
# ✅ Save results to writable Kaggle directory
if data:
    df = pd.DataFrame(data)
    df["label"] = label
    output_path = "/kaggle/working/hand_landmarks.csv"
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} samples to {output_path}")
else:
    print("⚠️ No valid hand landmarks found. Nothing saved.")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("/kaggle/working/hand_landmarks.csv")
X = df.drop("label" , axis = 1).values
y = df["label"]

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_final = to_categorical(y_encoded, num_classes=28)

X_train, X_test, y_train, y_test = train_test_split(X,y_final, test_size = 0.3, random_state = 43)
print(X_train.shape)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define model save path in Kaggle working directory
model_save_dir = '/kaggle/working/models'
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, 'asl_model.h5')

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(28, activation='softmax')   # <-- updated here
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Best model checkpoint
checkpoint = ModelCheckpoint(
    model_save_path,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Stop training if val_accuracy doesn't improve after 5 consecutive epochs
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Fit the model with both callbacks
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[checkpoint, early_stop]
)

# Save the final model
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# Evaluate test performance
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.2f}")
import sys
import tensorflow as tf
import h5py
import os

# Define model path from your training code
model_save_dir = '/kaggle/working/data/models'
model_save_path = os.path.join(model_save_dir, 'asl_model.h5')

# Print environment versions
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {tf.keras.__version__}")
print(f"h5py version: {h5py.__version__}")

print(f"numpy version: {numpy.__version__}")

# Check model file existence and details
print(f"\nModel file exists: {os.path.exists(model_save_path)}")
if os.path.exists(model_save_path):
    print(f"Model file size: {os.path.getsize(model_save_path)} bytes")
    print(f"Model file permissions: {oct(os.stat(model_save_path).st_mode)[-3:]}")
    
    # Inspect model structure
    try:
        with h5py.File(model_save_path, 'r') as f:
            print("\nHDF5 file contents:")
            print(f"Keys: {list(f.keys())}")
            if 'model_config' in f:
                print("Model config found in HDF5 file")
            if 'model_weights' in f:
                print("Model weights found in HDF5 file")
        
        # Load and summarize model
        model = tf.keras.models.load_model(model_save_path)
        print("\nModel loaded successfully")
        model.summary()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
else:
    print("Model file not found at:", model_save_path)

