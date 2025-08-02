ASL to Speech Translator

Project Overview
Developed an ASL to Speech Translator as an add-on app for Zoom calls, enabling real-time translation of American Sign Language (ASL) gestures into spoken letters without the need for human translators. Built using MediaPipe and a Multi-Layer Perceptron (MLP) model as part of the AI4ALL Ignite accelerator program, this project showcases skills in computer vision, neural network design, and web development to enhance accessibility for the deaf and hard-of-hearing community.
Problem Statement
Individuals who use ASL often require human translators to communicate effectively during Zoom calls, which can be resource-intensive and limit spontaneity. This project addresses this challenge by providing an automated, real-time translation tool with live, snapshot, and upload image modes, converting ASL gestures into spoken letters to foster seamless and inclusive communication in virtual settings like meetings, webinars, and social interactions.
Key Results

Implemented a MediaPipe-based pipeline to detect hand gestures from live video feeds, snapshots, or uploaded images, extracting keypoints for ASL letter recognition.
Trained an MLP model primarily on the Synthetic ASL Alphabet dataset, augmented with the ASL Alphabet dataset to mitigate bias and improve accuracy, though further training is needed.
Developed a Streamlit web app with three modes—live, snapshot, and upload image—allowing users to test ASL letter recognition and receive spoken output.
Integrated text-to-speech functionality to vocalize recognized letters, enabling direct communication on Zoom without external translators.

Methodologies
Utilized MediaPipe for real-time hand detection and keypoint extraction from live Zoom video feeds, snapshots, or uploaded images. An MLP model was designed and trained using TensorFlow to classify ASL letters based on keypoint data extracted from static images. The model, trained primarily on synthetic images and augmented with real images to reduce bias, requires additional training for improved accuracy. The Streamlit framework, enhanced with streamlit-camera-input-live, powers the interactive web interface with three modes: live video, snapshot capture, and image upload. The gTTS library converts recognized letters into audible output. Image preprocessing and keypoint analysis were performed using OpenCV, NumPy, and Pillow, with NLTK and rich used for text processing and console output formatting, respectively. The h5py library supported model storage, and requests facilitated API interactions.
Data Sources

Synthetic ASL Alphabet Dataset (Primary): https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet
ASL Alphabet Dataset (Augmenting): https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Technologies Used

Python
TensorFlow
MediaPipe
Streamlit
OpenCV
NumPy
gTTS
rich
NLTK
requests
streamlit-camera-input-live
Pillow
h5py
GitHub Pages (for project documentation)

Authors
This project was completed in collaboration with:

Kassandra Ring: Kazwolff85@gmail.com
Gemmie
