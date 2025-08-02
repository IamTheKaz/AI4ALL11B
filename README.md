#🤟 ASL to Speech Translator

Project Overview
Developed an ASL to Speech Translator that converts American Sign Language (ASL) gestures into spoken letters in real-time, built using MediaPipe and a Multi-Layer Perceptron (MLP) model as part of the AI4ALL Ignite accelerator program. This project demonstrates skills in computer vision, neural network design, and web development to enhance accessibility for the deaf and hard-of-hearing community, with a future goal of integration as an add-on for Zoom calls.
Problem Statement
Individuals who use ASL often face communication barriers with those unfamiliar with sign language, particularly in virtual settings where human translators are resource-intensive. This project addresses this challenge by providing an automated, real-time translation tool with live, snapshot, and upload image modes, converting ASL gestures into spoken letters to foster inclusivity, with potential future applications in platforms like Zoom for seamless virtual communication.
Key Results

Implemented a MediaPipe-based pipeline to detect hand gestures from live video feeds, snapshots, or uploaded images, extracting keypoints for ASL letter recognition.
Trained an MLP model primarily on the Synthetic ASL Alphabet dataset, augmented with the ASL Alphabet dataset to mitigate bias and improve accuracy, though further training is needed.
Developed a Streamlit web app with three modes—live, snapshot, and upload image—allowing users to test ASL letter recognition and receive spoken output.
Integrated text-to-speech functionality to vocalize recognized letters, laying the groundwork for future Zoom integration to enable direct communication without translators.

Methodologies
Utilized MediaPipe for real-time hand detection and keypoint extraction from live video feeds, snapshots, or uploaded images. An MLP model was designed and trained using TensorFlow to classify ASL letters based on keypoint data extracted from static images. The model, trained primarily on synthetic images and augmented with real images to reduce bias, requires additional training for improved accuracy. The Streamlit framework, enhanced with streamlit-camera-input-live, powers the interactive web interface with three modes: live video, snapshot capture, and image upload. The gTTS library converts recognized letters into audible output. Image preprocessing and keypoint analysis were performed using OpenCV, NumPy, and Pillow, with NLTK and rich used for text processing and console output formatting, respectively. The h5py library supported model storage, and requests facilitated API interactions.
Data Sources

Synthetic ASL Alphabet Dataset (Primary): Kaggle Synthetic ASL Alphabet (used as the primary dataset for training the MLP model with synthetic images).
ASL Alphabet Dataset (Augmenting): Kaggle ASL Alphabet Dataset (used to augment training data to mitigate bias and improve model accuracy).
Custom ASL Image Dataset: Collected and annotated 1,000+ images of ASL letter gestures for model training.

Installation and Usage Instructions
To deploy and use the ASL to Speech Translator, follow these steps to clone the repository and connect it to a Streamlit account for hosting the web app.
Prerequisites

A GitHub account to clone the repository.
A Streamlit account (sign up at streamlit.io).
Git installed on your local machine.

Installation

Clone the Repository:Clone the project repository to your local machine using the following command: git clone https://github.com/IamTheKaz/AI4ALL11B.git


Set Up Streamlit:
Log in to your Streamlit account at streamlit.io.
Connect your GitHub account to Streamlit by navigating to the Streamlit dashboard and selecting "New app" > "From GitHub."
Select the cloned repository (AI4ALL11B) and specify the main Python file (e.g., app.py) that contains the Streamlit app code.
Streamlit will automatically detect the required dependencies (listed in requirements.txt) and deploy the app.


Deploy the App:
Once configured, Streamlit will build and host the app, providing a URL (e.g., https://asldemo.streamlit.app).
Ensure the repository includes a requirements.txt file with all necessary dependencies (Python, TensorFlow, MediaPipe, Streamlit, OpenCV, NumPy, gTTS, rich, NLTK, requests, streamlit-camera-input-live, Pillow, h5py).



Usage

Access the App: Open the deployed Streamlit app URL in a browser (e.g., https://asldemo.streamlit.app/).
Select a Mode:
Live Mode: Use your webcam to perform ASL gestures in real-time. The app detects hand keypoints and translates them into spoken letters.
Snapshot Mode: Capture a single frame of your ASL gesture using the app’s snapshot feature, which processes the image and outputs the spoken letter.
Upload Image Mode: Upload a static image of an ASL gesture, and the app will recognize and vocalize the corresponding letter.


Real-World Scenario: The app is designed for future integration with Zoom calls. Currently, users can run the Streamlit app in a browser to test ASL translation, performing gestures to produce spoken letters audible to others (e.g., via shared audio in a virtual meeting). For example, signing the letter "A" will trigger the app to say "A" aloud, facilitating communication for ASL users. In the future, deploying this as a Zoom add-on will enable seamless translation during virtual meetings or webinars.
Interpreting Results: The app outputs the recognized ASL letter as spoken audio via gTTS. If the model misclassifies a letter due to ongoing training needs, try adjusting lighting, hand positioning, or using snapshot/upload modes for clearer input. Share the app’s audio output in virtual settings to communicate recognized letters.

Example Commands

This app runs with buttons that are easy to interpret such as start and stop webcam and take photo. 

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


Synthetic ASL Alphabet Dataset (Primary): https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet
ASL Alphabet Dataset (Augmenting): https://www.kaggle.com/datasets/grassknoted/asl-alphabet


Authors
This project was completed in collaboration with:

Kassandra Ring: Kazwolff85@gmail.com
Gemmie
