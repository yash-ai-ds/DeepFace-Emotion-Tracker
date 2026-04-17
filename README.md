# Real-Time AI Mood Detector 🎭
An intelligent, real-time emotion recognition system built with Python, OpenCV, and Keras-based deep-learning models (`DeepFace`).

It features both a **Desktop Application** with text-to-speech feedback and keyboard controls, and a **Premium Web System** powered by Flask for broadcasting to a modern UI.

![AI Mood Detector UI Concept](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?style=for-the-badge&logo=opencv)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?style=for-the-badge&logo=flask)

## Features Included
1. **Real-Time Detection:** Live webcam streaming via `OpenCV`.
2. **Deep Learning Pipeline:** Employs pre-trained CNN (via `DeepFace` Keras wrapper).
3. **Optimized Threading:** AI models run in a background daemon thread mapping to the primary stream for max frames per second (FPS).
4. **Smart Actions & Visual Overlays:** Displays bounding boxes, confidence values, and suggests rule-based actions entirely local.
5. **Text-To-Speech API (Desktop):** Verbally speaks out recognized emotions asynchronously at distinct intervals.
6. **Save Context (Desktop):** Hit `s` during runtime to freeze the frame into your screenshots folder.
7. **Premium Web Interface:** Includes a fully-designed, glassmorphic webpage to cast the webcam. 

## Installation

Ensure your webcam is not being accessed by another application (like Zoom or Teams) before proceeding. No Database is utilized, everything is entirely local.

1. **Clone & Setup:**
```bash
# Recommended: Create a Python virtual environment
python -m venv venv
.\venv\Scripts\activate   # For Windows Power Shell
```

2. **Install the dependencies:**
```bash
pip install -r requirements.txt
```
*(Note: DeepFace will download the `facial_expression_model_weights.h5` internally on the very first run, taking a few seconds).*

## Project Structure
- `desktop_app.py` - Standalone OpenCV execution showing results in an OS window (Supports keys `'s'`, `'q'`, and `Text-to-Speech`).
- `app.py` - Flask web server script wrapping the OpenCV pipeline.
- `requirements.txt` - Project dependencies list.
- `templates/index.html` - Premium UI frontend designed via HTML/Vanilla JS/CSS.

## Usage Guide
### Mode 1: Local Desktop Client
Run the classic Python Desktop App for all local hotkeys.
```bash
python desktop_app.py
```
- Wait till the camera LED turns on.
- Press **`s`** to save a screenshot of the frame into the logic directory.
- Press **`q`** to safely terminate the stream and close.

### Mode 2: Web Server Dashboard
Start up the web experience to see the dynamic web app interface.
```bash
python app.py
```
Then navigate to `http://localhost:5000` via your favorite browser.

## Built With Quality in Mind 🚀
This system does NOT suffer from blocked stream freezing because it intentionally splits local geometry discovery (Haar Cascades for face tracking) vs heavy CNN inference (Emotion detection) into asynchronous threading logic! Enjoy smooth playback while identifying mood states dynamically.
