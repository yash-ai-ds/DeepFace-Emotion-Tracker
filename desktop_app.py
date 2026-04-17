import cv2
import os
import time
import threading
from deepface import DeepFace
import pyttsx3
from collections import Counter

# Initialize Text-to-Speech Engine
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150) # Set speaking rate
except Exception:
    engine = None

# Global variables for background prediction
current_emotion = "Analyzing..."
current_confidence = 0.0
is_predicting = False
emotion_history = []
last_spoken = time.time()

# Rule-based suggestions
actions_map = {
    'happy': 'Suggestion: Play energetic music! \u266B',
    'sad': 'Suggestion: Listen to relaxing music. \u266A',
    'angry': 'Suggestion: Try a 5-min breathing exercise. \u262E',
    'surprise': 'Suggestion: Document what surprised you! \u2728',
    'neutral': 'Suggestion: Keep up the steady work! \u2714',
    'fear': 'Suggestion: Get some fresh air and relax. \u2615',
    'disgust': 'Suggestion: Maybe look at something pleasant! \u26F0'
}

def speak_emotion(emotion):
    """Speaks the detected emotion using pyttsx3."""
    if engine is None: return
    try:
        engine.say(f"You represent {emotion}")
        engine.runAndWait()
    except Exception:
        pass

def analyze_emotion(face_image):
    """Runs DeepFace inference in the background to ensure main thread stays fast."""
    global current_emotion, current_confidence, is_predicting, last_spoken
    try:
        # We set enforce_detection to False because the face is already cropped
        result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, silent=True)
        
        # DeepFace might return a list if it handles multiple inputs
        if isinstance(result, list):
            result = result[0]
            
        raw_emotion = result['dominant_emotion']
        
        # Temporal smoothing to prevent fluttering
        emotion_history.append(raw_emotion)
        if len(emotion_history) > 7:
            emotion_history.pop(0)
        
        # Select the most frequently occurring emotion in the history
        stable_emotion = Counter(emotion_history).most_common(1)[0][0]
            
        current_emotion = stable_emotion
        current_confidence = result['emotion'][stable_emotion]
        
        # Optional: Announce emotion every 15 seconds
        if time.time() - last_spoken > 15:
            threading.Thread(target=speak_emotion, args=(current_emotion,), daemon=True).start()
            last_spoken = time.time()
            
    except Exception as e:
        print("Model Inference Error:", e)
    finally:
        is_predicting = False

def main():
    global is_predicting
    
    # Load OpenCV's pre-trained Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return
        
    print("\n--- AI Mood Detector Started ---")
    print("Press 's' to Save a Screenshot")
    print("Press 'q' to Quit\n")
    
    # Create directory for screenshots if it doesn't exist
    os.makedirs('screenshots', exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1) # Mirror effect for natural feeling
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            # Draw bounding box around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Predict emotion in background on cropped face to maintain high FPS
            if not is_predicting:
                is_predicting = True
                
                # Add safe padding to the crop
                padding = 20
                y1 = max(0, y - padding)
                y2 = min(frame.shape[0], y + h + padding)
                x1 = max(0, x - padding)
                x2 = min(frame.shape[1], x + w + padding)
                face_crop = frame[y1:y2, x1:x2]
                
                # Start analysis thread
                threading.Thread(target=analyze_emotion, args=(face_crop.copy(),), daemon=True).start()
                
            # Render Emotion Label and Confidence
            label = f"{current_emotion.capitalize()} ({current_confidence:.1f}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Render Rule-Based Action
            suggestion = actions_map.get(current_emotion.lower(), "Analyzing...")
            cv2.putText(frame, suggestion, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Live AI Mood Detector', frame)
        
        # Handle Keyboard Inputs
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            filepath = f"screenshots/screenshot_{int(time.time())}.png"
            cv2.imwrite(filepath, frame)
            print(f"Screenshot saved -> {filepath}")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
