from flask import Flask, render_template, Response
import cv2
import threading
from deepface import DeepFace
from collections import Counter

app = Flask(__name__)

# Load OpenCV's Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Global variables to pass state between detection threads and main thread
current_emotion = "Analyzing..."
current_confidence = 0.0
is_predicting = False
emotion_history = []

actions_map = {
    'happy': 'Suggestion: Play energetic music! \u266B',
    'sad': 'Suggestion: Listen to relaxing music. \u266A',
    'angry': 'Suggestion: Take deep breaths. \u262E',
    'surprise': 'Suggestion: What a surprise! Enjoy the moment! \u2728',
    'neutral': 'Suggestion: Keep up the productive work! \u2714',
    'fear': 'Suggestion: Drink some water and relax. \u2615',
    'disgust': 'Suggestion: Take a quick walk outside. \u26F0'
}

def analyze_emotion(face_image):
    """DeepFace inference running in the background to ensure steady web stream FPS."""
    global current_emotion, current_confidence, is_predicting
    try:
        result = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False, silent=True)
        if isinstance(result, list): 
            result = result[0]
            
        raw_emotion = result['dominant_emotion']
        
        # Temporal smoothing
        emotion_history.append(raw_emotion)
        if len(emotion_history) > 7:
            emotion_history.pop(0)
            
        stable_emotion = Counter(emotion_history).most_common(1)[0][0]
        
        current_emotion = stable_emotion
        current_confidence = result['emotion'][stable_emotion]
        
    except Exception as e:
        print("Backend Inference Error:", e)
    finally:
        is_predicting = False

def generate_frames():
    """Generator function that continuously reads frames, processes them, and yields JPEG bytes."""
    global is_predicting
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face Detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        
        for (x, y, w, h) in faces:
            # Bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
            
            if not is_predicting:
                is_predicting = True
                
                # Dynamic Cropping with Padding
                padding = 20
                face_crop = frame[max(0, y-padding):min(frame.shape[0], y+h+padding), max(0, x-padding):min(frame.shape[1], x+w+padding)]
                
                threading.Thread(target=analyze_emotion, args=(face_crop.copy(),), daemon=True).start()
                
            # Text Overlays
            label = f"{current_emotion.capitalize()} ({current_confidence:.1f}%)"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            suggestion = actions_map.get(current_emotion.lower(), "Analyzing...")
            cv2.putText(frame, suggestion, (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Encode frame as JPEG to push to web view
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("\n\u2728 Web Server Started. Open http://127.0.0.1:5005 in your browser \u2728\n")
    app.run(host='0.0.0.0', port=5005, debug=False)
