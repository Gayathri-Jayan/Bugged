from flask import Flask, render_template, Response, jsonify, redirect, url_for
import cv2

app = Flask(__name__)

# Load Haar Cascades for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Global variable to store smile detection status
smile_detected = False
smile_percentage = 0

def calculate_smile_percentage(smile_width, face_width):
    """Calculates the smile percentage based on smile and face width."""
    return int((smile_width / face_width) * 100)

def generate_frames():
    global smile_detected, smile_percentage
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            smile_detected = False  # Reset for each frame

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw rectangle around face
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]

                # Detect smiles
                smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
                if len(smiles) > 0:
                    for (sx, sy, sw, sh) in smiles:
                        smile_percentage = calculate_smile_percentage(sw, w)
                        if smile_percentage > 20:  # Minimum smile threshold for detection
                            smile_detected = True
                        cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 2)
                        cv2.putText(frame, f"Smile: {smile_percentage}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/smile_check')
def smile_check():
    global smile_detected
    return jsonify(smile_detected=smile_detected)

@app.route('/next_page')
def next_page():
    return render_template('next_page.html')

@app.route('/smile_percentage')
def smile_percentage_route():
    global smile_percentage
    return jsonify(smile_percentage=smile_percentage)


@app.route('/detect_smile', methods=['POST'])
def detect_smile():
    # Your smile detection code here
    pass


if __name__ == '__main__':
    from os import environ
    port = int(environ.get('PORT', 5000))  # Get PORT from environment variable (for deployment)
    app.run(host='0.0.0.0', port=port, debug=True)
