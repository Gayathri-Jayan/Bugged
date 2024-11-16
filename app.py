from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load pre-trained classifiers for face and smile detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

@app.route('/detect_smile', methods=['POST'])
def detect_smile():
    data = request.get_json()
    image_data = data['image']  # Get the base64-encoded image data

    # Decode the base64 image data
    img_data = base64.b64decode(image_data.split(',')[1])
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    img = cv2.imdecode(np_img, flags=1)  # Convert byte data to an image

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    smile_detected = False

    # Check for smiles in the detected faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            smile_detected = True

    return jsonify(smileDetected=smile_detected)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
