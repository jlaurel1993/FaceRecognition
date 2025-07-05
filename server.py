from flask import Flask, request, jsonify
import face_recognition
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# Define storage paths
UPLOAD_FOLDER = "uploads"
FACE_DATA_FILE = "faces_data.npy"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global storage for known faces
known_faces = []
known_names = []

# Load known faces
def load_faces():
    """Load stored face encodings and names from a file."""
    global known_faces, known_names
    if os.path.exists(FACE_DATA_FILE):
        data = np.load(FACE_DATA_FILE, allow_pickle=True).item()
        known_faces = data["encodings"]
        known_names = data["names"]
        print(f"📂 Loaded {len(known_faces)} known faces: {known_names}")
    else:
        print("⚠️ No known faces found. Please register at least one face.")

# Save known faces
def save_faces():
    """Save face encodings and names to a file."""
    data = {"encodings": known_faces, "names": known_names}
    np.save(FACE_DATA_FILE, data)
    print("✅ Face data saved.")

# Register a face manually
def register_face(name, image_path):
    """Register a new face manually from an image."""
    if not os.path.exists(image_path):
        print(f"❌ File not found: {image_path}")
        return

    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image, num_jitters=3, model="large")

    if encoding:
        known_faces.append(encoding[0])
        known_names.append(name)
        save_faces()  # Save after registering a new face
        print(f"✅ Face registered: {name}")
    else:
        print(f"❌ No face detected in {image_path}")

# Load faces at startup
load_faces()

# Ensure faces are saved and loaded properly
if len(known_faces) == 0:
    print("⚠️ No registered faces found! Registering default faces now.")

    register_face("Alan", "uploads/Alan.jpg")
    register_face("Grandma", "uploads/Grandma.jpg")
    register_face("Jessika", "uploads/Jessika.jpg")
    register_face("Lilah", "uploads/Lilah.jpeg")
    register_face("Elayna", "uploads/Elayna.jpg")

    save_faces()  # Save all registered faces
    load_faces()  # Reload after saving

@app.route('/recognize', methods=['POST'])
def recognize():
    """Receive an image, process it, and return the recognized name."""
    if 'file' not in request.files:
        print("❌ No file provided")
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    print(f"📸 Image received: {file.filename}")

    # Check image properties
    img = Image.open(filepath)
    print(f"📏 Image size: {img.size}, Format: {img.format}")

    # Load image for face recognition
    image = face_recognition.load_image_file(filepath)
    face_encodings = face_recognition.face_encodings(image, num_jitters=3, model="large")

    print(f"🔍 Detected Faces: {len(face_encodings)}")

    detected_name = "Not Recognized"
    if face_encodings:
        for encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_faces, encoding)

            # Debugging: Print all face distances
            for i, distance in enumerate(face_distances):
                print(f"📏 Distance from {known_names[i]}: {distance}")

            if len(face_distances) > 0 and np.min(face_distances) < 0.5:  # Stricter matching
                best_match_index = np.argmin(face_distances)
                detected_name = known_names[best_match_index]

    print(f"✅ Recognized Name: {detected_name}")
    return jsonify({"name": detected_name})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
