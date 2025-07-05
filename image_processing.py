import os
import cv2
import face_recognition
import numpy as np
import bluetooth_audio  
import time

# Test if the 'speak' function is present in the bluetooth_audio module
print("Attributes of bluetooth_audio:", dir(bluetooth_audio))  # Debugging the module

# Load and encode known faces
def load_face(image_path):
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    return encoding[0] if encoding else None  # Handle case where no face is detected

# Store known faces and names
known_faces = []
known_names = []

# Add known faces
# Use correct path for uploaded images
UPLOAD_FOLDER = "uploads"

faces_list = [
    ("Alan", os.path.join(UPLOAD_FOLDER, "Alan.jpg")),
    ("Lilah", os.path.join(UPLOAD_FOLDER, "Lilah.jpeg")),
    ("Jessika", os.path.join(UPLOAD_FOLDER, "Jessika.jpg")),
    ("Grandma", os.path.join(UPLOAD_FOLDER, "Grandma.jpg")),
    ("Elayna", os.path.join(UPLOAD_FOLDER, "Elayna.jpg"))
]

for name, img_path in faces_list:
    encoding = load_face(img_path)
    if encoding is not None:
        known_faces.append(encoding)
        known_names.append(name)
    else:
        print(f"Warning: No face detected in {img_path}. Skipping.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Track detected faces
current_faces = set()
last_seen = {}  # Dictionary to track when a face was last seen

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Optimize performance
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    detected_faces = set()  # Temporary storage for currently detected faces

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)
        name = "Unknown"

        if face_distances[best_match_index] < 0.6:
            name = known_names[best_match_index]
            detected_faces.add(name)  # Add detected name to the set

            # Announce only if the face was gone for 3+ seconds or is newly detected
            if name not in current_faces or (name in last_seen and time.time() - last_seen[name] > 3):
                bluetooth_audio.speak(f"This is {name}")  # Use the speak function from the bluetooth_audio module
            
            # Update last seen time
            last_seen[name] = time.time()

        # Scale back the face locations since we resized the frame
        top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Track faces that have disappeared
    for person in current_faces:
        if person not in detected_faces:
            last_seen[person] = time.time()  # Store the last time the person was seen

    # Update the current_faces set
    current_faces = detected_faces

    # Show the result
    cv2.imshow("Face Recognition", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Cleanup
video_capture.release()
cv2.destroyAllWindows()
