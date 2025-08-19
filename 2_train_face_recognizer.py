import face_recognition
import pickle
import cv2
import os

DATASET_DIR = 'dataset'
ENCODINGS_FILE = 'encodings.pickle'

print("[INFO] Starting to process faces...")
known_encodings = []
known_names = []

# Loop over the folders in the dataset
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing images for: {person_name}")
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)
        
        # Load image and convert it from BGR to RGB
        image = cv2.imread(image_path)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect the (x, y)-coordinates of the bounding boxes corresponding to each face
        boxes = face_recognition.face_locations(rgb, model='hog')

        # Compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # Add each encoding + name to our list
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save the facial encodings and names to disk
print("[INFO] Serializing encodings...")
data = {"encodings": known_encodings, "names": known_names}
with open(ENCODINGS_FILE, "wb") as f:
    f.write(pickle.dumps(data))
    
print(f"[INFO] Encodings saved to {ENCODINGS_FILE}")