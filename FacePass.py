import os
import cv2
import face_recognition
import math
import numpy as np
from collections import defaultdict
import json
import time

# Converts face distance to a percentage-based confidence score
def face_confidence(face_distance, face_match_threshold=0.45):
    value_range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (value_range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.face_detection_model = "hog"  # Options: "hog" (CPU-friendly), "cnn" (requires GPU)
        self.people_data = {}  # Dictionary for storing user profile info

        # Recognition strictness settings
        self.recognition_tolerance = 0.45
        self.maybe_tolerance = 0.55

        # Minimum confidence required to confirm identity
        self.confidence_threshold = 60.0

        # Frame history tracking for consistent recognition
        self.face_history = defaultdict(list)
        self.history_size = 10
        self.consistency_threshold = 0.8

        # Enable cross-angle face verification
        self.angle_verification = True

        # Store timestamps of the last successful recognitions
        self.last_recognition_time = {}

        # Locate current script directory
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Ensure face image directory exists
        faces_dir = os.path.join(self.script_dir, 'faces')
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print("📁 Created 'faces' directory. Please add facial images.")
            print("   Naming convention: name_number.jpg (e.g., john_1.jpg)")

        # Load associated user profile data
        self.load_people_data()
        self.encode_faces()

    def load_people_data(self):
        """Load user profile data from people_data.json"""
        json_path = os.path.join(self.script_dir, 'people_data.json')

        # If missing, create a default JSON template
        if not os.path.exists(json_path):
            print("📄 Creating default people_data.json...")
            template_data = {
                "John": {
                    "Title": "Professor",
                    "Department": "Computer Science",
                    "Major": "AI",
                    "Graduate": "Yes"
                },
                "Alice": {
                    "Title": "Student",
                    "Department": "Engineering",
                    "Major": "Robotics",
                    "Graduate": "No"
                }
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(template_data, f, indent=4, ensure_ascii=False)
            print("✅ Default people_data.json created. Please update it with actual user data.")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.people_data = json.load(f)
            print(f"✅ Loaded {len(self.people_data)} user profiles from people_data.json")
        except Exception as e:
            print(f"⚠️ Failed to load user profiles: {str(e)}")
            self.people_data = {}

    def encode_faces(self):
        print("📂 Scanning 'faces' directory for images...")
        face_data = defaultdict(list)

        faces_dir = os.path.join(self.script_dir, 'faces')
        face_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not face_files:
            print("⚠️ No image files found. Please add photos and rerun.")
            return

        for image in face_files:
            try:
                image_path = os.path.join(faces_dir, image)
                print(f"🔍 Processing: {image}")

                # Validate file existence and size
                if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
                    print(f"⚠️ File {image} is missing or empty.")
                    continue

                # Read the image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"⚠️ Unable to load {image} using OpenCV.")
                    continue

                # Convert color format to RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Detect face locations
                face_locations = face_recognition.face_locations(rgb_img, model=self.face_detection_model)

                if not face_locations:
                    print(f"⚠️ No face detected in {image}. Retrying with adjusted parameters...")
                    face_locations = face_recognition.face_locations(rgb_img, model=self.face_detection_model, number_of_times_to_upsample=2)

                if face_locations:
                    print(f"✓ Detected {len(face_locations)} face(s) in {image}")
                    encodings = face_recognition.face_encodings(rgb_img, face_locations, num_jitters=5)

                    if encodings:
                        name = os.path.splitext(image)[0]
                        if '_' in name:
                            name = name.split('_')[0]

                        face_data[name].append(encodings[0])
                        print(f"✅ Encoded face for: {name}")

                        if name not in self.people_data:
                            print(f"⚠️ {name} not found in people_data.json. Please add their details.")
                    else:
                        print(f"⚠️ Failed to extract face encoding from {image}")
                else:
                    print(f"⚠️ No faces found in {image}. Please check image quality.")
            except Exception as e:
                print(f"🚨 Error processing {image}: {str(e)}")

        if not face_data:
            print("⚠️ No valid faces encoded. Please verify image format and quality.")
            return

        for name, encodings in face_data.items():
            for encoding in encodings:
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)

        print(f"✅ Loaded {len(self.known_face_encodings)} encodings for {len(face_data)} users")
