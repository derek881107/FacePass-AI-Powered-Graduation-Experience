import os
import cv2
import face_recognition
import math
import numpy as np
from collections import defaultdict
import json
import time

def face_confidence(face_distance, face_match_threshold=0.45):  # 進一步降低閾值從0.5到0.45
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
        self.face_detection_model = "hog"  # Can be set to "hog" (CPU) or "cnn" (GPU)
        self.people_data = {}  # To store detailed information about people
        
        # 容忍度設定 (更嚴格)
        self.recognition_tolerance = 0.45  # 降低到0.45
        self.maybe_tolerance = 0.55       # 降低到0.55
        
        # 信心度門檻 (提高)
        self.confidence_threshold = 60.0  # 提高到60%
        
        # 連續幀確認 (更嚴格)
        self.face_history = defaultdict(list)  # 儲存過去幾幀的辨識結果
        self.history_size = 10               # 增加到10幀
        self.consistency_threshold = 0.8     # 提高到80%一致才確認身份
        
        # 多角度驗證 (確保正臉和側臉都能辨識)
        self.angle_verification = True
        
        # 記錄最後一次辨識成功時間
        self.last_recognition_time = {}
        
        # Get the directory of the current script
        self.script_dir = os.path.dirname(os.path.abspath(__file__))

        # Ensure faces directory exists
        faces_dir = os.path.join(self.script_dir, 'faces')
        if not os.path.exists(faces_dir):
            os.makedirs(faces_dir)
            print("📁 Created faces directory, please add facial photos")
            print("   Photo filename format should be: name_number.jpg (e.g., john_1.jpg)")
        
        # Load people data from JSON file
        self.load_people_data()
        
        self.encode_faces()
    
    def load_people_data(self):
        """Load detailed information about people from JSON file"""
        json_path = os.path.join(self.script_dir, 'people_data.json')
        
        # Create a template JSON if it doesn't exist
        if not os.path.exists(json_path):
            print("📄 Creating template people_data.json file...")
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
            
            print("✅ Created template people_data.json. Please edit it with your actual data.")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.people_data = json.load(f)
            print(f"✅ Successfully loaded data for {len(self.people_data)} people from people_data.json")
        except Exception as e:
            print(f"⚠️ Error loading people data: {str(e)}")
            self.people_data = {}

    def encode_faces(self):
        print("📂 Scanning faces directory...")
        face_data = defaultdict(list)
        
        faces_dir = os.path.join(self.script_dir, 'faces')
        face_files = [f for f in os.listdir(faces_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not face_files:
            print("⚠️ No photo files in faces directory! Please add photos and run again")
            return

        for image in face_files:
            try:
                image_path = os.path.join(faces_dir, image)
                print(f"🔍 Processing photo: {image}")
                
                # Check if file exists and is readable
                if not os.path.isfile(image_path) or os.path.getsize(image_path) == 0:
                    print(f"⚠️ File {image} does not exist or is empty")
                    continue
                
                # Directly use OpenCV to read the photo and check if successful
                img = cv2.imread(image_path)
                if img is None:
                    print(f"⚠️ Cannot read {image} with OpenCV")
                    continue
                
                # Convert color space with OpenCV before providing to face_recognition
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Try to detect face locations
                face_locations = face_recognition.face_locations(rgb_img, model=self.face_detection_model)
                
                if not face_locations:
                    print(f"⚠️ Cannot find face in {image}, trying to adjust parameters...")
                    # Try lowering detection threshold
                    face_locations = face_recognition.face_locations(
                        rgb_img, 
                        model=self.face_detection_model,
                        number_of_times_to_upsample=2  # Increase this value may help detect smaller or unclear faces
                    )
                
                if face_locations:
                    print(f"✓ Found {len(face_locations)} faces in {image}")
                    
                    # Get face encodings - increased jitters for more accuracy
                    encodings = face_recognition.face_encodings(rgb_img, face_locations, num_jitters=5)  # 提高到5
                    
                    if encodings:
                        # Extract name from filename (supports more naming formats)
                        name = os.path.splitext(image)[0]
                        if '_' in name:
                            name = name.split('_')[0]  # Take the part before "_" as the name
                        
                        face_data[name].append(encodings[0])
                        print(f"✅ Successfully encoded: {name}")
                        
                        # Check if this person is in the JSON data
                        if name not in self.people_data:
                            print(f"⚠️ {name} is not in people_data.json. Please add their information.")
                    else:
                        print(f"⚠️ Cannot generate face encoding in {image}")
                else:
                    print(f"⚠️ Cannot find face in {image}, please check photo clarity")
            except Exception as e:
                print(f"🚨 Error occurred when processing {image}: {str(e)}")

        if not face_data:
            print("⚠️ No faces successfully read! Please check photo quality or format")
            return

        for name, encodings in face_data.items():
            # 使用所有照片而不只是平均值，以保存更多面部特徵
            for encoding in encodings:
                self.known_face_encodings.append(encoding)
                self.known_face_names.append(name)

        print(f"✅ Successfully loaded {len(self.known_face_encodings)} face encodings for {len(face_data)} people")

    def verify_face_consistency(self, face_location, name):
        """Check if this face has been consistently identified as the same person"""
        # Create a unique key for this face based on its location
        # (we assume a face at roughly the same position is the same person)
        top, right, bottom, left = face_location
        face_center = ((left + right) // 2, (top + bottom) // 2)
        face_size = (right - left) * (bottom - top)  # 臉部大小作為額外匹配依據
        
        # Find all tracked faces and update or add this one
        matched_key = None
        for key in list(self.face_history.keys()):
            last_center, last_size, last_names = self.face_history[key][-1]
            # 同時考慮位置和大小的一致性
            center_distance = ((last_center[0] - face_center[0])**2 + (last_center[1] - face_center[1])**2)**0.5
            size_ratio = max(face_size, last_size) / max(1, min(face_size, last_size))
            
            if center_distance < 50 and size_ratio < 1.5:  # 更嚴格的匹配條件
                matched_key = key
                break
        
        if matched_key is None:
            # New face, create a new entry
            matched_key = len(self.face_history)
            
        # Add this detection to history
        self.face_history[matched_key].append((face_center, face_size, name))
        
        # Keep only the last N detections
        if len(self.face_history[matched_key]) > self.history_size:
            self.face_history[matched_key].pop(0)
        
        # 檢查時間一致性 - 如果某人剛被辨識過，加強確信度
        current_time = time.time()
        recently_recognized = False
        
        if name in self.last_recognition_time:
            time_since_last = current_time - self.last_recognition_time[name]
            if time_since_last < 5.0:  # 5秒內
                recently_recognized = True
        
        # Return true if we have enough history and the detections are consistent
        if len(self.face_history[matched_key]) >= 5:  # 需要至少5幀來確認
            names = [n for _, _, n in self.face_history[matched_key]]
            # Count occurrences of the current name
            name_count = sum(1 for n in names if n == name)
            consistency = name_count / len(names)
            
            # 如果最近剛辨識過，降低一致性要求
            threshold = self.consistency_threshold
            if recently_recognized:
                threshold = max(0.5, threshold - 0.2)
                
            if consistency >= threshold:
                self.last_recognition_time[name] = current_time
                return True
        
        return False

    def verify_with_multiple_models(self, frame, face_location, name):
        """使用多種模型驗證同一張臉"""
        # 這裡僅做示範，實際應調用不同模型或參數
        top, right, bottom, left = face_location
        # 擴大臉部區域以確保包含整個臉
        top = max(0, top - 20)
        right = min(frame.shape[1], right + 20)
        bottom = min(frame.shape[0], bottom + 20)
        left = max(0, left - 20)
        
        # 提取臉部區域
        face_img = frame[top:bottom, left:right]
        
        if face_img.size == 0:
            return False
            
        # 轉換為RGB
        rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # 使用更高的jitters再次編碼
        face_encodings = face_recognition.face_encodings(rgb_face, [(0, face_img.shape[1], face_img.shape[0], 0)], num_jitters=10)
        
        if not face_encodings:
            return False
            
        # 與所有已知臉進行比對，使用更嚴格的容忍度
        strict_matches = face_recognition.compare_faces(
            self.known_face_encodings,
            face_encodings[0],
            tolerance=self.recognition_tolerance - 0.05  # 更嚴格的容忍度
        )
        
        # 找出所有匹配的名字
        matched_names = [self.known_face_names[i] for i in range(len(strict_matches)) if strict_matches[i]]
        
        # 檢查名字是否一致
        return name in matched_names and matched_names.count(name) / max(1, len(matched_names)) >= 0.7

    def run_recognition(self):
        # Try to open webcam
        video_capture = cv2.VideoCapture(0)
        
        # If cannot open webcam
        if not video_capture.isOpened():
            print("❌ Cannot open webcam, trying other cameras...")
            # Try other camera indices
            for i in range(1, 3):
                video_capture = cv2.VideoCapture(i)
                if video_capture.isOpened():
                    print(f"✅ Successfully opened camera #{i}")
                    break
            
            if not video_capture.isOpened():
                print("❌ Cannot open any camera, program terminated")
                return

        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Face Recognition', 800, 600)

        print("🎥 Starting webcam face recognition (press q to exit)")
        
        # Confirm if known faces are loaded
        if not self.known_face_encodings:
            print("⚠️ Warning: No known faces loaded, cannot perform matching!")

        frame_count = 0
        detection_interval = 2  # Detect every few frames
        
        # Initialize these attributes to prevent errors
        self.face_details = []
        self.face_colors = []
        self.verified_faces = []  # 儲存已驗證的臉部
        self.verification_level = []  # 儲存驗證等級

        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("⚠️ Cannot read camera frame")
                break

            frame_count += 1
            
            # Resize frame to speed up processing, but not too small
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Perform face detection every fixed number of frames
            if frame_count % detection_interval == 0:
                self.face_locations = face_recognition.face_locations(
                    rgb_small_frame,
                    model=self.face_detection_model, 
                    number_of_times_to_upsample=1  # Increase this value can improve small face detection rate, but will be slower
                )
                
                if self.face_locations:
                    print(f"🧠 Detected {len(self.face_locations)} faces")

                    try:
                        self.face_encodings = face_recognition.face_encodings(
                            rgb_small_frame, 
                            self.face_locations,
                            num_jitters=3  # 進一步提高到3
                        )
                    except Exception as e:
                        print(f"🚨 Error occurred during face encoding: {e}")
                        self.face_encodings = []

                    self.face_names = []
                    self.face_details = []  # To store detailed info for each face
                    self.face_colors = []   # To store color for each face
                    self.verified_faces = []  # 儲存是否為已驗證臉部
                    self.verification_level = []  # 儲存驗證等級 (0-100)

                    for face_index, face_encoding in enumerate(self.face_encodings):
                        if face_encoding is None or len(self.known_face_encodings) == 0:
                            self.face_names.append("Unknown")
                            self.face_details.append(None)
                            self.face_colors.append((0, 0, 255))  # Red color for unknown faces
                            self.verified_faces.append(False)
                            self.verification_level.append(0)
                            continue

                        # Compare face features with stricter tolerance
                        matches = face_recognition.compare_faces(
                            self.known_face_encodings, 
                            face_encoding, 
                            tolerance=self.recognition_tolerance  # 更嚴格的比對
                        )
                        name = "Unknown"
                        confidence = '???'
                        details = None
                        color = (0, 0, 255)  # Default red for unknown
                        is_verified = False
                        verify_level = 0

                        # Find the closest known face
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings, 
                            face_encoding
                        )
                        
                        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None

                        if best_match_index is not None:
                            # Calculate similarity
                            distance = face_distances[best_match_index]
                            confidence = face_confidence(distance)
                            
                            # Extract confidence percentage
                            confidence_value = float(confidence.strip('%'))
                            
                            # Display closest name and similarity
                            if matches[best_match_index] and confidence_value >= self.confidence_threshold:
                                name = self.known_face_names[best_match_index]
                                
                                # 第一階段：連續幀一致性驗證
                                frame_consistent = self.verify_face_consistency(self.face_locations[face_index], name)
                                
                                # 如果連續幀驗證通過，進行第二階段：多模型交叉驗證
                                if frame_consistent and self.angle_verification:
                                    multi_model_verified = self.verify_with_multiple_models(frame, 
                                                               [t*2 for t in self.face_locations[face_index]], name)
                                    
                                    if multi_model_verified:
                                        # 設定為已完全驗證
                                        if name in self.people_data:
                                            details = self.people_data[name]
                                        color = (0, 255, 0)  # Green for recognized
                                        is_verified = True
                                        verify_level = 100
                                    else:
                                        # 只通過了單一驗證，但信心值較高
                                        color = (0, 255, 128)  # 淡綠色
                                        name = f"Validating {name}"
                                        verify_level = 75
                                else:
                                    if frame_consistent:
                                        # 通過了連續幀驗證，但沒啟用多模型驗證
                                        color = (0, 255, 128)  # 淡綠色
                                        verify_level = 80
                                    else:
                                        # 不夠連續幀驗證，顯示為黃色
                                        color = (0, 255, 255)  # Yellow
                                        name = f"Verifying {name}"
                                        verify_level = 50
                            else:
                                # Even if not completely matched, display closest name with low similarity
                                if distance < self.maybe_tolerance and confidence_value >= self.confidence_threshold * 0.7:
                                    name = f"Maybe {self.known_face_names[best_match_index]}"
                                    if self.known_face_names[best_match_index] in self.people_data:
                                        details = self.people_data[self.known_face_names[best_match_index]]
                                    color = (0, 255, 255)  # Yellow for maybe
                                    verify_level = 30
                                else:
                                    color = (0, 0, 255)  # Red for unknown
                                    verify_level = 0

                        self.face_names.append(f'{name} ({confidence})')
                        self.face_details.append(details)
                        self.face_colors.append(color)
                        self.verified_faces.append(is_verified)
                        self.verification_level.append(verify_level)

            # Mark all detected faces
            for i, ((top, right, bottom, left), name, color, is_verified, verify_level) in enumerate(zip(
                [(t * 2, r * 2, b * 2, l * 2) for (t, r, b, l) in self.face_locations],  # Because scaling ratio is 0.5
                self.face_names,
                self.face_colors,
                self.verified_faces,
                self.verification_level)):
                
                # Face border with color based on recognition status - thicker border (4 pixels)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 4)
                
                # Make the name box larger
                name_box_height = 50
                name_font_size = 1.0
                
                # Name background - larger box
                cv2.rectangle(frame, (left, bottom - name_box_height), (right, bottom), color, cv2.FILLED)
                
                # Display name and confidence - larger text
                cv2.putText(frame, name, (left + 6, bottom - 15), 
                           cv2.FONT_HERSHEY_DUPLEX, name_font_size, (0, 0, 0), 2)
                
                # Add a verification indicator
                if is_verified:
                    cv2.putText(frame, "✓✓", (right - 40, top + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                elif verify_level >= 75:
                    cv2.putText(frame, "✓", (right - 25, top + 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)
                
                # 顯示驗證等級條
                bar_length = 60
                filled_length = int(bar_length * verify_level / 100)
                cv2.rectangle(frame, (left, top - 15), (left + bar_length, top - 5), (100, 100, 100), cv2.FILLED)
                if verify_level > 0:
                    # 顏色隨驗證等級變化
                    if verify_level < 40:
                        bar_color = (0, 0, 255)  # 紅色 (低)
                    elif verify_level < 80:
                        bar_color = (0, 255, 255)  # 黃色 (中)
                    else:
                        bar_color = (0, 255, 0)  # 綠色 (高)
                    cv2.rectangle(frame, (left, top - 15), (left + filled_length, top - 5), bar_color, cv2.FILLED)
                
                # Display detailed information if available
                details = None
                if i < len(self.face_details):
                    details = self.face_details[i]
                    
                if details and is_verified:  # 只顯示已驗證臉部的詳細資訊
                    # Increase font size and adjust spacing
                    font_size = 1.0
                    line_height = 40
                    info_width = 400
                    info_height = 180
                    
                    y_offset = top - 15
                    
                    # Background for details (semi-transparent)
                    detail_bg = frame.copy()
                    cv2.rectangle(detail_bg, (right, top - 15), (right + info_width, top + info_height), (0, 0, 0), cv2.FILLED)
                    cv2.addWeighted(detail_bg, 0.7, frame, 0.3, 0, frame)
                    
                    # Display each piece of information with larger text
                    if "Title" in details:
                        y_offset += line_height
                        cv2.putText(frame, f"Title: {details['Title']}", (right + 10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
                    
                    if "Department" in details:
                        y_offset += line_height
                        cv2.putText(frame, f"Dept: {details['Department']}", (right + 10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
                    
                    if "Major" in details:
                        y_offset += line_height
                        cv2.putText(frame, f"Major: {details['Major']}", (right + 10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
                    
                    if "Graduate" in details:
                        y_offset += line_height
                        cv2.putText(frame, f"Graduate: {details['Graduate']}", (right + 10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)

            # Show current recognition strictness
            strictness_info = f"Strict Mode: Tolerance={self.recognition_tolerance} | Confidence={self.confidence_threshold}% | Consistency={self.consistency_threshold*100}%"
            cv2.putText(frame, strictness_info, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to exit | 'r' to reload | '+/-' to adjust strictness", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Face Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("🔄 Reloading face data...")
                self.known_face_encodings = []
                self.known_face_names = []
                self.face_history.clear()
                self.last_recognition_time.clear()
                self.load_people_data()
                self.encode_faces()
            elif key == ord('+'):
                # 提高嚴謹度 (降低容忍度)
                self.recognition_tolerance = max(0.3, self.recognition_tolerance - 0.05)
                self.maybe_tolerance = max(0.4, self.maybe_tolerance - 0.05)
                self.confidence_threshold = min(90.0, self.confidence_threshold + 5.0)
                self.consistency_threshold = min(0.95, self.consistency_threshold + 0.05)
                print(f"🔒 Maximum strictness: tolerance={self.recognition_tolerance}, threshold={self.confidence_threshold}%, consistency={self.consistency_threshold}")
            elif key == ord('-'):
                # 降低嚴謹度 (提高容忍度)
                self.recognition_tolerance = min(0.7, self.recognition_tolerance + 0.05)
                self.maybe_tolerance = min(0.8, self.maybe_tolerance + 0.05)
                self.confidence_threshold = max(30.0, self.confidence_threshold - 5.0)
                self.consistency_threshold = max(0.5, self.consistency_threshold - 0.05)
                print(f"🔓 Decreased strictness: tolerance={self.recognition_tolerance}, threshold={self.confidence_threshold}%, consistency={self.consistency_threshold}")

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    print("🚀 Starting Real-time Face Recognition System in STRICT MODE...")
    fr = FaceRecognition()
    fr.run_recognition()