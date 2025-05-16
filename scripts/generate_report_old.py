import cv2
import numpy as np
import os
import pandas as pd
from datetime import timedelta
from keras_facenet import FaceNet
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
import mediapipe as mp
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import screeninfo
import os
import sys

videofile = sys.argv[1]

# -------------------- Configuration --------------------
video_path = r"C:\node js\face-rec\uploads\test-video\{}".format(videofile)
print(video_path)
embeddings_dir = r"C:\node js\face-rec\embeddings"
excel_path =sys.argv[2]
outputPath=r"C:\node js\face-rec\outputs\attendance_report.xlsx"
EAR_THRESHOLD = 0.2
LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
# -------------------------------------------------------

# Load or initialize attendance DataFrame
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
else:
    df = pd.DataFrame(columns=["Name", "Entry Timestamp", "Exit Timestamp", "% of Presence", "Sleep Duration"])

# Initialize models
embedder = FaceNet()
detector = MTCNN()
mp_face_mesh = mp.solutions.face_mesh
yolo_model = YOLO("yolov8n.pt")  # Ensure this model file is available
tracker = DeepSort(max_age=1)
frame_skip=30
# Load stored face embeddings
stored_embeddings = {}
for file in os.listdir(embeddings_dir):
    if file.endswith(".npy"):
        person_name = os.path.splitext(file)[0].split("-")[1]
        stored_embeddings[person_name] = np.load(os.path.join(embeddings_dir, file))

# Function to detect faces using MTCNN
def detect_faces(image):
    if image is None or image.size == 0:
        print("Warning: Empty or invalid image passed to face detector.")
        return [], []

    faces = detector.detect_faces(image)
    face_boxes, face_crops = [], []

    for face in faces:
        x, y, width, height = face["box"]
        x, y = max(0, x), max(0, y)
        
        # Check if the bounding box size is valid (not a point)
        if width <= 0 or height <= 0:
            print(f"Warning: Invalid face box detected: {x}, {y}, {width}, {height}")
            continue

        face_crop = image[y:y+height, x:x+width]
        if face_crop.size == 0:
            print(f"Warning: Empty face crop for box: {x}, {y}, {width}, {height}")
            continue

        face_boxes.append((x, y, x+width, y+height))
        face_crops.append(face_crop)

    return face_boxes, face_crops

# Function to extract embeddings
def get_embeddings(faces):
    embeddings = []
    for face in faces:
        face_resized = cv2.resize(face, (160, 160))
        face_emb = embedder.embeddings([face_resized])[0]
        embeddings.append(face_emb)
    return np.array(embeddings)

# Function to match faces

# Function to match faces
def match_faces(new_embeddings, stored_embeddings, threshold=0.3):
    matches = []

    for new_emb in new_embeddings:
        best_match, best_score = None, float("inf")

        for person, stored_emb_list in stored_embeddings.items():
            for stored_emb in stored_emb_list:
                score = cosine(new_emb, stored_emb)
                if score < best_score:
                    best_score = score
                    best_match = person

        if best_match and best_score < threshold:
            matches.append(best_match)
        else:
            matches.append("Unknown")

    return matches

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(landmarks, eye_points):
    A = np.linalg.norm(landmarks[eye_points[1]] - landmarks[eye_points[5]])
    B = np.linalg.norm(landmarks[eye_points[2]] - landmarks[eye_points[4]])
    C = np.linalg.norm(landmarks[eye_points[0]] - landmarks[eye_points[3]])
    return (A + B) / (2.0 * C)

# Initialize video capture
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_no = 0
track_id_to_name = {}  # TrackID -> Name
track_id_to_data = {}  # TrackID -> Attendance and sleep data

# Get screen size for display
screen = screeninfo.get_monitors()[0]
max_width, max_height = screen.width, screen.height

# ... (top unchanged code till video capture)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.2, min_tracking_confidence=0.2) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        if frame_no % frame_skip!=0:
            continue
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect persons using YOLOv8
        results = yolo_model.predict(frame, classes=[0], conf=0.5, verbose=False)[0]

        detections = []
        if results.boxes is not None and results.boxes.data is not None:
            for box in results.boxes.data.tolist():
                x1, y1, x2, y2 = map(int, box[:4])
                conf = float(box[4])
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
                print("sand",[x1, y1, x2 - x1, y2 - y1])
        # Update tracker
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r, b = track.to_ltrb()
            x1, y1, x2, y2 = map(int, [l, t, r, b])
            print("land",[x1, y1, x2 - x1, y2 - y1])
            if(x2-x1 <0 or y2-y1 <0):
                continue
            name = track_id_to_name.get(track_id, "Unknown")

            # Perform face recognition if not already recognized
            if name == "Unknown":
                face_crop = frame_rgb[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_boxes, face_crops = detect_faces(face_crop)

                    if face_crops:
                        embeddings = get_embeddings(face_crops)
                        names = match_faces(embeddings, stored_embeddings)

                        if names[0] != "Unknown":
                            name = names[0]
                            track_id_to_name[track_id] = name

            if name == "Unknown":
                continue

            # Initialize or update attendance data
            if track_id not in track_id_to_data:
                track_id_to_data[track_id] = {
                    "name": name,
                    "entry_time": frame_no,
                    "exit_time": frame_no,
                    "presence_count": 0,
                    "sleep_duration": 0,
                    "sleep_start": None
                }

            data = track_id_to_data[track_id]
            data["exit_time"] = frame_no
            data["presence_count"] += 30

            # Sleep detection using MediaPipe FaceMesh
            face_roi = frame_rgb[y1:y2, x1:x2]
            results_mesh = face_mesh.process(face_roi)
            if results_mesh.multi_face_landmarks:
                for face_landmarks in results_mesh.multi_face_landmarks:
                    try:
                        landmarks = np.array([[p.x * (x2 - x1), p.y * (y2 - y1)] for p in face_landmarks.landmark])
                        if landmarks.shape[0] > max(max(LEFT_EYE_LANDMARKS), max(RIGHT_EYE_LANDMARKS)):
                            left_EAR = eye_aspect_ratio(landmarks, LEFT_EYE_LANDMARKS)
                            right_EAR = eye_aspect_ratio(landmarks, RIGHT_EYE_LANDMARKS)
                            avg_EAR = (left_EAR + right_EAR) / 2.0

                            if avg_EAR < EAR_THRESHOLD:
                                if data["sleep_start"] is None:
                                    data["sleep_start"] = frame_no
                            else:
                                if data["sleep_start"] is not None:
                                    data["sleep_duration"] += (frame_no - data["sleep_start"])
                                    data["sleep_start"] = None
                    except Exception as e:
                        print(f"[WARN] EAR landmark error: {e}")

            # Draw bounding boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Resize if too large for screen
        if frame.shape[1] > max_width or frame.shape[0] > max_height:
            scale_x = max_width / frame.shape[1]
            scale_y = max_height / frame.shape[0]
            scale = min(scale_x, scale_y)
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        # Display the frame
        cv2.imshow("YOLOv8 + DeepSORT Attendance", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
# Final attendance write
from collections import defaultdict

# Step 1: Aggregate all data per person
person_to_data = defaultdict(lambda: {
    "entry_time": float("inf"),
    "exit_time": 0,
    "presence_count": 0,
    "sleep_duration": 0.0
})

for data in track_id_to_data.values():
    person = data["name"]
    person_to_data[person]["entry_time"] = min(person_to_data[person]["entry_time"], data["entry_time"])
    person_to_data[person]["exit_time"] = max(person_to_data[person]["exit_time"], data["exit_time"])
    person_to_data[person]["presence_count"] += data["presence_count"]
    person_to_data[person]["sleep_duration"] += data["sleep_duration"]

# Step 2: Fill/update the DataFrame
for person, pdata in person_to_data.items():
    attendance_percentage = (pdata["presence_count"] / frame_no) * 100
    entry_time = str(timedelta(seconds=(pdata["entry_time"] / 30)))  # Assuming 30 fps
    exit_time = str(timedelta(seconds=(pdata["exit_time"] / 30)))
    sleep_duration = round(pdata["sleep_duration"] / 30, 2)

    if person in df["Name"].values:
        df.loc[df["Name"] == person, ["Entry Timestamp", "Exit Timestamp", "% of Presence", "Sleep Duration"]] = [
            entry_time, exit_time, round(attendance_percentage, 2), sleep_duration
        ]
    else:
        new_entry = pd.DataFrame({
            "Name": [person],
            "Entry Timestamp": [entry_time],
            "Exit Timestamp": [exit_time],
            "% of Presence": [round(attendance_percentage, 2)],
            "Sleep Duration": [sleep_duration]
        })
        df = pd.concat([df, new_entry], ignore_index=True)

# Save to Excel
df.to_excel(outputPath, index=False)
print("✅ Attendance updated successfully.")

cap.release()
cv2.destroyAllWindows()


