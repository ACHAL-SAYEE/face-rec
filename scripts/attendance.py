import cv2
import numpy as np
import os
import pandas as pd
from keras_facenet import FaceNet
from mtcnn import MTCNN
from scipy.spatial.distance import cosine
import sys

# Arguments: image path, excel path
raw_path = sys.argv[1]
print(raw_path)
# Normalize to handle backslashes and spaces properly
image_path = r"C:\node js\face-rec\uploads\test-image\{}".format(raw_path)
excel_path = sys.argv[2]
outputPath=r"C:\node js\face-rec\outputs\attendance.xlsx"

# Load models
embedder = FaceNet()
detector = MTCNN()

# Load stored embeddings
embeddings_dir = r"C:\node js\face-rec\embeddings"
stored_embeddings = {}
for file in os.listdir(embeddings_dir):
    if file.endswith(".npy"):
        person_name = os.path.splitext(file)[0].split("-")[1]
        stored_embeddings[person_name] = np.load(os.path.join(embeddings_dir, file))

# Read or initialize Excel
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
else:
    df = pd.DataFrame({"Name": list(stored_embeddings.keys()), "Marked": [0]*len(stored_embeddings)})

# Read the image
image_path = os.path.abspath(image_path)
print("image_path ",image_path)

image = cv2.imread(image_path)
if image is None:
    print("Error: Could not read image.")
    sys.exit(1)

# Detect faces
faces = detector.detect_faces(image)
identified_names = set()

for face in faces:
    x, y, w, h = face['box']
    x, y = max(0, x), max(0, y)
    face_crop = image[y:y+h, x:x+w]
    face_resized = cv2.resize(face_crop, (160, 160))
    new_embedding = embedder.embeddings([face_resized])[0]

    best_match, best_score = None, float('inf')
    for name, emb_array in stored_embeddings.items():
        for emb in emb_array:
            score = cosine(new_embedding, emb)
            if score < best_score:
                best_score = score
                best_match = name

    if best_score < 0.3:  # Threshold
        identified_names.add(best_match)

# Update DataFrame
df["Marked"] = df["Name"].apply(lambda name: 1 if name in identified_names else 0)

# Save Excel
df.to_excel(outputPath, index=False)
print(f"Recognition complete. Results saved to {excel_path}")
