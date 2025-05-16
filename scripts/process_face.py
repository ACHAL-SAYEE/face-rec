import sys
import os
import cv2
from retinaface import RetinaFace

image_path = sys.argv[1]
name = sys.argv[2]
rollno = sys.argv[3]

# database_path = "database"
# os.makedirs(database_path, exist_ok=True)

image = cv2.imread(image_path)
if image is None:
    print("ERROR: Cannot load image", file=sys.stderr)
    sys.exit(1)

faces = RetinaFace.detect_faces(image)

if not faces:
    print("No face detected.")
    sys.exit(0)

if len(faces) > 1:
    print("MULTIPLE_FACES")
    sys.exit(0)

# Only one face
for key in faces:
    x1, y1, x2, y2 = faces[key]["facial_area"]
    cropped = image[y1:y2, x1:x2]
    # filename = f"{name}_{rollno}.jpg"
    out_path = os.path.join(image_path)
    cv2.imwrite(out_path, cropped)
    print(f"Saved {image_path}")
    break
