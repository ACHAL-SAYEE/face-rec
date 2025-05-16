import os
import cv2
import numpy as np
import sys
from keras_facenet import FaceNet

# Parse arguments from Node.js
if len(sys.argv) != 3:
    print("Usage: python generate_embeddings.py <image_dir> <output_file>")
    sys.exit(1)

image_dir = sys.argv[1]
output_file = sys.argv[2]
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Initialize FaceNet model
embedder = FaceNet()

def load_images_from_directory(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def extract_embeddings(image_paths):
    embeddings = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Unable to read {img_path}")
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = embedder.extract(image_rgb, threshold=0.95)
        for face in faces:
            embeddings.append(face['embedding'])
    return np.array(embeddings)

image_paths = load_images_from_directory(image_dir)
face_embeddings = extract_embeddings(image_paths)
np.save(output_file, face_embeddings)

print(f"Saved {len(face_embeddings)} face embeddings to {output_file}")
