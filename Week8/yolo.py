"""
This file crops face images to a folder.
"""

import os
import cv2
from ultralytics import YOLO

OUTPUT_DIR = "detected_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 0 for webcam, a video file name if you want to use a video file
VIDEO_SOURCE = 0

def main():
    model = YOLO("yolov8n-face.pt")
    seen_ids = set()

    results = model.track(VIDEO_SOURCE, show=True, stream=True, persist=True, verbose=False)

    for result in results:
        frame = result.orig_img

        for obj in result.boxes:
            obj_id = int(obj.id)
            x1, y1, x2, y2 = map(int, obj.xyxy[0])  # Bounding box coordinates

            # Check if the ID is new
            if obj_id not in seen_ids:
                # Crop the face from the frame
                face = frame[y1:y2, x1:x2]
                face_filename = f"{OUTPUT_DIR}/face_id_{obj_id}.jpg"
                cv2.imwrite(face_filename, face)
                seen_ids.add(obj_id)

                print(f"New face detected with ID {obj_id}, saved as {face_filename}")

if __name__ == "__main__":
    main()
