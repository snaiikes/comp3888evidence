"""
This file continuously captures images of faces from a video source.
Face images are saved to the 'detected_faces' folder every few frames.
Face images saved in the folder are periodically processed by DeepFace in demographics.py.
"""

import multiprocessing as mp
import subprocess
import os
import cv2
from ultralytics import YOLO

CAPTURE_IMAGE_PER_FRAMES = 5 # An image will be captured every "this" number of frames
PROCESS_IMAGES_PER_FRAMES = 15 # DeepFace will run analysis every "this" number of frames

OUTPUT_DIR = "detected_faces"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 0 for webcam, a video file name if you want to use a video file
VIDEO_SOURCE = "lalala.mp4"

def main():
    model = YOLO("yolov8n-face.pt")
    seen_ids = set()
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    frame_counter = 1

    while cap.isOpened():
        if frame_counter % PROCESS_IMAGES_PER_FRAMES == 0:
            processid = os.fork()

            if processid <= 0: # Child process
                subprocess.run(['python3', 'demographics.py'], check=True)

        else: # Parent process
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            results = model.track(frame, show=True, stream=False, persist=True, verbose=False)

            for result in results:
                ids = result.boxes.id
                bboxes = result.boxes.xyxy

                if frame_counter % CAPTURE_IMAGE_PER_FRAMES == 0:
                    try:
                        mapping = {int(ids[i]): bboxes[i] for i in range(len(ids))}
                        save_faces(frame, frame_counter, mapping, seen_ids)
                    except IndexError:
                        print("No faces detected at the moment")

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def save_faces(frame, frame_counter, mapping, seen_ids):
    """Save each detected face as an image file in the output directory."""
    for obj_id, bbox in mapping.items():
        if obj_id not in seen_ids:
            x1, y1, x2, y2 = map(int, bbox) # Find coordinates of face box

            face = frame[y1:y2, x1:x2] # Crop image of face
            face_filename = f"{OUTPUT_DIR}/id_{obj_id}_FRAME_{frame_counter}.jpg"

            cv2.imwrite(face_filename, face) # Save image

            seen_ids.add(obj_id)
            print(f"New face detected with ID {obj_id}, saved as {face_filename}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
