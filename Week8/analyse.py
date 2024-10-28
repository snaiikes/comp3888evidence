"""
This file runs DeepFace on all files in a folder. Result is printed to terminal.
"""

import os
from deepface import DeepFace

DIRECTORY = 'detected_faces'

def analyse_face_images(directory):
    """Analyse face images in the specified directory."""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Processing file: {filename}")

            try:
                analysis = DeepFace.analyze(
                    img_path=file_path,
                    actions=['age', 'gender', 'emotion', 'race']
                )

                # Just prints to terminal
                print(f"Results for {filename}:")
                print(f"  Age: {analysis[0]['age']}")
                print(f"  Gender: {analysis[0]['gender']}")
                print(f"  Emotion: {analysis[0]['dominant_emotion']}")
                print(f"  Race: {analysis[0]['dominant_race']}")
                print("\n")

            except (TypeError, ValueError) as e:
                print(f"Error analysing {filename}: {e}")

def main():
    analyse_face_images(DIRECTORY)

if __name__ == "__main__":
    main()
