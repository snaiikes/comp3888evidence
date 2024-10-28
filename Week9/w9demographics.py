"""
This file analyses faces in the whole directory.
Runs sequentially after the YOLO script is completed.
Results are saved to a JSON file (JSON format).
"""

import os
import json
from deepface import DeepFace

INPUT_FOLDER = 'detected_faces' # Containing face images
OUTPUT_FILE = 'dict.json' # Demographics analysis output file

def analyse_faces(input_folder):
    """
    Analyse faces in the given input folder and return
    demographics data in dictionary.
    """
    demographics = {}

    for filename in os.listdir(input_folder):
        face_image_path = os.path.join(input_folder, filename)

        try:
            id_number = int(filename.split("_")[1])
            analysis = DeepFace.analyze(
                img_path=face_image_path,
                actions=['emotion', 'age', 'gender', 'race']
            )

            data = {
                "Face confidence": analysis[0]['face_confidence'],
                "Dominant emotion": analysis[0]['dominant_emotion'],
                "Age": analysis[0]['age'],
                "Dominant gender": analysis[0]['dominant_gender'],
                "Dominant race": analysis[0]['dominant_race']
            }

            # Update demographics data if face confidence is better than previously stored
            if id_number not in demographics or \
                    analysis[0]['face_confidence'] > demographics[id_number]["Face confidence"]:
                demographics[id_number] = data
                print(f"Updated demographics for ID {id_number} with new data.")

        except ValueError as ve:
            print(f"ValueError for file {filename}: {ve}")

    return demographics

def save_data_to_json(data, file_path):
    """Save demographics data to a JSON file."""
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
        file.write("\n")

def main():
    demographics = analyse_faces(INPUT_FOLDER)
    save_data_to_json(demographics, OUTPUT_FILE)

if __name__ == "__main__":
    main()
