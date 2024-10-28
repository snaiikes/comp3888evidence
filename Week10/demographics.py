"""
This file analyses faces in a specified directory at a moment in time using DeepFace. 
Results are saved to a JSON file (JSON format).
"""

import os
import shutil
import json
from datetime import datetime
from deepface import DeepFace

INPUT_FOLDER = 'detected_faces' # Containing face images
OUTPUT_FILE = 'dict.json' # Demographics analysis output file

def create_analysis_directory(base_folder):
    """
    Create unique directory name (assume folder will be refreshed every day, 
    so time must be unique). This folder will be used for DeepFace analysis.
    """
    now = datetime.now()
    current_time = now.strftime('%H.%M.%S')
    new_dir = os.path.join(base_folder, current_time)
    os.mkdir(new_dir)
    return new_dir

def move_files_to_directory(source_folder, destination_folder):
    """Capture all files at current moment in time and move to source_folder."""
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg"):
            shutil.move(
                os.path.join(source_folder, filename),
                os.path.join(destination_folder, filename)
            )

def analyse_faces(directory):
    """
    Analyse faces in the given input folder and return
    demographics data in dictionary.
    """
    demographics = {}
    for filename in os.listdir(directory):
        face_image_path = os.path.join(directory, filename)
        try:
            id_number = int(filename.split("_")[1])
            analysis = DeepFace.analyze(img_path=face_image_path,
                                        actions=['emotion', 'age', 'gender', 'race'])

            data = {
                "Face confidence": analysis[0]['face_confidence'],
                "Dominant emotion": analysis[0]['dominant_emotion'],
                "Age": analysis[0]['age'],
                "Dominant gender": analysis[0]['dominant_gender'],
                "Dominant race": analysis[0]['dominant_race']
            }

            if (id_number not in demographics or
                    analysis[0]['face_confidence'] > demographics[id_number]["Face confidence"]):
                demographics[id_number] = data

        except (ValueError, IndexError, KeyError) as e:
            print(f"Error analysing face {filename}: {e}")
    return demographics

def save_data_to_json(data, file_path):
    """Appends data to a JSON file. Note JSON is continuously updated, not rewritten."""
    with open(file_path, 'a', encoding='utf-8') as file:
        json.dump(data, file, indent=4)
        file.write("\n")

def main():
    new_dir = create_analysis_directory(INPUT_FOLDER)
    move_files_to_directory(INPUT_FOLDER, new_dir)
    demographics = analyse_faces(new_dir)
    save_data_to_json(demographics, OUTPUT_FILE)

if __name__ == "__main__":
    main()
