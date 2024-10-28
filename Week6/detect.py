import cv2
import logging
import argparse
import warnings
import numpy as np


import torch
import torch.nn.functional as F
from torchvision import transforms
from ultralytics import YOLO


from models import SCRFD
from config import data_config
from utils.helpers import get_model, draw_bbox_gaze

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Gaze estimation inference")
    parser.add_argument("--arch", type=str, default="resnet34", help="Model name, default `resnet18`")
    parser.add_argument(
        "--gaze-weights",
        type=str,
        default="output/gaze360_resnet34_1724339168/best_model.pt",
        help="Path to gaze esimation model weights"
    )
    parser.add_argument(
        "--face-weights",
        type=str,
        default="weights/det_10g.onnx",
        help="Path to face detection model weights"
    )
    parser.add_argument("--view", action="store_true", help="Display the inference results")
    parser.add_argument("--input", type=str, default="assets/in_video.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output file")
    parser.add_argument("--dataset", type=str, default="gaze360", help="Dataset name to get dataset related configs")
    args = parser.parse_args()

    # Override default values based on selected dataset
    if args.dataset in data_config:
        dataset_config = data_config[args.dataset]
        args.bins = dataset_config["bins"]
        args.binwidth = dataset_config["binwidth"]
        args.angle = dataset_config["angle"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Available options: {list(data_config.keys())}")

    return args


def pre_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)
    image_batch = image.unsqueeze(0)
    return image_batch


from ultralytics import YOLO  # for YOLOv8n-person

people = {}

def main(params):

    # THE BELOW HAS BEEN MODIFIED BY ME, EG. CHANGES INCLUDE LAYERING YOLOV8N PERSON ON TOP AND GAZE BOXES.

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx_tensor = torch.arange(params.bins, device=device, dtype=torch.float32)

    try:
        face_detector = SCRFD(model_path=params.face_weights)
        logging.info("Face Detection model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occurred while loading pre-trained weights of face detection model. Exception: {e}")

    try:
        gaze_detector = get_model(params.arch, params.bins, inference_mode=True)
        state_dict = torch.load(params.gaze_weights, map_location=device)
        gaze_detector.load_state_dict(state_dict)
        logging.info("Gaze Estimation model weights loaded.")
    except Exception as e:
        logging.info(f"Exception occurred while loading pre-trained weights of gaze estimation model. Exception: {e}")

    gaze_detector.to(device)
    gaze_detector.eval()

    video_source = params.input
    if video_source.isdigit() or video_source == '0':
        cap = cv2.VideoCapture(int(video_source))
    else:
        cap = cv2.VideoCapture(video_source)

    if params.output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(params.output, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    if not cap.isOpened():
        raise IOError("Cannot open video source")
    
    # Load YOLOv8n-person model for people detection
    model = YOLO("yolov8n-person.pt")

    with torch.no_grad():
        while True:
            success, frame = cap.read()

            if not success:
                logging.info("Failed to obtain frame or EOF")
                break

            # Step 1: Run person detection
            results = model.track(frame, persist=True)
            annotated_frame = results[0].plot()

            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    if box.id not in list(people.keys()):
                        people.update({box.id: [box, 0]})

            # Step 2: Run face detection and gaze estimation
            bboxes, keypoints = face_detector.detect(frame)
            for bbox, keypoint in zip(bboxes, keypoints):
                x_min, y_min, x_max, y_max = map(int, bbox[:4])

                # Extract face region for gaze estimation
                face_image = frame[y_min:y_max, x_min:x_max]
                face_image = pre_process(face_image)
                face_image = face_image.to(device)

                # Perform gaze estimation on the detected face
                pitch, yaw = gaze_detector(face_image)

                pitch_predicted, yaw_predicted = F.softmax(pitch, dim=1), F.softmax(yaw, dim=1)
                pitch_predicted = torch.sum(pitch_predicted * idx_tensor, dim=1) * params.binwidth - params.angle
                yaw_predicted = torch.sum(yaw_predicted * idx_tensor, dim=1) * params.binwidth - params.angle

                # Convert angles from degrees to radians
                pitch_predicted = np.radians(pitch_predicted.cpu())
                yaw_predicted = np.radians(yaw_predicted.cpu())

                # Draw gaze direction on the frame
                gaze_inside = draw_bbox_gaze(annotated_frame, bbox, pitch_predicted, yaw_predicted)

            # Save the output or display it
            if params.output:
                out.write(annotated_frame)

            if params.view:
                cv2.imshow('Demo', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if params.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()

    if not args.view and not args.output:
        raise Exception("At least one of --view or --ouput must be provided.")

    main(args)
