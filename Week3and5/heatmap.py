"""
This file gets a basic heatmap and saves the video file.
"""

import cv2
from ultralytics import YOLO, solutions

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Output video will be "heatmap_output.mp4"
    video_writer = cv2.VideoWriter("heatmap_output.mp4",
                                   cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    # Initialise heatmap object
    heatmap_obj = solutions.Heatmap(
        colormap=cv2.COLORMAP_PARULA,
        view_img=True,
        shape="circle",
        names=model.names,
    )

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break  # Exit loop if no frames left/end of video

        tracks = model.track(frame, persist=True, show=False)
        heatmap_frame = heatmap_obj.generate_heatmap(frame, tracks)
        video_writer.write(heatmap_frame)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
