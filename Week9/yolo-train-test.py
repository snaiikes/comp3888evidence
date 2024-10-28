import os
import glob
from ultralytics import YOLO
import torch

# PROVIDED BY LONG

# Load the model (change here if you want another model)
model = YOLO("/root/yolo/yolov10x.pt")
# Define the base directory for YOLO training runs
base_dir = "runs/detect"
# Find the latest run directory by looking for the most recently modified directory
run_dirs = sorted(glob.glob(os.path.join(base_dir, "train*")), key=os.path.getmtime)

# Set training args based on what you need https://docs.ultralytics.com/modes/train/#train-settings
epochs = 100
imgsz = 640
batch = 16 * torch.cuda.device_count() # Must be a multiple of GPU numbers !!!
device = [i for i in range(torch.cuda.device_count())] # Use all available gpus
data = "/root/yolo/coco8.yaml" # Change to your desired yaml

if run_dirs:
    latest_run_dir = run_dirs[-1]
    # Check if a checkpoint exists in the latest run directory
    last_checkpoint_path = os.path.join(latest_run_dir, "weights", "last.pt")
    print(f">>> FROM SCRIPT >>> Latest run directory: {latest_run_dir}")
    print(f">>> FROM SCRIPT >>> Last checkpoint: {last_checkpoint_path}")

    # Last.pt exists
    if os.path.exists(last_checkpoint_path):
        model2 = YOLO(last_checkpoint_path)
        start_epoch = model2.ckpt['epoch']
        # Finished training all epochs, start a new one
        if (start_epoch < 0):
            # Train model again. Use different train parameters if needed.
            print(f">>> FROM SCRIPT >>> FINISHED EPOCHS, NEW RUN!!!")
            results = model.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz, device=device, resume=False)
        else:
            # Resume training
            print(f">>> FROM SCRIPT >>> Interrupted, resuming training")
            results = model2.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz, device=device, resume=True)
    
    else:
        print(">>> FROM SCRIPT >>> No last.pt found. Starting a new training session...")
        results = model.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz, device=device, resume=False)

else:
    print(">>> FROM SCRIPT >>> No previous run directories found. Starting a new training session...")
    results = model.train(data=data, batch=batch, epochs=epochs, imgsz=imgsz, device=device, resume=False)