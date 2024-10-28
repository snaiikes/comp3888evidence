#!/bin/bash

# Remember to chmod +x first!!!

# Run yolo.py and wait for it to finish
python3 getids.py

# Check if yolo.py completed successfully
if [ $? -eq 0 ]; then
    echo "YOLO detection finished"
    python3 w9demographics.py

else
    echo "Not finished"
fi
