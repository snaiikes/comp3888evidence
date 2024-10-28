import cv2
from deepface import DeepFace

# Use this for the simplest version of DeepFace. It's literally one line.
# DeepFace.stream(db_path="/Users/connieyeung/Desktop/comp3888evidence/Week6") 

# 0 for webcam, filename otherwise
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Could not read frame.")
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        # Display the emotion detected. Can be modified to detect more.
        emotion = result[0]['dominant_emotion']
        cv2.putText(frame, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    except Exception as e:
        print("Error analyzing frame:", e)

    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
