import cv2
from ultralytics import YOLO
import torch

# Check if GPU is available and set device accordingly
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the YOLOv8 model
model_path = 'best.pt'  # Update this path if needed
model = YOLO(model_path)
model.to(device)

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam successfully opened. Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run inference on the current frame
    results = model(frame)

    # Annotate frame with predictions
    # results[0].plot() draws bounding boxes on a copy of the frame.
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: release webcam and close windows
cap.release()
cv2.destroyAllWindows()