import serial  # Import pyserial for Arduino communication
import time
from ultralytics import YOLO
import cv2
import threading
from fastapi import FastAPI, Request
from fastapi.responses  import HTMLResponse, StreamingResponse
from jinja2 import Template
import uvicorn

# Initialize serial communication with Arduino (Adjust COM port as needed)
arduino = serial.Serial(port='COM8', baudrate=9600, timeout=1)  # Change 'COM3' for your system

app = FastAPI()

# Load the YOLO model
model = YOLO("best.pt")  # Use your custom model

# Global variables to store detection results
detected_labels = []
detection_status = 0  # 0: No detection, 1: Non-burnable, 2: Burnable

# Define burnable and non-burnable categories
burnable_items = {"paper", "biodegradable", "cardboard"}
non_burnable_items = {"plastic", "metal", "garbage", "glass"}

# Open the webcam
webcam_index = 1  # Change if needed
cap = cv2.VideoCapture(webcam_index)

if not cap.isOpened():
    raise RuntimeError(f"Error opening video stream from camera index {webcam_index}")

def classify_detection(labels):
    """Classify detected labels into the defined categories."""
    global detection_status
    if not labels:
        detection_status = 0  # No detection
    else:
        labels_lower = [label.lower() for label in labels]
        print(f"Detected Labels: {labels_lower}")  # Debug: Print detected labels
        burnable_detected = any(label in burnable_items for label in labels_lower)
        non_burnable_detected = any(label in non_burnable_items for label in labels_lower)

        print(f"Burnable Detected: {burnable_detected}")  # Debug
        print(f"Non-burnable Detected: {non_burnable_detected}")  # Debug

        if non_burnable_detected:
            detection_status = 1  # Non-burnable items detected
        elif burnable_detected:
            detection_status = 2  # Burnable items detected
        else:
            detection_status = 0  # Undefined category

    send_to_arduino(detection_status)  # Send status to Arduino

def send_to_arduino(status):
    """Send the detection status to Arduino via serial."""
    try:
        arduino.write(f"{status}\n".encode())  # Send status as a string
        print(f"Sent to Arduino: {status}")  # Debug: Show sent status
    except Exception as e:
        print(f"Error sending to Arduino: {e}")

def run_detection():
    """Continuously capture frames and run YOLO detection."""
    global detected_labels
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame)

        # Extract labels from the results
        detected_labels = [model.names[int(cls)] for box in results[0].boxes for cls in box.cls]

        # Classify the detection
        classify_detection(detected_labels)

        time.sleep(0.1)  # Avoid high CPU usage

# Run detection in a separate thread
thread = threading.Thread(target=run_detection, daemon=True)
thread.start()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main HTML page."""
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return Template(html_content).render()

@app.get("/video_feed")
async def video_feed():
    """Stream video frames with detections."""
    def frame_generator():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            annotated_frame = results[0].plot()  # Annotate frame with detection results

            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/detections")
async def get_detections():
    """Return the detected labels and classification status as JSON."""
    return {
        "detections": detected_labels,
        "status": detection_status  # 0, 1, or 2 based on detection
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
