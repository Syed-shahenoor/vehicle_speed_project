import cv2
import math
import csv
from ultralytics import YOLO
import time

# Constants
SPEED_LIMIT_KMPH = 60  # Overspeed threshold
PIXEL_TO_METER_RATIO = 0.05  # Approx. distance a vehicle moves per pixel

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("highway.mp4")
if not cap.isOpened():
    print("Error opening video file")
    exit()

# CSV setup
csv_file = open("vehicle_speeds.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Vehicle", "Speed (km/h)", "Overspeed"])

# Variables for tracking
prev_centroids = {}
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
    current_centroids = {}

    if results.boxes.id is not None:
        ids = results.boxes.id.cpu().numpy().astype(int)
        classes = results.boxes.cls.cpu().numpy().astype(int)
        boxes = results.boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            cls = classes[i]
            track_id = ids[i]
            x1, y1, x2, y2 = map(int, box)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_centroids[track_id] = (cx, cy)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calculate speed if we have seen this ID before
            if track_id in prev_centroids:
                prev_cx, prev_cy = prev_centroids[track_id]
                pixel_distance = math.hypot(cx - prev_cx, cy - prev_cy)
                meters_moved = pixel_distance * PIXEL_TO_METER_RATIO
                fps = cap.get(cv2.CAP_PROP_FPS)
                speed_mps = meters_moved * fps
                speed_kmph = speed_mps * 3.6

                # Draw speed on frame
                cv2.putText(frame, f"{int(speed_kmph)} km/h", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Overspeeding check
                if speed_kmph > SPEED_LIMIT_KMPH:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

                # Log to CSV
                csv_writer.writerow([
                    frame_count,
                    model.names[cls],
                    round(speed_kmph, 2),
                    "Yes" if speed_kmph > SPEED_LIMIT_KMPH else "No"
                ])

            # Label with class name
            label = model.names[cls]
            cv2.putText(frame, label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("Vehicle Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_centroids = current_centroids

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()
