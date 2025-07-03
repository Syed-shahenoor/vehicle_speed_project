import cv2
import math
import csv
from ultralytics import YOLO
import time
import platform
import os

# OS beep detection
IS_WINDOWS = platform.system() == "Windows"

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load video
cap = cv2.VideoCapture("highway.mp4")
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Calibration constants
PIXEL_DISTANCE = 250  # Pixels between 2 real points (manual)
REAL_DISTANCE = 10    # Real distance in meters
PIXEL_TO_METER_RATIO = REAL_DISTANCE / PIXEL_DISTANCE
print(f"Using calibrated PIXEL_TO_METER_RATIO: {PIXEL_TO_METER_RATIO:.5f}")

SPEED_LIMIT_KMPH = 60

# Frame info
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
DETECTION_LINE_Y = int(frame_height * 0.90)  # Logic line (speed check)
VISIBLE_LINE_Y = 420  # 👁️ Visible line (drawn higher for visibility)

# CSV setup
csv_file = open("vehicle_speeds.csv", mode="w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Frame", "Vehicle ID", "Vehicle", "Speed (km/h)", "Overspeed"])

# Tracking
prev_centroids = {}
frame_count = 0
beeped_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]
    current_centroids = {}

    # Draw visible purple line (visual only)
    cv2.line(frame, (0, VISIBLE_LINE_Y), (frame.shape[1], VISIBLE_LINE_Y), (255, 0, 255), 2)
    cv2.putText(frame, "Speed Detection Line", (10, VISIBLE_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

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

            # Draw green box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # ✅ Calculate speed if tracked before
            if track_id in prev_centroids:
                prev_cx, prev_cy = prev_centroids[track_id]
                pixel_distance = math.hypot(cx - prev_cx, cy - prev_cy)
                meters_moved = pixel_distance * PIXEL_TO_METER_RATIO
                fps = cap.get(cv2.CAP_PROP_FPS)
                speed_mps = meters_moved * fps
                speed_kmph = speed_mps * 3.6

                # ✅ Show speed always
                cv2.putText(frame, f"{int(speed_kmph)} km/h", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # ✅ Overspeed check
                is_overspeed = speed_kmph > SPEED_LIMIT_KMPH

                # ✅ Log + Beep only when crossing detection line
                if cy >= DETECTION_LINE_Y:
                    if is_overspeed:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        if track_id not in beeped_ids:
                            if IS_WINDOWS:
                                import winsound
                                winsound.Beep(1000, 200)
                            else:
                                os.system('play -nq -t alsa synth 0.2 sine 1000')
                            beeped_ids.add(track_id)

                    # ✅ Log in CSV (fixed to log proper Yes/No)
                    csv_writer.writerow([
                        frame_count,
                        track_id,
                        model.names[cls],
                        round(speed_kmph, 2),
                        "Yes" if is_overspeed else "No"
                    ])

            # Vehicle label
            label = model.names[cls]
            cv2.putText(frame, label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Show output
    cv2.imshow("Vehicle Speed Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    prev_centroids = current_centroids

# Cleanup
csv_file.close()
cap.release()
cv2.destroyAllWindows()
