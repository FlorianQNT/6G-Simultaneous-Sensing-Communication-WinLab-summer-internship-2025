import cv2
#import numpy as np
from ultralytics import YOLO
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        exit()
    return cap

def load_yolo_model():
    model = YOLO("yolov8l.pt")  # You can switch to yolov8n.pt or yolov8s.ptd
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def initialize_tracker():
    return DeepSort(
        max_age=200,
        n_init=20,
        max_iou_distance=0.7
    )

def detect_and_track(frame, model, tracker, used_class_ids):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)

    deepsort_inputs = []
    all_class_ids = []

    if results:
        detections = results[0].boxes.xywh.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = model.names

        for i, class_id in enumerate(class_ids):
            all_class_ids.append(class_id)

            if class_id not in used_class_ids:
                continue

            x_center, y_center, width, height = detections[i]
            confidence = confidences[i]
            class_name = class_names[class_id]

            # Convert to bounding box corners
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Draw bounding box and label
            color = (0, 255, 0)
            label = f"{class_name} ({confidence * 100:.1f}%)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Add to DeepSORT
            deepsort_inputs.append(([x1, y1, x2, y2], confidence, class_id))

    # Update DeepSORT and draw track IDs
    tracks = tracker.update_tracks(deepsort_inputs, frame=frame)
    for track in tracks:
        if not track.is_confirmed() or track.time_since_update > 150:
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)

        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    return frame, all_class_ids

def main():
    cap = initialize_camera()
    model = load_yolo_model()
    tracker = initialize_tracker()

    # Define class IDs you want to keep: 0 = person, 56 = chair
    # https://gist.github.com/rcland12/dc48e1963268ff98c8b2c4543e7a9be8
    used_class_ids = [0, 56]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        processed_frame, all_class_ids = detect_and_track(frame, model, tracker, used_class_ids)

        cv2.imshow("YOLOv8 + DeepSORT (Webcam)", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

main()
