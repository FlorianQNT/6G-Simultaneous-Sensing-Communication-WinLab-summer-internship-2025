import cv2
import torch
import time
from collections import Counter
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def initialize_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        exit()
    return cap

def load_yolo_model():
    model = YOLO("yolov8l.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model

def initialize_tracker():
    return DeepSort(
        max_age=200,
        n_init=20,
        max_iou_distance=0.7
    )

def detect_and_track(frame, model, tracker, used_class_ids, resize=(800, 600)):
    start_pre = time.time()
    if resize is not None:
        frame = cv2.resize(frame, resize)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    end_pre = time.time()

    start_inf = time.time()
    with torch.no_grad():
        results = model.predict(rgb_frame, verbose=False)
    end_inf = time.time()

    start_post = time.time()

    deepsort_inputs = []
    all_class_ids = []
    class_count = Counter()

    if results:
        detections = results[0].boxes.xywh.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        class_names = model.names

        for i, class_id in enumerate(class_ids):
            all_class_ids.append(class_id)

            if class_id not in used_class_ids:
                continue

            class_name = class_names[class_id]
            class_count[class_name] += 1

            x_center, y_center, width, height = detections[i]
            confidence = confidences[i].item()
            if confidence < 0.6:
                continue

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

    # Log detection summary
    h, w, _ = frame.shape
    shape_str = f"(1, 3, {h}, {w})"
    detection_summary = ', '.join([f"{v} {k}" for k, v in class_count.items()])
    inference_time = (end_inf - start_inf) * 1000
    preprocess_time = (end_pre - start_pre) * 1000
    postprocess_time = (time.time() - start_post) * 1000

    print(f"0: {h}x{w} {detection_summary}, {inference_time:.1f}ms")
    print(f"Speed: {preprocess_time:.1f}ms preprocess, {inference_time:.1f}ms inference, {postprocess_time:.1f}ms postprocess per image at shape {shape_str}")

    return frame, all_class_ids, tracks

def main():
    cap = initialize_camera()
    model = load_yolo_model()
    tracker = initialize_tracker()

    # 0 = person, 56 = chair
    used_class_ids = [0, 56]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        processed_frame, all_class_ids, tracks = detect_and_track(frame, model, tracker, used_class_ids)

        cv2.imshow("YOLOv8 + DeepSORT (Webcam)", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

try:
    main()
except KeyboardInterrupt:
    print("Interrupted. Exiting...")