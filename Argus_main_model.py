
import os
import torch
from ultralytics import YOLO
from detection_runner import run_detection

def main():
    print(" YOLOv8 Human Highlighter + Distance Estimator")
    print("-----------------------------------------------")
    print("Select Source:")
    print("1️ Webcam")
    print("2️ MP4 Video File")
    choice = input("Enter your choice (1/2): ").strip()

    if choice == "1":
        source = 0
    elif choice == "2":
        source = input("Enter full path of video file (e.g., video.mp4): ").strip()
        if not os.path.exists(source):
            print(f" Error: File '{source}' not found.")
            return
    else:
        print(" Invalid choice.")
        return

    model_path = input("Enter YOLOv8 model path (default: yolov8n-seg.pt): ").strip()
    if model_path == "":
        model_path = "yolov8n-seg.pt"

    if not os.path.exists(model_path):
        print(f" Error: Model file '{model_path}' not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Using device: {device.upper()}")

    # model loading
    model = YOLO(model_path)
    model.to(device)
    print(f" Model '{model_path}' loaded successfully.")

    run_detection(model, source, device)

if __name__ == "__main__":
    main()




# packages
import cv2
import numpy as np

# Config
KNOWN_HEIGHT = 170  
FOCAL_LENGTH = 600  

def apply_human_mask(frame, mask, color=(0, 0, 255), alpha=0.6):
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    overlay = np.zeros_like(frame, dtype=np.uint8)
    overlay[:, :] = color
    mask_area = mask[..., None] > 0.5
    return np.where(mask_area, cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0), frame)

def estimate_distance(bbox_height):
    if bbox_height == 0:
        return 0
    distance_cm = (KNOWN_HEIGHT * FOCAL_LENGTH) / bbox_height
    return round(distance_cm / 100, 2)

def draw_box_and_label(frame, label, conf, x1, y1, x2, y2, color=(0, 255, 0)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


import cv2
from utils_processing import apply_human_mask, estimate_distance, draw_box_and_label

def run_detection(model, source, device, conf_thresh=0.5):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f" Error: Cannot open video source '{source}'")
        return

    print(" Running segmentation (press 'q' to quit)")
    frame_center = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print(" End of stream or no frame captured.")
            break

        results = model.predict(frame, conf=conf_thresh, device=device, verbose=False)
        annotated_frame = frame.copy()

        if frame_center is None:
            frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)

        cv2.circle(annotated_frame, frame_center, 5, (255, 0, 0), -1)
        cv2.putText(annotated_frame, "Camera", (frame_center[0] - 30, frame_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        for result in results:
            boxes = result.boxes
            masks = result.masks

            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox_height = y2 - y1

               
                draw_box_and_label(annotated_frame, label, conf, x1, y1, x2, y2)

             
                if label.lower() == "person":
                    if masks is not None:
                        mask = masks.data[i].cpu().numpy()
                        annotated_frame = apply_human_mask(annotated_frame, mask)

                    distance = estimate_distance(bbox_height)
                    person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.line(annotated_frame, frame_center, person_center, (0, 0, 255), 2)
                    mid_x = (frame_center[0] + person_center[0]) // 2
                    mid_y = (frame_center[1] + person_center[1]) // 2
                    cv2.putText(annotated_frame, f"{distance}m",
                                (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 0), 2)

        cv2.imshow("YOLOv8 - Human Distance & Highlight", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Quitting...")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Done! Full human highlighting + distance display complete.")


