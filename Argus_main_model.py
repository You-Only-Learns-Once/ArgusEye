import cv2
import torch
import numpy as np
import os
import time
from ultralytics import YOLO



# Confi

KNOWN_PERSON_HEIGHT_CM = 170   
FOCAL_LENGTH = 600             
CONF_THRESHOLD = 0.5           
TARGET_FPS = 60                



torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')



def apply_human_mask(frame, mask, color=(0, 0, 255), alpha=0.6):
    """Overlay a colored mask on detected human regions."""
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    overlay = np.full_like(frame, color, dtype=np.uint8)
    mask_area = mask[..., None] > 0.5
    return np.where(mask_area, cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0), frame)


def estimate_distance(bbox_height):
    """Estimate distance to object based on bounding box height."""
    if bbox_height == 0:
        return 0
    distance_cm = (KNOWN_PERSON_HEIGHT_CM * FOCAL_LENGTH) / bbox_height
    return round(distance_cm / 100, 2)  # Convert to meters


def draw_box_and_label(frame, label, conf, x1, y1, x2, y2, color=(0, 255, 0)):
    """Draw bounding box and label with confidence."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)



def main():
    print("=== YOLOv8 Segmentation and Distance Estimation ===")
    print("1. Use Webcam")
    print("2. Use MP4 Video File")
    choice = input("Enter choice (1/2): ").strip()

    # Input source selection
    if choice == "1":
        source = 0
    elif choice == "2":
        source = input("Enter video file path: ").strip()
        if not os.path.exists(source):
            print("Error: File not found.")
            return
    else:
        print("Invalid input. Exiting.")
        return

    # Model load
    model_path = input("Enter YOLO model path (default: yolov8n-seg.pt): ").strip() or "yolov8n-seg.pt"
    if not os.path.exists(model_path):
        print("Error: Model file not found.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(model_path).to(device)
    model.fuse()
    print("Model loaded successfully.")

    # Video setup
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    window_name = "YOLOv8 Segmentation Output"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.moveWindow(window_name, 100, 100)

    prev_time = 0
    print("Press 'Q' to exit.")

    # Frame processing 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        small_frame = cv2.resize(frame, (1280, 720))
        results = model.predict(small_frame, conf=CONF_THRESHOLD, device=device, verbose=False)

        annotated_frame = cv2.resize(frame, (1280, 720))
        frame_center = (annotated_frame.shape[1] // 2, annotated_frame.shape[0] // 2)
        cv2.circle(annotated_frame, frame_center, 5, (255, 0, 0), -1)

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

                # Handle person distance
                if label.lower() == "person":
                    if masks is not None:
                        mask = masks.data[i].cpu().numpy()
                        annotated_frame = apply_human_mask(annotated_frame, mask)
                    distance = estimate_distance(bbox_height)
                    person_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    cv2.line(annotated_frame, frame_center, person_center, (0, 0, 255), 2)
                    mid_x = (frame_center[0] + person_center[0]) // 2
                    mid_y = (frame_center[1] + person_center[1]) // 2
                    cv2.putText(annotated_frame, f"{distance} m", (mid_x, mid_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # FPS calc
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)

        cv2.imshow(window_name, annotated_frame)

        # Quit 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Processing complete.")


# -------------------------------
# Entry 
# -------------------------------
if __name__ == "__main__":
    main()
