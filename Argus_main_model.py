
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
