

# ArgusEye

**Real-Time Human Detection and Distance Estimation using YOLOv8**

---

## Overview

**ArgusEye** is an open-source computer vision project built on **YOLOv8** for real-time object detection and human segmentation.
It focuses on identifying humans, highlighting them with a red overlay, and estimating their distance from the camera based on object dimensions and focal length.

This module supports both **webcam** and **video file inputs** with high performance and clean visualization.

---

## Key Features

* **Human Detection:** Identifies people using YOLOv8 segmentation models.
* **Red Overlay Highlight:** Applies a semi-transparent red mask on detected humans.
* **Distance Estimation:** Calculates approximate distance between camera and person.
* **Bounding Boxes:** Displays labeled boxes for all detected objects.
* **Dual Input Mode:** Supports live webcam and local video file inputs.
* **Real-Time Inference:** Optimized for smooth, frame-by-frame analysis.

---

## File Structure

| File                  | Description                                                                                   | Contributor                                           |
| --------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `main.py`             | Entry point for the application. Handles model loading, device setup, and source selection.   | [RandomRohit-hub](https://github.com/RandomRohit-hub) |
| `utils_processing.py` | Utility module containing core functions for masking, distance estimation, and label drawing. | [Rajdeep-183](https://github.com/Rajdeep-183)         |

---

## Installation

### Prerequisites

* Python 3.8 or higher
* pip
* A GPU is recommended for optimal performance.

### Setup

```bash
git clone https://github.com/You-Only-Learns-Once/ArgusEye.git
cd ArgusEye
pip install ultralytics opencv-python torch numpy
```

---

## Usage

Run the main script:

```bash
python main.py
```

When prompted:

* Choose **1** for webcam or **2** for a video file.
* Provide the path to the YOLOv8 segmentation model (e.g., `yolov8x-seg.pt`), or press Enter to use the default.

---

## How It Works

1. The YOLOv8 segmentation model detects objects in each frame.
2. For the “person” class, a red overlay mask is applied to the segmented area.
3. The bounding box height is used to estimate distance based on focal length and known human height.
4. The processed frames are displayed in real time with labels and measurements.

---

## Contributors

| Name        | GitHub                                                | Contribution                                                |
| ----------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| **Rohit**   | [RandomRohit-hub](https://github.com/RandomRohit-hub) | Main application logic, model integration, user interaction |
| **Rajdeep** | [Rajdeep-183](https://github.com/Rajdeep-183)         | Utility module, distance and mask computation functions     |

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute it with attribution.

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## Acknowledgements

* [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [PyTorch](https://pytorch.org/)

---

Would you like me to extend this README to include **Friend 3 (detection_runner.py)** next — keeping the same minimal professional style?
