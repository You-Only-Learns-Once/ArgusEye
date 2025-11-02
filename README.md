# 🧠 **ArgusEye — Real-Time Human Detection & Distance Estimation**

**ArgusEye** is a next-generation, **YOLOv8-powered real-time human detection and segmentation system**.
It not only detects and segments humans from live video or webcam feeds but also **estimates their distance from the camera** — combining **AI vision** with **practical spatial intelligence**.

---

## 🚀 **Unique Features**

✅ **YOLOv8 Segmentation Integration**
Performs precise, real-time object and human segmentation with dynamic bounding boxes and masks.

✅ **Distance Estimation Engine**
Calculates real-world human distance using bounding box geometry and calibrated focal length.

✅ **Smart Visual Overlay System**
Adds visually rich annotations — glowing masks, bounding boxes, and live distance lines — for easy understanding.

✅ **High Performance Optimization**

* GPU acceleration via CUDA (auto device detection).
* Torch backend tuned for maximum frame rate and accuracy.

✅ **Flexible Input Options**
Seamlessly switch between **webcam** or **MP4 video** input from the terminal.

✅ **4K, 60 FPS Support**
Handles high-resolution streams smoothly with minimal latency.

---

## 🧩 **Tech Stack**

* **Language:** Python
* **Libraries:** `ultralytics`, `torch`, `opencv-python`, `numpy`
* **Model:** YOLOv8 Segmentation (`yolov8n-seg.pt` or custom weights)

---

## 🧪 **How It Works**

1. **Load YOLOv8 Model** – Default or custom model weights.
2. **Capture Frames** – From webcam or video file.
3. **Detect & Segment** – Identify humans and generate segmentation masks.
4. **Estimate Distance** – Using bounding box scaling and focal calibration.
5. **Render Output** – With annotated masks, bounding boxes, and distance text.

---

## ⚙️ **Usage**

```bash
# Clone the repository
git clone https://github.com/You-Only-Learns-Once/ArgusEye.git
cd ArgusEye

# Install dependencies
pip install -r requirements.txt

# Run the program
python arguseye.py
```

Select input when prompted:

```
1. Use Webcam
2. Use MP4 Video File
```

Press **Q** anytime to quit.

---

## 👥 **Team ArgusEye**

| Developer      | GitHub Profile                                        | Key Contribution                                                       |
| -------------- | ----------------------------------------------------- | ---------------------------------------------------------------------- |
| 🧠 **Rohit**   | [RandomRohit-hub](https://github.com/RandomRohit-hub) | Core vision pipeline and YOLOv8 segmentation integration               |
| ⚙️ **Rajdeep** | [Rajdeep-183](https://github.com/Rajdeep-183)         | Distance estimation algorithm and mask overlay system                  |
| 💡 **Srijan**  | [Srijanprasad](https://github.com/Srijanprasad)       | System integration, performance tuning, and final ArgusEye unification |

> 🔗 *The final build merges all three modules into one unified ArgusEye system — an example of true collaborative AI engineering.*

---

## 📜 **License**

Licensed under the **MIT License** — you’re free to use, modify, and distribute with proper credit.

---

## 🌟 **Future Enhancements**

* 3D-aware distance visualization
* Multi-object distance mapping (humans, vehicles, objects)
* Depth-based AR/VR spatial integration

---

