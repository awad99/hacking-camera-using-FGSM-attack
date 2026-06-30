# Interactive FGSM Camera Evasion Demo (Hacking Surveillance)

An interactive demonstration showcasing how the **Fast Gradient Sign Method (FGSM)** adversarial attack can be utilized to bypass object detection algorithms (YOLOv11) in real-time surveillance camera feeds, making targets (specifically cars) evade detection.

---

## 📸 Demo Features
- **Real-Time Side-by-Side Comparison**:
  - **Left Screen**: Original camera feed showing successful object detection (surveillance camera tracking cars normally).
  - **Right Screen**: Attacked camera feed showing detection evasion (cars become invisible to the AI model due to adversarial noise).
- **Interactive Controls**: Dynamically increase or decrease the attack strength (epsilon $\epsilon$) in real-time to observe the model's behavior.
- **Evaded Indicator**: Displays a success watermark (`DETECTION EVADED`) when a previously detected object successfully vanishes from the bounding box predictions.
- **Automatic Simulation Video**: Auto-generates a panning camera path from a static image if no video file is provided.

---

## 🛠️ How it Works
The application applies the mathematical principles of the **Fast Gradient Sign Method (FGSM)** to create adversarial perturbations targeting the convolution layers of YOLO:

$$\eta = \epsilon \cdot \text{sign}(\nabla_X L(\theta, X, y))$$

To disrupt the detection specifically on targeted objects (e.g., cars):
1. The frame is fed into PyTorch.
2. Gradients are computed at the regions where objects are located.
3. High-frequency adversarial grid patterns and noise are injected into these regions.
4. The output frame remains visually similar to the human eye, but completely confuses YOLO's filters, evading bounding box generation.

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Install the required dependencies:
```bash
pip install torch torchvision ultralytics opencv-python numpy
```

### 2. Running the Demo
Start the visualization tool:
```bash
python main.py
```

### 3. Keyboard Controls
* Press **`+`** or **`=`** to increase the adversarial noise ($\epsilon$ Epsilon) and strengthen the evasion.
* Press **`-`** or **`_`** to decrease the noise.
* Press **`Q`** or **`q`** to quit the application.

---

## 📂 Project Structure
* `main.py` - Core PyTorch & OpenCV execution script containing FGSM logic and interactive UI.
* `car_image.jpg` - Base image used for synthetic video generation.
* `video.mp4` - Generated test video containing camera motion.
* `yolo11n.pt` - PyTorch model weights used for detection.

---

## ⚠️ Disclaimer
This repository is created strictly for **educational and research purposes** to demonstrate vulnerabilities in computer vision models. Do not use these methods against production systems without proper authorization.
