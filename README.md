# AI Driver Monitoring System

A real-time driver monitoring and safety assistance system built using **Flask, OpenCV, MediaPipe, and YOLOv8**. The system continuously analyzes driver behavior and detects unsafe activities such as drowsiness, mobile phone usage, smoking, improper mask usage, and driver distraction.

The objective of this project is to improve road safety through intelligent computer vision-based monitoring and automated alert generation.

---

## Overview

Driver distraction and fatigue are among the leading causes of road accidents worldwide. This project leverages deep learning and computer vision techniques to monitor driver behavior in real time and generate alerts whenever unsafe conditions are detected.

The system supports both:

* **Live webcam-based monitoring**
* **Image-based analysis**

Detected violations can automatically trigger alerts through WhatsApp along with captured evidence images uploaded to cloud storage.

---

## Key Features

### Driver Safety Monitoring

* Eye blink and drowsiness detection using Eye Aspect Ratio (EAR)
* Mobile phone usage detection
* Smoking detection
* Face mask detection
* Hand-on-mouth distraction detection
* Multi-behavior monitoring in a single pipeline

### Real-Time Alert System

* WhatsApp notifications using Twilio
* Automatic violation snapshot capture
* Cloud image hosting through Cloudinary
* Cooldown mechanism to prevent repetitive alerts

### Interactive Web Interface

* Live video stream monitoring
* Image upload and analysis
* Bounding box visualization
* Detection confidence display
* Responsive Flask-based dashboard

---

## System Architecture

```text
Web Interface
      │
      ▼
 Flask Backend
      │
 ┌────┼─────────────────────┐
 │    │    │    │    │
 ▼    ▼    ▼    ▼    ▼
Blink Mask Phone Smoke Hand
Detection Modules
      │
      ▼
 Alert Manager
      │
 ┌───────────────┐
 │ Cloudinary    │
 │ Twilio API    │
 └───────────────┘
```

---

## Detection Modules

### 1. Drowsiness Detection

The system uses MediaPipe Face Mesh landmarks to calculate the Eye Aspect Ratio (EAR).

**Features**

* Blink counting
* Eye closure tracking
* Continuous drowsiness monitoring
* Real-time alert generation

**Default Parameters**

| Parameter            | Value       |
| -------------------- | ----------- |
| EAR Threshold        | 0.25        |
| Drowsiness Threshold | 2.0 seconds |

---

### 2. Face Mask Detection

A custom YOLOv8 model is used to classify:

* With Mask
* Without Mask
* Mask Worn Incorrectly

Additional hand-mouth overlap analysis helps reduce false detections.

---

### 3. Mobile Phone Detection

Phone detection combines:

* Object detection using YOLOv8
* Hand-phone overlap verification
* Person-phone contextual validation

This significantly reduces false positives compared to standard object detection approaches.

---

### 4. Smoking Detection

A custom-trained YOLO model detects smoking-related objects and behaviors in real time.

Features include:

* Continuous monitoring
* Visual feedback
* Alert integration

---

### 5. Hand Detection

Hand tracking is used to improve context awareness across multiple modules including:

* Phone usage detection
* Mask detection
* Hand-on-mouth detection

---

## Technology Stack

### Backend

* Flask
* Python

### Computer Vision

* OpenCV
* MediaPipe
* Ultralytics YOLOv8

### Cloud Services

* Twilio WhatsApp API
* Cloudinary

### Frontend

* HTML
* CSS
* Bootstrap
* JavaScript

---

## Project Structure

```text
Driver_Monitoring_System/
│
├── app.py
├── requirements.txt
├── train.ipynb
│
├── models/
│   ├── best.pt
│   ├── yolov8n.pt
│   ├── handdsa.pt
│   └── detection_module.pt
│
├── templates/
│   └── index.html
│
├── alert_system.py
├── blink_detection.py
├── hand_detection.py
├── mask_detection.py
├── phone_detection.py
├── smoke_detection.py
└── utils.py
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Driver_Monitoring_System.git
cd Driver_Monitoring_System
```

### Create Virtual Environment

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Model Setup

Place the following model files inside the `models/` directory.

| Model               | Purpose                  |
| ------------------- | ------------------------ |
| best.pt             | Face Mask Detection      |
| yolov8n.pt          | General Object Detection |
| handdsa.pt          | Hand Detection           |
| detection_module.pt | Smoking Detection        |

---

## Running the Application

Start the server:

```bash
python app.py
```

Open your browser:

```text
http://localhost:5000
```

### Available Modes

#### Live Monitoring

Uses webcam input for real-time driver analysis.

#### Image Analysis

Upload an image and receive detection results instantly.

---

## Alert Workflow

1. Unsafe behavior detected.
2. Snapshot captured automatically.
3. Image uploaded to Cloudinary.
4. WhatsApp alert sent through Twilio.
5. Cooldown timer activated to avoid duplicate alerts.

---

## Configuration

Important parameters can be adjusted directly in the source code.

```python
EAR_BLINK_THRESHOLD = 0.25
DROWSY_SECONDS_THRESHOLD = 2.0

HAND_ON_MOUTH_THRESHOLD = 0.40

PHONE_CONF = 0.30
PHONE_IOU = 0.45

ALERT_COOLDOWN_SECONDS = 60
DETECT_HEAVY_EVERY_N = 3
```

---

## Future Improvements

* Multi-driver monitoring
* Head pose estimation
* Driver identity verification
* Fatigue prediction using temporal models
* Voice-based warnings
* GPS and telematics integration
* Cloud deployment support
* Docker containerization
* Analytics dashboard

---

## Contributing

Contributions are welcome.

```bash
git checkout -b feature/new-feature
git commit -m "Add new feature"
git push origin feature/new-feature
```

Create a Pull Request describing your changes and proposed improvements.

---

## License

This project is released under the MIT License.

---

## Acknowledgements

* Ultralytics YOLOv8
* MediaPipe
* OpenCV
* Flask
* Twilio
* Cloudinary

---

**Developed to enhance driver awareness, reduce road accidents, and promote safer transportation through computer vision and AI.**

Author - Harsh Dave
