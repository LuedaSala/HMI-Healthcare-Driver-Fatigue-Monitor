# HMI Healthcare Driver Fatigue Monitor

A non-invasive, real-time driver fatigue and microsleep detection system built using **Google MediaPipe Face Mesh** and **OpenCV**. Developed as a project report requirement for the course *Human-Machine Interaction in Healthcare* (Academic Year 2025/2026).

## Key Features
* **Contactless Biometric Tracking:** Uses standard 720p RGB webcam footage to map 478 3D facial landmarks without intrusive wearable sensors.
* **Dual-Threshold Validation System:** Uses a spatial threshold (EAR < 0.20) and a temporal confirmation window (20 consecutive frames / ~660ms) to successfully filter out natural blinks and prevent alarm fatigue.
* **Asynchronous Actuation Alerts:** Instantly triggers a high-visibility visual overlay and a synchronized 1000 Hz auditory tone upon tracking a microsleep event.
* **Local Edge Computing:** Runs entirely on the host CPU with zero cloud-computing dependence, protecting driver privacy and eliminating network transmission latency.

## Repository Structure
* `fatigue_monitor.py`: The core Python implementation script including the main execution loop, real-time calculation nodes, and live telemetry tracking graph export.
* `face_landmarker.task`: The quantized MediaPipe machine learning model bundle binary optimized for low-power edge computing execution.

## Local Execution Instructions
1. Clone this repository or download `fatigue_monitor.py`.
2. Install the necessary baseline dependencies:
   ```bash
   pip install opencv-python mediapipe matplotlib
