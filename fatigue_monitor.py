# -*- coding: utf-8 -*-
"""
Created on Fri May  8 12:33:24 2026

@author: dimit
"""

import cv2
import math
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import matplotlib.pyplot as plt  # NEW: For generating the IEEE report graph
import winsound                  # NEW: For the audio actuator (Windows only)


model_path = 'face_landmarker.task'
if not os.path.exists(model_path):
    print("Downloading the new MediaPipe AI Model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")


def calculate_ear(eye_points):
    v1 = math.dist(eye_points[1], eye_points[5])
    v2 = math.dist(eye_points[2], eye_points[4])
    h = math.dist(eye_points[0], eye_points[3])
    return (v1 + v2) / (2.0 * h)


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.20      
FRAME_LIMIT = 20          
closed_frame_count = 0    


ear_data_log = [] 


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = frame.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            
            left_eye_pts = [(int(face_landmarks[i].x * w_img), 
                             int(face_landmarks[i].y * h_img)) for i in LEFT_EYE]
            right_eye_pts = [(int(face_landmarks[i].x * w_img), 
                              int(face_landmarks[i].y * h_img)) for i in RIGHT_EYE]

            average_ear = (calculate_ear(left_eye_pts) + calculate_ear(right_eye_pts)) / 2.0

            # --- NEW: SAVE DATA FOR THE GRAPH ---
            ear_data_log.append(average_ear)

            # State Logic
            if average_ear < EAR_THRESHOLD:
                closed_frame_count += 1
            else:
                closed_frame_count = 0 

            cv2.putText(frame, f"EAR: {average_ear:.2f}", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if closed_frame_count >= FRAME_LIMIT:
                cv2.rectangle(frame, (0,0), (w_img, h_img), (0, 0, 255), 10)
                cv2.putText(frame, "FATIGUE ALERT!", (w_img//4, h_img//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 6)
                
                
                winsound.Beep(1000, 200)

            for pt in left_eye_pts + right_eye_pts:
                cv2.circle(frame, pt, 2, (0, 255, 0), -1)

    cv2.imshow("Driver Fatigue Monitor", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()


if len(ear_data_log) > 0:
    print("Generating EAR data graph for your report...")
    plt.figure(figsize=(10, 5))
    
  
    plt.plot(ear_data_log, label="Real-time EAR", color='blue', linewidth=1.5)
    
    
    plt.axhline(y=EAR_THRESHOLD, color='red', linestyle='--', label=f"Fatigue Threshold ({EAR_THRESHOLD})")
    
   
    plt.title("Eye Aspect Ratio (EAR) Telemetry During Simulation", fontsize=14)
    plt.xlabel("Elapsed Frames (Time)", fontsize=12)
    plt.ylabel("EAR Value", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
   
    plt.show()