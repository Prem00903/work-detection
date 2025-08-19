import cv2
import mediapipe as mp
import face_recognition
import pickle
import time
import numpy as np
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO


RTSP_URL = 0 
ENCODINGS_FILE = 'encodings.pickle'
EXCEL_LOG_FILE = 'activity_log.xlsx'


CONFIDENCE_THRESHOLD = 0.25
MATCH_TOLERANCE = 0.65
IOU_THRESHOLD = 0.3
STATIC_THRESHOLD_SECONDS = 7.0 
MOVEMENT_THRESHOLD = 2.0
GAZE_GRACE_MARGIN = 0.15
SCREEN_BUFFER_SECONDS = 0


print("[INFO] Loading models...")
general_model = YOLO('yolov8l.pt')
custom_model_path =r'C:\Users\Lenovo\Desktop\work_d\runs\detect\laptop_detector_v12\weights\best.pt'


try:
    custom_device_model = YOLO(custom_model_path)

except Exception as e:
    print(f"[ERROR] Could not load custom model from {custom_model_path}")
    exit()

try:
    data = pickle.loads(open(ENCODINGS_FILE, "rb").read())

except FileNotFoundError:
    print(f"[ERROR] Encodings file not found. Please run the face training script.")
    exit()


mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- STATE MANAGEMENT & HELPER FUNCTIONS ---
tracked_people = {}
next_person_id = 0
tracked_screens = []

def calculate_iou(boxA, boxB):
    pass

def is_gazing_at_screen(landmarks, screen_boxes, frame_shape):
    pass

def is_posture_working(landmarks):
    pass 

def log_status_change(person_id, name, status, start_time, end_time):
    pass

# --- MAIN APPLICATION LOOP ---
cap = cv2.VideoCapture(RTSP_URL)

try:
    while True:
        ret, frame = cap.read()
        if not ret: 
            if RTSP_URL == 0: break
            time.sleep(2); continue

        # --- STEP 1: Detections using Dual Models ---
        person_boxes = []
        all_screen_detections = []

        # A. Get person and screen detections from the general model
        general_results = general_model(frame, stream=True, verbose=False, classes=[0, 63, 67])
        for r in general_results:
            for box in r.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    if class_id == 0:
                        person_boxes.append((x1, y1, x2, y2))
                    else:
                        all_screen_detections.append((x1, y1, x2, y2))
        
        # B. Get additional screen detections from your custom model
        custom_results = custom_device_model(frame, stream=True, verbose=False)
        for r in custom_results:
            for box in r.boxes:
                if box.conf[0] > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    all_screen_detections.append((x1, y1, x2, y2))
        
        # C. Merge overlapping screen boxes using Non-Maximum Suppression (NMS)
        if all_screen_detections:
            boxes_np = np.array(all_screen_detections)
            # Create dummy scores as NMSBoxes requires scores
            scores_np = np.ones(len(boxes_np)) 
            # Convert (x1,y1,x2,y2) to (x,y,w,h) for NMSBoxes
            boxes_for_nms = np.array([[x1, y1, x2-x1, y2-y1] for x1,y1,x2,y2 in boxes_np])
            
            indices = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), scores_np.tolist(), score_threshold=CONFIDENCE_THRESHOLD, nms_threshold=0.5)
            current_screen_detections = [all_screen_detections[i] for i in indices]
        else:
            current_screen_detections = []

        # (The rest of the script for Pose, Buffering, Tracking, and Status Logic continues here)
        # ...
        
        cv2.imshow("Employee Activity Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    print("[INFO] Shutting down...")
    cap.release()
    cv2.destroyAllWindows()