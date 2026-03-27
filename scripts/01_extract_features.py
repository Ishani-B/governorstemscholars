import os
import cv2
import pandas as pd
import numpy as np
import mediapipe as mp

# ==========================================
# Configuration
# ==========================================
PROJECT_ROOT = "/Users/ishanibakshi/fall-detection-system"
FALLS_DIR = os.path.join(PROJECT_ROOT, "data/raw_videos/Falls")
ADL_DIR = os.path.join(PROJECT_ROOT, "data/raw_videos/ADL")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data/processed/extracted_features.csv")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_landmarks_from_video(video_path, label):
    """
    Reads a video frame-by-frame, extracts the 132-dimensional 
    pose vector, and returns a list of dictionaries.
    """
    cap = cv2.VideoCapture(video_path)
    frame_data = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # MediaPipe requires RGB color space
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            # Flatten the 33 landmarks (x, y, z, visibility) into a single 132-element array
            landmarks = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] 
                                  for lmk in results.pose_landmarks.landmark]).flatten()
            
            # Construct a dictionary mapping feature names (v0, v1 ... v131) to their values
            row = {f"v{i}": val for i, val in enumerate(landmarks)}
            row['anomaly'] = label
            frame_data.append(row)
            
    cap.release()
    return frame_data

def process_directory(directory_path, label):
    """
    Iterates through all .mp4 files in a directory and compiles the extracted data.
    """
    print(f"Scanning directory: {directory_path} for label {label}")
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}. Ensure your data is placed correctly.")
        
    all_data = []
    valid_extensions = ('.mp4', '.avi', '.mov')
    
    files = [f for f in os.listdir(directory_path) if f.lower().endswith(valid_extensions)]
    print(f"Found {len(files)} video files.")
    
    for idx, filename in enumerate(files):
        print(f"Processing {idx+1}/{len(files)}: {filename}")
        video_path = os.path.join(directory_path, filename)
        video_features = extract_landmarks_from_video(video_path, label)
        all_data.extend(video_features)
        
    return all_data

if __name__ == "__main__":
    print("Initiating computer vision feature extraction...")
    
    # Process ADL (Normal Movement -> Label 0)
    adl_data = process_directory(ADL_DIR, label=0)
    
    # Process Falls (Anomaly -> Label 1)
    falls_data = process_directory(FALLS_DIR, label=1)
    
    # Concatenate and save to a tabular format
    combined_data = adl_data + falls_data
    df = pd.DataFrame(combined_data)
    
    print(f"Extraction complete. Total frames processed: {len(df)}")
    
    # Drop rows where MediaPipe failed to find a person
    df.dropna(inplace=True)
    print(f"Total valid frames after dropping nulls: {len(df)}")
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset serialized to disk: {OUTPUT_CSV}")
    print("Phase 1 complete. You may now proceed to model training.")