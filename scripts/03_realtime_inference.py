import os
import cv2
import joblib
import numpy as np
import pandas as pd
import mediapipe as mp
from collections import deque
from twilio.rest import Client

PROJECT_ROOT = "/Users/ishanibakshi/fall-detection-system"
MODEL_PATH = os.path.join(PROJECT_ROOT, "models/lightgbm_fall_model.pkl")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models/feature_scaler.pkl")

# for now, its just a single video, will be live-cam
LIVE_VIDEO_SOURCE = os.path.join(PROJECT_ROOT, "data/raw_videos/Falls/20240912_101331.mp4")

TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "your_account_sid")
TWILIO_AUTH_TOKEN = os.environ.get("TWILIO_AUTH_TOKEN", "your_auth_token")

TWILIO_PHONE_NUMBER = "+18556304753" 
EMERGENCY_CONTACT = "+17327628925"   #only added my # to account so far 

CONFIDENCE_THRESHOLD = 0.85
TEMPORAL_WINDOW_SIZE = 10 
COOLDOWN_FRAMES = 150     
# Telephony Subsystem
def dispatch_emergency_call(confidence, frame_number):
    print(f"\n[NETWORK] Initiating SIP/PSTN bridge to {EMERGENCY_CONTACT}...")
    if TWILIO_ACCOUNT_SID == "your_account_sid":
        print("[WARN] Twilio credentials default. Bypassing actual network request.")
        return

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    twiml_instructions = f"""
    <Response>
        <Say voice="Polly.Joanna" language="en-US">
            Emergency alert. A severe fall event has been visually detected at frame {frame_number} 
            with {confidence:.1%} sustained confidence. Immediate medical dispatch is required.
        </Say>
    </Response>
    """
    try:
        call = client.calls.create(
            twiml=twiml_instructions,
            to=EMERGENCY_CONTACT,
            from_=TWILIO_PHONE_NUMBER
        )
        print(f"[SUCCESS] Call queued. Twilio SID: {call.sid}")
    except Exception as e:
        print(f"[FATAL] Telephony dispatch failed: {e}")
# Inference Loop
def run_surveillance():
    print("Loading binary artifacts into memory...")
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Serialized artifacts missing. Run 02_train_model.py first.")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    cap = cv2.VideoCapture(LIVE_VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {LIVE_VIDEO_SOURCE}")
        
    print(f"Monitoring video feed: {LIVE_VIDEO_SOURCE}")
    
    # Extract native framerate to normalize playback speed
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0
    frame_delay = int(1000 / fps)
    
    feature_cols = [f"v{i}" for i in range(132)]
    probability_buffer = deque(maxlen=TEMPORAL_WINDOW_SIZE)
    
    # State tracking variables
    frame_count = 0
    frames_since_last_alert = COOLDOWN_FRAMES 

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\nEnd of video stream reached.")
            break
            
        frame_count += 1
        frames_since_last_alert += 1
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        current_prob = 0.0
        
        if results.pose_landmarks:
            landmarks = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility] 
                                  for lmk in results.pose_landmarks.landmark]).flatten()
            
            # Format and scale coordinates
            raw_features_df = pd.DataFrame([landmarks], columns=feature_cols)
            scaled_features = scaler.transform(raw_features_df)
            
            # Execute forward pass
            current_prob = model.predict_proba(scaled_features)[0][1]
            
            # Draw spatial topology
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        # Update temporal buffer
        probability_buffer.append(current_prob)
        moving_avg = sum(probability_buffer) / len(probability_buffer) if probability_buffer else 0.0
        
        status_color = (0, 0, 255) if moving_avg > CONFIDENCE_THRESHOLD else (0, 255, 0)
        cv2.putText(frame, f"Fall Prob (Avg): {moving_avg:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
        
        # Display Cooldown Status
        if frames_since_last_alert < COOLDOWN_FRAMES:
            cv2.putText(frame, f"COOLDOWN: {COOLDOWN_FRAMES - frames_since_last_alert} frames", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        
        cv2.imshow("Surveillance Feed", frame)
        
        # Evaluate decision boundary and cooldown state
        if moving_avg > CONFIDENCE_THRESHOLD and frame_count > TEMPORAL_WINDOW_SIZE:
            if frames_since_last_alert >= COOLDOWN_FRAMES:
                print(f"\n[ALERT] Temporal threshold breached at frame {frame_count}. Sustained Probability: {moving_avg:.4f}")
                dispatch_emergency_call(moving_avg, frame_count)
                
                # Reset state machine
                frames_since_last_alert = 0
                probability_buffer.clear()
            
        # Enforce real-time playback speed
        if cv2.waitKey(frame_delay) & 0xFF == ord('q'):
            print("\nUser terminated surveillance.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_surveillance()