import cv2
import os
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# here we set up mediapipe + provide the process method to turn pixels into landmark coordinates
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

DATA_PATH = "data"
ACTIONS = np.array(['not-seizures', 'seizures'])
SEQUENCE_LENGTH = 30  # number of frames per sequence (approx 1 second of video), industry avg

# feature extraction - we convert a single video frame into a flat array of 132 numerical coordinates in the xyz-plane
def extract_landmarks(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    if results.pose_landmarks:
        landmarks = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        landmarks = np.zeros(33 * 4) #to handle edge cases of no person in the frame 
    return landmarks

# here we loop through our data folder to build the training arrays X and y, standardizing each video into a movement sequence
def load_data():
    sequences, labels = [], []
    for action_idx, action in enumerate(ACTIONS):
        action_path = os.path.join(DATA_PATH, action)
        video_files = [f for f in os.listdir(action_path) if f.endswith(('.mp4', '.avi'))]
        
        print(f"Processing category: {action}")
        for video_file in video_files:
            cap = cv2.VideoCapture(os.path.join(action_path, video_file))
            window = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                landmarks = extract_landmarks(frame)
                window.append(landmarks)
                
                # Once we have a full sequence (e.g., 30 frames), save it and slide the window
                if len(window) == SEQUENCE_LENGTH:
                    sequences.append(window)
                    labels.append(action_idx)
                    window = window[10:] # Overlap sequences to get more training data
            cap.release()
            
    return np.array(sequences), to_categorical(labels).astype(int)
#the lstm model - a long short-term memory model is a special type of neural network that is good w time-sensitive data
# here we define and train the LSTM brain by taking the (30, 132) shaped data and teaching it classification
# main methods we use below to do so: Sequential(), LSTM(), model.fit()
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = Sequential([
        # input_shape = (number of frames, number of landmarks per frame)
        LSTM(64, return_sequences=True, activation='relu', input_shape=(SEQUENCE_LENGTH, 132)),
        LSTM(128, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dropout(0.2), # prevents overfitting to specific videos
        Dense(len(ACTIONS), activation='softmax')
    ])

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    model.save('seizure_model.h5')
    return model


if __name__ == "__main__":
    X, y = load_data()
    model = train_model(X, y)
    print('training complete, model saved as seizure_model.h5')
