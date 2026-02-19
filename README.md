# Overdose Detection Project 
This project uses MediaPipe to track body movements and an LSTM AI model to identify if a movement looks like a seizure. It works by "watching" video clips and learning the difference between normal activity and rhythmic or rigid seizure patterns.

How to set up your directory?

data (main folder) --> seizures (sub folder) and non-seizures (subfolder)               
main.py               

Install the tools: Run pip install -r requirements.txt in your terminal.
Add your videos: Make sure you have at least a few .mp4 or .avi files in both folders.
Train the AI: Run python main.py.

How does this work generally?
The script uses MediaPipe to turn the person into a "skeleton" of 33 points.

Step 1: It extracts the (x, y, z) coordinates of the joints.

Step 2: It groups 30 frames together to see how those points move over time.

Step 3: The LSTM model analyzes the rhythm. If it sees high-frequency shaking or sudden stiffness, it classifies it as a seizure.
