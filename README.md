# Overdose Detection Project 
This repository contains a continuous monitoring computer vision pipeline designed to detect severe human fall events. It maps raw video pixel data into a high-dimensional spatial manifold, evaluates biomechanical anomalies using a LightGBM decision tree ensemble, and dispatches real-time automated telephony alerts via the Twilio REST API.

* Computer Vision Frontend: Google MediaPipe Tasks API maps video frames into a $\mathbb{R}^{132}$ spatial coordinate space (33 anatomical landmarks $\times$ 4 vectors: $X, Y, Z, \text{Visibility}$).

* Machine Learning Engine: Scikit-Learn maintains stateful preprocessing (MinMaxScaler), while LightGBM optimizes the binary classification decision boundary.

* Telephony Integration: Twilio bridges the local Python execution environment to the global SIP/PSTN network to execute physical phone calls.

* Temporal Smoothing: A double-ended queue calculates moving averages to filter out high-frequency noise and sudden probabilistic spikes.


