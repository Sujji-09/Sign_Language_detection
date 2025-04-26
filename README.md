# Sign_Language_detection
## 📜 Code Explanation

This project consists of two major scripts: `train_model.py` for data collection and model training, and `real_time_detection.py` for live sign language recognition using a webcam feed.

In `train_model.py`, the process begins by importing essential libraries such as OpenCV, MediaPipe, TensorFlow, and supporting modules like NumPy and scikit-learn. The goal of this script is to collect video sequences for three specific American Sign Language (ASL) gestures: "please," "thankyou," and "sorry." For each gesture, 30 short video clips (sequences) are recorded using the webcam. Each sequence consists of 30 frames. MediaPipe Holistic is used to detect landmarks for the face, body pose, and both hands in each frame. These landmarks are then extracted as structured keypoint arrays, which represent the gesture numerically.

The extracted keypoints are saved into organized folders under the `MP_Data/` directory for each action. Once the data collection is completed, all saved keypoints are loaded into memory, labeled, and split into training and testing sets. An LSTM (Long Short-Term Memory) neural network is then built using TensorFlow Keras. The model consists of stacked LSTM layers followed by dense layers, designed to learn temporal patterns in the sequences of keypoints. After training the model on the collected gesture data, it is saved as `asl_action_model.h5`, which can be later loaded for real-time prediction.

The second script, `real_time_detection.py`, uses the trained LSTM model to perform live gesture recognition. This script starts by loading the `asl_action_model.h5` model and initializing the webcam feed using OpenCV. As the webcam captures frames, MediaPipe Holistic continuously detects landmarks on the body, face, and hands. For every frame, keypoints are extracted and added into a rolling sequence buffer of the last 30 frames.

When the sequence buffer is full, the model predicts the most probable action (gesture) from the collected sequence. If the prediction is confident enough (above a set threshold), the corresponding gesture name is displayed on top of the webcam feed in a colored rectangle. The real-time detection loop continues until the user presses the `q` key, which cleanly closes the webcam feed and exits the application.

Through this project, we build a complete pipeline that captures ASL gestures, trains a custom deep learning model, and deploys the model in real-time to detect user gestures live. The combination of MediaPipe for landmark detection and LSTM for temporal sequence modeling enables smooth and responsive gesture recognition without needing large datasets or heavy computation.

