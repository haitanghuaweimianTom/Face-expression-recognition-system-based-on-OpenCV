# Emotion Recognition System - Functional Documentation

> 🌐 Language Switch: [中文版本](README.md)

## 1. System Overview
This program is a real-time emotion recognition system built with a visual interface using `tkinter`, combined with OpenCV face detection and a deep learning model (TensorFlow/Keras) **trained locally on the developer's personal computer** using a public facial expression dataset. The system can access the computer's camera to capture human faces in real time, identify their emotion categories, display the probability distribution of each emotion type, and present the recognition results intuitively through a graphical interface.

## 2. Core Features
| Function Module | Detailed Description |
|-----------------|----------------------|
| Visual GUI Interface | Build an elegant graphical interface with tkinter, including title, camera display area, recognition result area, and exit button |
| Real-time Camera Capture | Access the local camera (default index 0), collect video frames in real time, and preprocess (horizontal flip + fixed-size scaling) |
| Face Detection | Load OpenCV's official face detection classifier (haarcascade_frontalface_default.xml) to accurately detect faces and mark with blue rectangles |
| Multi-category Emotion Recognition | Based on a **self-trained Keras model (best_emotion_model.h5)** from public datasets, identify 7 core emotions: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise |
| Probability Quantification Display | Calculate and display the probability ratio of each emotion (2 decimal places), and mark the "Current Emotion" with the highest probability |
| Elegant Exit Mechanism | Click the "Exit" button to release camera resources and close the GUI window to avoid resource occupation |
| Cross-environment Compatibility | Compatible with Python development environment and packaged executable files (dist directory), automatically identify the running environment and adapt file paths |

## 3. Technology Stack
| Category | Technologies |
|----------|--------------|
| Interface Framework | tkinter (Python built-in GUI library) |
| Computer Vision | OpenCV (camera operation, face detection, image preprocessing) |
| Image Processing | PIL (convert OpenCV images to tkinter display format) |
| Deep Learning | TensorFlow/Keras (load self-trained model, emotion prediction) |
| Numerical Calculation | NumPy (data format conversion, tensor processing) |
| System Adaptation | sys/os (identify running environment, file path processing) |

## 4. Model Training Background
### 4.1 Training Environment
- Hardware: Personal computer (CPU/GPU based on actual training configuration)
- Software: Python 3.7~3.10, TensorFlow 2.x, NumPy, Pandas, OpenCV
- Training Dataset: Public facial expression dataset (e.g., FER-2013, CK+, JAFFE)

### 4.2 Training Process
1. Dataset preprocessing: Clean the public dataset, normalize face images to 48×48 grayscale images, and divide into training/validation/test sets
2. Model construction: Build a convolutional neural network (CNN) suitable for emotion recognition (including convolutional layers, pooling layers, fully connected layers)
3. Model training: Train the model on a personal computer, adjust hyperparameters (learning rate, batch size, epochs) to optimize accuracy
4. Model optimization: Use techniques such as dropout, batch normalization to prevent overfitting, and save the best-performing model as `best_emotion_model.h5`

## 5. Running Environment Requirements
### 5.1 Python Version
Recommended: Python 3.7~3.10 (compatible with mainstream TensorFlow/Keras versions)

### 5.2 Dependencies Installation
```bash
pip install opencv-python pillow tensorflow numpy
