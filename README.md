# 🚦 AI Traffic Sign Recognition for Autonomous Vehicles

An AI-powered traffic sign recognition system that uses deep learning to classify traffic signs from uploaded images and recommend an appropriate autonomous vehicle action.

## 🌐 Live Demo

Try the deployed application:

https://ai-traffic-sign-recongnition-wxn6jhqwkzeuyypotugty9.streamlit.app

## 🎯 Project Overview

Traffic sign recognition is an important component of autonomous driving systems.

This project demonstrates an end-to-end machine learning pipeline that:

- Processes traffic sign image data
- Trains an image classification model
- Recognizes traffic signs from uploaded images
- Displays prediction confidence
- Maps predictions to autonomous vehicle actions
- Provides an interactive web interface using Streamlit

## 🚘 Supported Traffic Signs

The current model recognizes five traffic sign categories:

| Traffic Sign | Autonomous Vehicle Action |
|---|---|
| Stop | Stop Vehicle |
| Yield | Slow Down and Yield |
| No Entry | Do Not Enter |
| Turn Right | Turn Right |
| Speed Limit | Adjust Vehicle Speed |

## 🧠 Machine Learning Pipeline

1. Prepare and organize traffic sign image data
2. Preprocess and resize images
3. Split data into training and validation sets
4. Train the image classification model
5. Evaluate model performance on validation data
6. Save the trained model
7. Load the model in the Streamlit application
8. Run inference on user-uploaded images
9. Convert the predicted traffic sign into a vehicle action

## 🛠️ Technologies Used

- Python
- PyTorch
- Computer Vision
- Pillow
- NumPy
- Streamlit
- Git & GitHub
- GitHub Codespaces
- Streamlit Community Cloud

## 📊 Model Results

During training, the model achieved approximately **99% validation accuracy** on the project's validation dataset.

> Validation accuracy reflects performance on the prepared dataset and does not guarantee equivalent performance on unseen real-world traffic images.

## 💻 Application

Users can upload a JPG or PNG traffic sign image.

The application displays:

- Uploaded traffic sign
- Predicted traffic sign class
- Model confidence
- Recommended autonomous vehicle action

Example:

**Prediction:** Yield  
**Confidence:** 99.31%  
**Vehicle Action:** Slow Down and Yield

## 📁 Project Structure

AI-Traffic-Sign-Recognition/
├── app.py
├── train_model.py
├── download_dataset.py
├── traffic_sign_model.pth
├── requirements.txt
└── README.md

## 🚀 Future Improvements

- Expand recognition to additional traffic sign categories
- Train with a larger and more diverse dataset
- Improve performance on real-world images
- Add confidence-based handling for uncertain predictions
- Evaluate performance using precision, recall, F1-score, and a confusion matrix
- Explore real-time traffic sign recognition from video

## 👨‍💻 Author

**Shashank Aravapalli**

B.S. Robotics  
University of California, Riverside
