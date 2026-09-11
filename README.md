# AI-Traffic-Sign-Recongnition
CNN-based traffic sign recognition system that classifies road signs and recommends autonomous vehicle actions using TensorFlow and Streamlit

## Features
- CNN-based traffic sign classification
- Recognizes Stop, Yield, No Entry, Speed Limit, and Turn Right signs
- Displays prediction confidence
- Recommendds an autonomous vehicle action for each prediction
- Interactive image upload using Streamlit

## Technologies
- Python
- TensorFlow/ Keras
- Streamlit
- NumPy
- Pillow
- GTSRB Database

## Supported Traffic Signs

| Traffic Sign | Vehicle Action|
|- — -|- - -|
| Stop | Stop Vehicle |
| Yield| Slow Down and Yield |
|No Entry| Do not Enter |
|Speed Limit| Adjust Vehicle Speed |
|Turn Right| Turn Right|

## How It Works

1. User uploads a traffic sign image.
2. The image is resized and preprocessed.
3. The trained CNN analyzes the image.
4. The model predicts the traffic sign class.
5. The application displays the prediction and confidence score.
6. The prediction is mapped to an autonomous vehicle action.
