import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
from torchvision import transforms


# --------------------------------------------------
# Page setup
# --------------------------------------------------

st.set_page_config(
    page_title="AI Traffic Sign Recognition",
    page_icon="🚦"
)

st.title("🚦 AI Traffic Sign Recognition")

st.write(
    "Upload a traffic sign image and the AI model will "
    "identify the sign and recommend an autonomous vehicle action."
)


# --------------------------------------------------
# CNN architecture
# Must match train_model.py
# --------------------------------------------------

class TrafficSignCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

@st.cache_resource
def load_model():

    checkpoint = torch.load(
        "traffic_sign_model.pth",
        map_location="cpu",
        weights_only=False
    )

    class_names = checkpoint["class_names"]

    model = TrafficSignCNN(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, class_names


model, class_names = load_model()


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# --------------------------------------------------
# Autonomous vehicle actions
# --------------------------------------------------

ACTIONS = {
    "stop": "STOP VEHICLE",
    "yield": "SLOW DOWN AND YIELD",
    "no_entry": "DO NOT ENTER",
    "speed_limit": "ADJUST VEHICLE SPEED",
    "turn_right": "TURN RIGHT"
}


# --------------------------------------------------
# Image upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a traffic sign image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Traffic Sign"
    )

    # Preprocess image
    image_tensor = transform(image).unsqueeze(0)

    # Run AI prediction
    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )

        confidence, predicted_index = torch.max(
            probabilities,
            1
        )

    predicted_class = class_names[
        predicted_index.item()
    ]

    confidence_percent = (
        confidence.item() * 100
    )

    vehicle_action = ACTIONS.get(
        predicted_class,
        "NO ACTION AVAILABLE"
    )

    # Display results
    st.success("Image analyzed successfully!")

    st.subheader("AI Prediction")

    st.write(
        f"**Traffic Sign:** "
        f"{predicted_class.replace('_', ' ').title()}"
    )

    st.write(
        f"**Confidence:** "
        f"{confidence_percent:.2f}%"
    )

    st.subheader(
        "Autonomous Vehicle Action"
    )

    st.info(vehicle_action)