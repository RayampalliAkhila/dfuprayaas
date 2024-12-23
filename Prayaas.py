import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import io
import base64
import requests
from io import BytesIO
import threading
from datetime import datetime

# --- FastAPI Backend ---
app = FastAPI()

# Enable CORS for communication between Streamlit and FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body structure
class ImageInput(BaseModel):
    file: str

# Load the model
from torchvision.models import mobilenet_v2

num_classes = 4
model = mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
model.load_state_dict(torch.load('mobilenet_model.pth', map_location=torch.device('cpu')))
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(image_data: ImageInput):
    img_bytes = base64.b64decode(image_data.file)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    return (probabilities * 100).tolist()

def run_fastapi():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)

fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
fastapi_thread.start()

# --- Streamlit Frontend ---
def process_image(image):
    """Process the image for better contrast using CLAHE."""
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b, g, r = clahe.apply(b), clahe.apply(g), clahe.apply(r)
    processed_image = cv2.merge((b, g, r))
    return image, cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

def predict_image(image):
    """Send the image to the FastAPI backend and get predictions."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    response = requests.post("http://127.0.0.1:5000/predict", json={"file": img_data})
    if response.status_code == 200:
        return response.json()
    st.error(f"API error {response.status_code}: {response.text}")
    return None

def streamlit_app():
    st.title("AI-Based Oral Health Screening")
    st.markdown("<h3 style='text-align: center;'>Enter Your Details and Upload an Image</h3>", unsafe_allow_html=True)

    # User Information Inputs
    name = st.text_input("Name", placeholder="Enter your name")
    date_of_birth = st.date_input("Date of Birth", min_value=datetime(1900, 1, 1))
    gender = st.radio("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, step=1, value=25)

    # Image Upload Section
    uploaded_file = st.file_uploader("Upload an Image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file and name and date_of_birth and age and gender:
        st.subheader("User Information")
        st.write(f"**Name:** {name}")
        st.write(f"**Date of Birth:** {date_of_birth}")
        st.write(f"**Age:** {age}")
        st.write(f"**Gender:** {gender}")

        image = Image.open(uploaded_file)
        original_image, processed_image = process_image(image)

        st.subheader("Uploaded Images")
        st.image(original_image, caption="Original Image", use_column_width=True)
        st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Perform Prediction
        st.subheader("Prediction Results")
        predictions = predict_image(Image.fromarray(processed_image))
        labels = ["Grade 1", "Grade 2", "Grade 3", "Healthy"]
        if predictions:
            for label, prob in zip(labels, predictions):
                st.write(f"{label}: {prob:.2f}%")
    else:
        st.info("Please fill all fields and upload an image.")

if __name__ == "__main__":
    streamlit_app()
