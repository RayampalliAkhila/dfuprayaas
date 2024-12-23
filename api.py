
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import base64
import streamlit as st
from io import BytesIO
import requests
import numpy as np
import cv2
import threading
import torch
torch.set_num_threads(1)


# Define the FastAPI app
app = FastAPI()

# Input schema for FastAPI
class ImageInput(BaseModel):
    file: str  # Base64 encoded image string

# Load the PyTorch model
model = torch.load('mobilenet_model.pth', map_location=torch.device('cpu'))
#model.eval()  # Set model to evaluation mode

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

# Labels for prediction
labels = ["Healthy", "Grade 1", "Grade 2", "Grade 3"]

@app.post("/predict")
async def predict(image_data: ImageInput):
    # Decode base64 image data to bytes
    img_bytes = base64.b64decode(image_data.file)

    # Open the image using PIL
    img_stream = io.BytesIO(img_bytes)
    img = Image.open(img_stream).convert('RGB')  # Ensure RGB format

    # Preprocess the image
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        pred = model(img_tensor)
        probabilities = torch.nn.functional.softmax(pred[0], dim=0)  # Apply softmax for probabilities

    # Convert probabilities to percentages
    pred_percent = probabilities * 100
    results = pred_percent.tolist()  # Convert to list

    return {"probabilities": results, "labels": labels}

# Streamlit app
def process_image(image):
    # Convert to RGB and resize
    image = image.convert("RGB")
    image = image.resize((224, 224))

    # Convert to NumPy array and apply CLAHE
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(image_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_b, clahe_g, clahe_r = clahe.apply(b), clahe.apply(g), clahe.apply(r)
    image_clahe = cv2.merge((clahe_b, clahe_g, clahe_r))
    image_clahe = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)

    return image, image_clahe

def predict_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Send image to the FastAPI backend
    response = requests.post("http://127.0.0.1:5000/predict", json={"file": img_data})
    response_data = response.json()

    # Extract probabilities and labels
    probabilities = response_data["probabilities"]
    labels = response_data["labels"]

    return list(zip(labels, probabilities))

def streamlit_app():
    st.title("AI-Based Oral Health Screening")
    st.markdown("""
        <h3 style="text-align: center;">Upload an image for screening</h3>
    """, unsafe_allow_html=True)

    # Image upload
    uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        original_image, processed_image = process_image(image)

        # Display the original and processed images
        st.image(original_image, caption="Original Image", use_column_width=True)
        st.image(processed_image, caption="Processed Image", use_column_width=True)

        # Predict the condition
        predictions = predict_image(Image.fromarray(processed_image))

        # Display predictions
        st.subheader("Predictions:")
        for label, prob in predictions:
            st.write(f"{label}: {float(prob):.2f}%")

# Start FastAPI server in a separate thread
def start_fastapi():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)

if __name__ == "__main__":
    threading.Thread(target=start_fastapi, daemon=True).start()  # Start FastAPI in a thread
    streamlit_app()  # Start Streamlit app
