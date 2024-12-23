import threading
import base64
import requests
from io import BytesIO
import random

import torch
import torchvision.transforms as transforms
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import streamlit as st
import numpy as np
import cv2
import uvicorn

# ---------------- FASTAPI BACKEND ---------------- #

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageInput(BaseModel):
    file: str


try:
    num_classes = 4
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load('mobilenet_model.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

LABELS = ["Healthy", "Grade 1", "Grade 2", "Grade 3"]


@app.post("/predict")
async def predict(image_data: ImageInput):
    try:
        img_bytes = base64.b64decode(image_data.file)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

        percentages = (probabilities * 100).tolist()
        return {"probabilities": percentages, "labels": LABELS}
    except Exception as e:
        return {"error": str(e)}


# ---------------- STREAMLIT MULTI-PAGE APP ---------------- #

def process_image(image):
    image = image.convert("RGB").resize((224, 224))
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    b, g, r = cv2.split(img_np)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b, g, r = clahe.apply(b), clahe.apply(g), clahe.apply(r)
    processed_img = cv2.merge((b, g, r))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

    return image, Image.fromarray(processed_img)


def predict_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    try:
        response = requests.post("http://127.0.0.1:5000/predict", json={"file": img_data})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


# Initialize session state for OTP and mobile number
if "otp" not in st.session_state:
    st.session_state.otp = None
if "mobile_number" not in st.session_state:
    st.session_state.mobile_number = None

def generate_otp():
    """Generate a random 6-digit OTP."""
    return random.randint(100000, 999999)


# Main application logic
def main():
    if "page" not in st.session_state:
        st.session_state.page = "Registration"

    # Navigation function
    def navigate_to(page_name):
        st.session_state.page = page_name

        # Page 1: Enter Name and Email

    if st.session_state.page == "Registration":
        st.title("User Registration")
        st.text_input("Full Name:", key="full_name")
        st.text_input("Email:", key="email")

        if st.button("Next", key="registration_next_button"):
            if st.session_state.full_name and st.session_state.email:
                navigate_to("Personal Details")
            else:
                st.error("Please enter your full name and email.")

    # Page 2: Enter OTP
    elif st.session_state.page == "Enter OTP":
        st.title("Enter OTP")
        st.write(f"An OTP has been sent to your mobile number: {st.session_state.mobile_number}")
        st.write(f"Your OTP is: **{st.session_state.otp}**")  # Display OTP on the screen
        otp_input = st.text_input("Enter the OTP you received:")

        if st.button("Verify OTP", key="verify_otp_button"):
            if otp_input == str(st.session_state.otp):
                st.success("OTP verified successfully!")
                navigate_to("Personal Details")
            else:
                st.error("Invalid OTP. Please try again.")

    # Page 3: Personal Details
    elif st.session_state.page == "Personal Details":
        st.title("Personal Details")
        st.text_input("Name:", key="name")
        st.date_input("Date of Birth:", key="dob")
        st.number_input("Age:", min_value=1, max_value=120, step=1, key="age")
        st.selectbox("Gender:", ["Male", "Female", "Other"], key="gender")
        st.text_input("Address:", key="address")
        st.number_input("Pincode:", key="pincode")
        if st.button("Next", key="personal_details_next_button"):
            navigate_to("Medical Questionnaire")

    # Page 4: Medical Questions
    elif st.session_state.page == "Medical Questionnaire":
        st.title("Medical Questionnaire")
        high_bp = st.radio("Do you have high blood pressure?", options=["Yes", "No"], key="high_bp")
        diabetes = st.radio("Do you have diabetes?", options=["Yes", "No"], key="diabetes")

        if diabetes == "Yes":
            diabetes_years = st.number_input("How many years have you had diabetes?", min_value=0, step=1,
                                             key="diabetes_years")
        else:
            ulcers = st.radio("Do you have ulcers on your foot?", options=["Yes", "No"], key="ulcers")

        other_conditions = st.text_area("Please list any other current medical conditions:", key="medical_conditions")
        medications = st.text_area("Are you taking any medications?", key="medications")

        if st.button("Next", key="medical_questionnaire_next_button"):
            navigate_to("Image Upload")

    # Page 5: Image Upload and Prediction
    elif st.session_state.page == "Image Upload":
        st.title("AI-Based Oral Health Screening")
        uploaded_file = st.file_uploader("Upload an image (JPG, JPEG, PNG):", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            original_image, processed_image = process_image(image)
            st.image(original_image, caption="Original Image", use_column_width=True)
            st.image(processed_image, caption="Processed Image", use_column_width=True)

            if st.button("Predict", key="predict_button"):
                predictions = predict_image(processed_image)

                if predictions and "probabilities" in predictions:
                    st.subheader("Predictions:")
                    for label, prob in zip(predictions["labels"], predictions["probabilities"]):
                        st.write(f"{label}: {float(prob):.2f}%")
                else:
                    st.error("Failed to get predictions. Please try again.")


def start_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=5000)


if __name__ == "__main__":
    threading.Thread(target=start_fastapi, daemon=True).start()
    main()
