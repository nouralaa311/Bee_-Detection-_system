import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Queen & Worker Bee Detection", layout="centered")

# 2. Custom UI Styling (Dark Theme)
st.markdown(
    """
    <style>
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    h1 {
        color: #FFD700 !important;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
    }
    p {
        text-align: center;
        color: #B0B0B0;
        font-size: 1.1rem;
    }
    .stFileUploader {
        background-color: #1E1E1E;
        border: 2px dashed #FFD700 !important;
        border-radius: 10px;
        padding: 10px;
    }
    .legend-card {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #FFD700;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>🐝 Queen & Worker Bee Detector</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a beehive image. Streamlit communicates with FastAPI backend to detect bees.</p>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="legend-card">
        <strong> Bounding Box Color Guide:</strong><br>
        🟥 <span style="color:#FF3333; font-weight:bold;">Red Box</span>: Represents the <b>Queen Bee</b>👑<br>
        🟨 <span style="color:#FFCC00; font-weight:bold;">Yellow Box</span>: Represents the <b>Worker Bees</b>🐝
    </div>
    """, 
    unsafe_allow_html=True
)

# FastAPI Server URL
FASTAPI_URL = "http://127.0.0.1:8000/predict"

uploaded_file = st.file_uploader("Choose a beehive image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)
    
    if st.button(" Run AI Detection via FastAPI"):
        with st.spinner("Sending image to FastAPI server and analyzing..."):
            try:
                # 1. تحويل الصورة إلى Bytes لإرسالها عبر الـ API
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format=image.format if image.format else 'JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                
                files = {"file": (uploaded_file.name, img_byte_arr, uploaded_file.type)}
                
                # 2. إرسال الطلب لـ FastAPI
                response = requests.post(FASTAPI_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"Detection completed! Total bees detected: {data['total_detected']} 🎉")
                    
                    # 3. رسم المربعات بناءً على البيانات القادمة من الـ API JSON
                    draw = ImageDraw.Draw(image)
                    
                    for detection in data["detections"]:
                        box = detection["box"] # [xmin, ymin, xmax, ymax]
                        class_name = detection["class_name"]
                        
                        # تحديد اللون بناءً على الكلاس القادم من السيرفر
                        color = "#FF3333" if class_name == "Queen-Bee" else "#FFCC00"
                        
                        # رسم المربع بسمك خفيف (width=3)
                        draw.rectangle(box, outline=color, width=3)
                    
                    st.subheader(" Detection Results")
                    st.image(image, use_container_width=True)
                else:
                    st.error("Error: Failed to get response from FastAPI server.")
                    
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Make sure your FastAPI server is running on http://127.0.0.1:8000")