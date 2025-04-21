# genai_fashion_matcher.py

# This code is intended to be run in a local or hosted Python environment with required dependencies installed.
# Streamlit is used for the front-end. Ensure the following packages are installed in your environment:
# pip install streamlit torch torchvision opencv-python pillow git+https://github.com/openai/CLIP.git ultralytics

import streamlit as st
import tempfile
import os
import torch
from PIL import Image
import clip
import cv2
from ultralytics import YOLO

st.set_page_config(page_title="GenAI Outfit Matcher", layout="wide")

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")

# Dummy fashion categories
fashion_labels = [
    "red dress", "blue jeans", "white t-shirt", "black saree", "green kurta",
    "formal suit", "denim jacket", "lehenga", "sportswear", "casual wear"
]

@st.cache_data
def encode_labels(labels):
    text_inputs = clip.tokenize(labels).to(device)
    with torch.no_grad():
        return clip_model.encode_text(text_inputs)

text_features = encode_labels(fashion_labels)

# Main UI
st.title("👗 GenAI-Powered Influencer Outfit Matcher")

uploaded_file = st.file_uploader("Upload an image or short video clip (mp4)", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    if suffix == ".mp4":
        st.video(temp_path)
        cap = cv2.VideoCapture(temp_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error("Failed to extract frame from video.")
            st.stop()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    else:
        image = Image.open(temp_path).convert("RGB")

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO
    st.subheader("🔍 Detecting outfits with YOLOv8")
    results = yolo_model(image)
    detections = results[0].boxes.xyxy.cpu().numpy()

    matched_items = []

    for i, (x1, y1, x2, y2, *_) in enumerate(detections):
        crop = image.crop((x1, y1, x2, y2))
        img_input = preprocess(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            image_feature = clip_model.encode_image(img_input)
            probs = (image_feature @ text_features.T).softmax(dim=-1).squeeze()
            top_idx = torch.argmax(probs).item()
            matched_label = fashion_labels[top_idx]
            matched_items.append((crop, matched_label, probs[top_idx].item()))

    if matched_items:
        st.subheader("🎯 Matched Fashion Items")
        for idx, (crop_img, label, score) in enumerate(matched_items):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(crop_img, caption=f"Detected Outfit #{idx+1}", width=150)
            with col2:
                st.markdown(f"**Prediction:** {label}  ")
                st.markdown(f"**Confidence:** {score:.2f}")
                st.markdown(f"🔗 Example: [Shop {label}](https://www.myntra.com/search/{label.replace(' ', '%20')})")
    else:
        st.warning("No fashion items detected. Try another image/video.")
