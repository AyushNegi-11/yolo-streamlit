import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("YOLOv8 Object Detection Demo")
st.write("Upload an image below and watch magic happen!")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption=' Uploaded Image', use_container_width=True)
    st.write("Detecting objects...")
    results = model.predict(image)
    res_plotted = results[0].plot()
    st.image(res_plotted, caption=' Detection Result', use_container_width=True)
