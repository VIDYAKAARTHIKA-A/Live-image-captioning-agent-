import streamlit as st
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import pyttsx3
import cv2
import time

# Load model and processor
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

# Image captioning function
def generate_caption(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    output_ids = model.generate(**inputs, max_length=64, num_beams=5)
    caption = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return caption.strip()

# Text-to-speech function
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("👓 Live Image Captioning Assistant for the Visually Impaired")
st.markdown("Capture webcam images automatically and get real-time spoken scene descriptions.")

# User can choose duration and frequency
run_time = st.slider("Run duration (seconds)", 10, 120, 30)
interval = st.slider("Caption frequency (seconds)", 1, 10, 2)

if st.button("🎥 Start Auto-Capture"):
    st.info("Auto-captioning started. Press 'Stop' in the terminal to exit.")
    cap = cv2.VideoCapture(0)
    frame_display = st.empty()
    caption_display = st.empty()

    start_time = time.time()

    while time.time() - start_time < run_time:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        # Convert OpenCV image to PIL format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Display current frame
        frame_display.image(pil_image, caption="Live Frame", use_column_width=True)

        # Generate and display caption
        caption = generate_caption(pil_image)
        caption_display.success(f"📝 Caption: {caption}")
        speak(caption)

        time.sleep(interval)

    cap.release()
    st.success("✅ Auto-captioning session completed.")

