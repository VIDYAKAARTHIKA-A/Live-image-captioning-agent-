from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Load models once
yolo_model = YOLO("yolov8m.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(pil_image):
    # YOLO detection
    cv_img = np.array(pil_image)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    results = yolo_model(cv_img)[0]

    detected_labels = set()
    for box in results.boxes:
        if float(box.conf[0]) > 0.5:
            label = results.names[int(box.cls[0])]
            detected_labels.add(label)

    # BLIP captioning
    inputs = blip_processor(pil_image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True).lower()

    if detected_labels:
        caption += ". Also detected: " + ", ".join(detected_labels) + "."

    return caption
