from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import torch
import cv2
import numpy as np

# Load models once
yolo_model = YOLO("yolov8m.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Define object categories
DANGER_OBJECTS = {"knife", "gun", "fire", "weapon"}
TRAFFIC_OBJECTS = {"car", "truck", "lorry", "bus", "motorcycle", "bicycle", "traffic light", "stop sign"}
OUTDOOR_OBJECTS = TRAFFIC_OBJECTS.union({"road", "tree", "sky", "building", "sidewalk"})  # symbols of outdoor environment

def generate_caption(pil_image):
    # Convert PIL to OpenCV
    cv_img = np.array(pil_image)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

    # Run YOLO
    results = yolo_model(cv_img)[0]
    detected_labels = set()
    person_detected = False
    outdoor_context = False
    danger_detected = False

    for box in results.boxes:
        if float(box.conf[0]) > 0.3:
            label = results.names[int(box.cls[0])]
            detected_labels.add(label)

            # Flags
            if label == "person":
                person_detected = True
            if label in OUTDOOR_OBJECTS:
                outdoor_context = True
            if label.lower() in DANGER_OBJECTS:
                danger_detected = True

    # Generate caption from BLIP
    inputs = blip_processor(pil_image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True).lower()

    # Append label summary
    if detected_labels:
        caption += ". Also detected: " + ", ".join(detected_labels) + "."

    # Append alert messages
    if danger_detected:
        caption += " ⚠️ Danger detected!"

    if person_detected:
        if outdoor_context:
            caption += " 🚦 Traffic detected, be careful."
        else:
            caption += " 👤 Person detected."
    
    print(f"[YOLO Labels]: {detected_labels}")

    print(results.names)





    return caption
