from flask import Flask, render_template, request, redirect, url_for
import cv2
from PIL import Image
import torch
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3
import time
import re

app = Flask(__name__)

# Load Models
yolo_model = YOLO("yolov8m.pt")  # better than yolov8n
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 150)

def generate_caption_and_objects(frame):
    # YOLO Detection
    results = yolo_model(frame)[0]
    traffic_candidates, danger_labels, general_labels = [], [], []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]
        if conf < 0.5:
            continue

        if label in ["person", "car", "bus", "truck", "motorbike", "bicycle"]:
            traffic_candidates.append(label)
        elif label in ["knife", "gun", "fire"]:
            danger_labels.append(label)
        elif label in ["hose", "cell phone", "remote", "tv monitor", "charger", "mouse"]:
            continue
        else:
            general_labels.append(label)

    # BLIP Captioning
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    inputs = blip_processor(pil_img, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True).lower()

    # Outdoor context
    outdoor_keywords = ["road", "street", "traffic", "intersection", "vehicle", "crosswalk", "sidewalk", "outside"]
    is_outdoor = any(word in caption for word in outdoor_keywords)

    # Augment with BLIP nouns
    detected_words = set(traffic_candidates + general_labels + danger_labels)
    objects_in_caption = re.findall(r"\ba\s+(\w+)|\ban\s+(\w+)|\bthe\s+(\w+)", caption)
    for match in objects_in_caption:
        for word in match:
            if word and word not in detected_words:
                general_labels.append(word)

    narration = caption.capitalize() + "."

    if is_outdoor and traffic_candidates:
        narration += " Traffic-related objects detected: " + ", ".join(set(traffic_candidates)) + "."
    else:
        general_labels += traffic_candidates

    if general_labels:
        narration += " Also detected: " + ", ".join(set(general_labels)) + "."

    if danger_labels:
        narration += " ⚠️ Danger! " + ", ".join(set(danger_labels)) + " detected."

    # Text-to-speech
    tts_engine.say(narration)
    tts_engine.runAndWait()

    return narration, caption, list(set(general_labels + traffic_candidates)), danger_labels

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/capture', methods=["POST"])
def capture():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        cv2.imwrite("static/captured.jpg", frame)
        narration, caption, objects, danger = generate_caption_and_objects(frame)
        return render_template("index.html", caption=caption.capitalize(), narration=narration, image="captured.jpg", objects=objects, danger=danger)
    else:
        return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)
