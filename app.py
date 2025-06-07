import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import threading
import torch
import time
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3

yolo_model = YOLO("yolov8m.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

running = False
last_caption = ""
last_time = 0
cap = None


app = tk.Tk()
app.title("Image captioning App for real life scenes")
app.geometry("800x600")

video_label = Label(app)
video_label.pack()

caption_var = tk.StringVar()
caption_label = Label(app, textvariable=caption_var, font=("Helvetica", 14), wraplength=750, pady=10)
caption_label.pack()


def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()


def generate_caption(frame):
    global last_caption

    
    results = yolo_model(frame)[0]
    traffic_candidates = []
    danger_labels = []
    general_labels = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]

        if conf < 0.3:
            continue

        
        if label in ["person", "car", "bus", "truck", "motorbike", "bicycle"]:
            traffic_candidates.append(label)
        elif label in ["knife", "gun", "fire"]:
            danger_labels.append(label)
        elif label in ["hose", "cell phone", "remote", "tv monitor", "charger", "mouse"]:
            continue  
        else:
            general_labels.append(label)

   
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    inputs = blip_processor(pil_img, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True).lower()

    
    outdoor_keywords = ["road", "street", "traffic", "sidewalk", "intersection", "vehicle", "crosswalk", "bus stop", "parking", "outside", "highway"]
    is_outdoor_scene = any(word in caption for word in outdoor_keywords)

    narration = caption.capitalize() + "."

    
    if is_outdoor_scene and traffic_candidates:
        narration += " Traffic-related objects detected: " + ", ".join(set(traffic_candidates)) + "."
    else:
        
        general_labels += traffic_candidates

    if general_labels:
        narration += " Also detected: " + ", ".join(set(general_labels)) + "."

    if danger_labels:
        narration += " ⚠️ Danger! " + ", ".join(set(danger_labels)) + " detected."

    
    if narration != last_caption:
        print("[NARRATION]:", narration)
        caption_var.set(narration)
        speak(narration)
        last_caption = narration
    import re
    detected_words = set(general_labels + traffic_candidates + danger_labels)
    possible_objects = re.findall(r"\ba\s+(\w+)|\ban\s+(\w+)|\bthe\s+(\w+)", caption)
    for match in possible_objects:
        for word in match:
            if word and word not in detected_words:
                general_labels.append(word)
    
def update_frame():
    global cap, running, last_time

    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    if time.time() - last_time > 5:
        threading.Thread(target=generate_caption, args=(frame,)).start()
        last_time = time.time()

    video_label.after(10, update_frame)


def start_camera():
    global cap, running
    if not running:
        cap = cv2.VideoCapture(0)
        running = True
        update_frame()
        caption_var.set("Camera started. Observing scene...")

def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
        cap = None
    caption_var.set("Camera stopped.")
    video_label.config(image='')


start_btn = Button(app, text="Start Camera", command=start_camera, bg="green", fg="white", font=("Helvetica", 12))
start_btn.pack(pady=10)

stop_btn = Button(app, text="Stop Camera", command=stop_camera, bg="red", fg="white", font=("Helvetica", 12))
stop_btn.pack(pady=5)


app.mainloop()


