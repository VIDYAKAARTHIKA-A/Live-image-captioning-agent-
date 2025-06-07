'''import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3
import time

# Load BLIP
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Initialize text-to-speech
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# Start camera
cap = cv2.VideoCapture(0)

last_caption = ""
last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the video feed
    cv2.imshow('Live Feed - Press q to quit', frame)

    # Run captioning every 2 seconds
    if time.time() - last_time > 2:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Generate caption
        inputs = processor(pil_img, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        if caption != last_caption:
            print("Caption:", caption)
            tts_engine.say(caption)
            tts_engine.runAndWait()
            last_caption = caption

        last_time = time.time()

    # Quit with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()'''


'''import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import cv2
import threading
import torch
import time
from ultralytics import YOLO
from transformers import BlipProcessor, BlipForConditionalGeneration
import pyttsx3

# ------------------- Load Models -------------------
yolo_model = YOLO("yolov8n.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# ------------------- App State -------------------
running = False
last_caption = ""
last_time = 0
cap = None

# ------------------- GUI Setup -------------------
app = tk.Tk()
app.title("Assistive Captioning App for Traffic Scenes")
app.geometry("800x600")

video_label = Label(app)
video_label.pack()

caption_var = tk.StringVar()
caption_label = Label(app, textvariable=caption_var, font=("Helvetica", 14), wraplength=750, pady=10)
caption_label.pack()

# ------------------- Functions -------------------
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def generate_caption(frame):
    global last_caption

    # Run YOLO
    results = yolo_model(frame)[0]
    labels = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]
        if conf > 0.5 and label in ["person", "car", "bus", "truck", "motorbike", "traffic light", "bicycle"]:
            labels.append(label)

    object_summary = ", ".join(set(labels)) if labels else "no major traffic objects"

    # BLIP captioning
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    inputs = blip_processor(pil_img, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)

    final_caption = f"{caption}. Objects detected: {object_summary}."

    if final_caption != last_caption:
        last_caption = final_caption
        caption_var.set(final_caption)
        speak(final_caption)

def update_frame():
    global cap, running, last_time

    if not running or cap is None:
        return

    ret, frame = cap.read()
    if not ret:
        return

    # Display frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Every 5 seconds, generate caption
    if time.time() - last_time > 5:
        threading.Thread(target=generate_caption, args=(frame,)).start()
        last_time = time.time()

    # Loop
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

# ------------------- Buttons -------------------
start_btn = Button(app, text="Start Camera", command=start_camera, bg="green", fg="white", font=("Helvetica", 12))
start_btn.pack(pady=10)

stop_btn = Button(app, text="Stop Camera", command=stop_camera, bg="red", fg="white", font=("Helvetica", 12))
stop_btn.pack(pady=5)

# ------------------- Launch -------------------
app.mainloop()'''

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

# ------------------- Load Models -------------------
yolo_model = YOLO("yolov8m.pt")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)

# ------------------- App State -------------------
running = False
last_caption = ""
last_time = 0
cap = None

# ------------------- GUI Setup -------------------
app = tk.Tk()
app.title("Image captioning App for real life scenes")
app.geometry("800x600")

video_label = Label(app)
video_label.pack()

caption_var = tk.StringVar()
caption_label = Label(app, textvariable=caption_var, font=("Helvetica", 14), wraplength=750, pady=10)
caption_label.pack()

# ------------------- Text-to-Speech -------------------
def speak(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

# ------------------- Scene + Object Detection -------------------
def generate_caption(frame):
    global last_caption

    # ------------------- YOLO Detection -------------------
    results = yolo_model(frame)[0]
    traffic_candidates = []
    danger_labels = []
    general_labels = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = results.names[cls_id]

        if conf < 0.5:
            continue

        # Classify later after context is known
        if label in ["person", "car", "bus", "truck", "motorbike", "bicycle"]:
            traffic_candidates.append(label)
        elif label in ["knife", "gun", "fire"]:
            danger_labels.append(label)
        elif label in ["hose", "cell phone", "remote", "tv monitor", "charger", "mouse"]:
            continue  # skip mislabels
        else:
            general_labels.append(label)

    # ------------------- BLIP Captioning -------------------
    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    inputs = blip_processor(pil_img, return_tensors="pt")
    output = blip_model.generate(**inputs)
    caption = blip_processor.decode(output[0], skip_special_tokens=True).lower()

    # ------------------- Context Check -------------------
    outdoor_keywords = ["road", "street", "traffic", "sidewalk", "intersection", "vehicle", "crosswalk", "bus stop", "parking", "outside", "highway"]
    is_outdoor_scene = any(word in caption for word in outdoor_keywords)

    narration = caption.capitalize() + "."

    # ------------------- Narration Assembly -------------------
    if is_outdoor_scene and traffic_candidates:
        narration += " Traffic-related objects detected: " + ", ".join(set(traffic_candidates)) + "."
    else:
        # Consider all detected objects general if indoors
        general_labels += traffic_candidates

    if general_labels:
        narration += " Also detected: " + ", ".join(set(general_labels)) + "."

    if danger_labels:
        narration += " ⚠️ Danger! " + ", ".join(set(danger_labels)) + " detected."

    # ------------------- Speak & Update GUI -------------------
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
    




# ------------------- Live Frame Update -------------------
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

# ------------------- Start & Stop -------------------
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

# ------------------- Buttons -------------------
start_btn = Button(app, text="Start Camera", command=start_camera, bg="green", fg="white", font=("Helvetica", 12))
start_btn.pack(pady=10)

stop_btn = Button(app, text="Stop Camera", command=stop_camera, bg="red", fg="white", font=("Helvetica", 12))
stop_btn.pack(pady=5)

# ------------------- Launch -------------------
app.mainloop()


