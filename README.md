**VISION MATE: SCENE CAPTIONING FOR THE VISUALLY IMPAIRED**


VisionMate is an intelligent vision assistant which helps visually impaired individuals understand their surroundings. It captures scenes through a webcam or mobile camera, identifies key objects (e.g., people, vehicles, signals), and generates natural language descriptions with voice narration.

This project is especially designed to assist during road-crossing or traffic navigation, where awareness of vehicles and potential hazards is critical. It also detects dangerous situations and gives alerts.

---

**FEATURES:**


📷 Real-time camera feed capture

🧠 Object detection using YOLOv8

🗣️ Scene captioning using BLIP (Bootstrapped Language-Image Pretraining)

🔊 Text-to-speech narration

🛑 Danger alerts for harmful or violent objects

🚦 Identifies traffic-related scenes

⚙️ Works as a desktop GUI or can be extended to mobile

---

**TECH STACK**



🐍 Python 🔦 PyTorch 📸 OpenCV 🖼️ Kivy 🌐 Flask

---


**PROJECT WORKFLOW**

![image](https://github.com/user-attachments/assets/d777dc14-adfd-4c0d-8548-d61358e4e67b)

---

**WHY USE BLIP?**

🔍 Highly Accurate Captions

⚡ Fast Inference for Real-Time Use

🧠 Strong Generalization

🧩 Easy Integration

🔧 No Manual Pipeline Needed

🗣️ Natural Language Output

---


**WHY USE YOLO(You only look once)?**


BLIP provides a high-level, descriptive caption, but it doesn’t localize objects or give bounding boxes. YOLO complements BLIP by providing precise detection and localization of individual objects within the frame, which can be essential for certain applications like tracking, interaction, or further analysis.

---

**FURTHER ENHANCEMENTS:**


* The model should generate more accurate captions for different types of objects.  
* The model should work efficiently in noisy environments.  
* The app should be deployed on mobile.

---

**SETUP AND INSTALLATION**


Clone the repo:  
`git clone https://github.com/yourusername/live-image-captioning-assistant.git`  
`cd live-image-captioning-assistant`

Create a virtual environment:  
`python -m venv venv`  
`source venv/bin/activate`  # Linux/macOS  
`venv\Scripts\activate`   # Windows

Install dependencies:  
`pip install -r requirements.txt`

For webcam live captioning:  
`python main.py`

For Flask:  
`python app.py`

---


**REFERENCES:**



BLIP: https://github.com/salesforce/BLIP

YOLOv8 (Ultralytics): https://github.com/ultralytics/ultralytics

