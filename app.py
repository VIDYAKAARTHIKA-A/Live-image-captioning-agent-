import cv2
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
cv2.destroyAllWindows()




