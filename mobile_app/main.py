import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image as KivyImage
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock

import requests
import cv2
import threading
import pyttsx3
import time

class CaptionClientApp(App):
    def build(self):
        self.layout = BoxLayout(orientation='vertical')
        self.image = KivyImage()
        self.label = Label(text="Press Start to begin", size_hint=(1, 0.2))
        self.button = Button(text="Start Live Caption", size_hint=(1, 0.2))
        self.button.bind(on_press=self.toggle_captioning)

        self.layout.add_widget(self.image)
        self.layout.add_widget(self.label)
        self.layout.add_widget(self.button)

        self.capture = cv2.VideoCapture(0)
        self.captioning = False

        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)  # 30 FPS
        return self.layout

    def update_frame(self, dt):
        ret, frame = self.capture.read()
        if ret:
            self.current_frame = frame
            frame = cv2.flip(frame, 0)
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.image.texture = texture

    def toggle_captioning(self, instance):
        if not self.captioning:
            self.captioning = True
            self.button.text = "Stop Live Caption"
            threading.Thread(target=self.caption_loop, daemon=True).start()
        else:
            self.captioning = False
            self.button.text = "Start Live Caption"

    def caption_loop(self):
        while self.captioning:
            if hasattr(self, 'current_frame'):
                _, img_encoded = cv2.imencode('.jpg', self.current_frame)
                image_bytes = img_encoded.tobytes()
                self.send_to_server(image_bytes)
            time.sleep(2)  # 1 caption every 2 seconds (adjustable)

    def send_to_server(self, image_bytes):
        try:
            server_url = "http://127.0.0.1:5000/caption"
            files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
            response = requests.post(server_url, files=files)

            if response.status_code == 200:
                caption = response.json().get('caption', 'No caption')

                # Update label on UI thread
                Clock.schedule_once(lambda dt: self.label.setter('text')(self.label, caption))

                # Speak
                engine = pyttsx3.init()
                engine.say(caption)
                engine.runAndWait()
            else:
                Clock.schedule_once(lambda dt: self.label.setter('text')(
                    self.label, f"Server error: {response.status_code}"))
        except Exception as e:
            Clock.schedule_once(lambda dt: self.label.setter('text')(self.label, f"Error: {str(e)}"))

if __name__ == "__main__":
    CaptionClientApp().run()
