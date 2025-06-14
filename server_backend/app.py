from flask import Flask, request, jsonify
from PIL import Image
import io
from models.model_loader import generate_caption

app = Flask(__name__)

@app.route('/caption', methods=['POST'])
def caption():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    caption_text = generate_caption(image)

    return jsonify({'caption': caption_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
