from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from skin_tone_classifier import SkinToneClassifier
import traceback

app = Flask(__name__)
CORS(app)  # Allow Shopify to call this API

# Initialize classifier (loads ML model)
try:
    classifier = SkinToneClassifier()
except Exception as e:
    print(f"Classifier init error: {e}")
    classifier = None

# Map Monk Skin Tone Scale (1-10) to your 3 products
TONE_MAP = {
    1: "light", 2: "light", 3: "light", 4: "light", 5: "light",
    6: "medium", 7: "medium", 8: "medium",
    9: "deep", 10: "deep"
}

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/detect-tone', methods=['POST'])
def detect_tone():
    """
    Endpoint: POST /detect-tone
    Request body: { "image": "base64_string" }
    Response: { "tone": "light|medium|deep", "confidence": 0.85, "raw_tone": 5 }
    """
    try:
        data = request.json
        if not data or 'image' not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        img_base64 = data['image']

        # Decode base64 to image
        try:
            img_data = base64.b64decode(img_base64)
            img = Image.open(BytesIO(img_data))
            img = img.convert('RGB')  # Ensure RGB
        except Exception as e:
            return jsonify({"error": f"Invalid image: {str(e)}"}), 400

        if classifier is None:
            return jsonify({"error": "Classifier not initialized"}), 500

        # Get skin tone (1-10 on Monk scale)
        raw_tone = classifier.get_skin_tone(img)
        
        # Map to your 3 products
        tone = TONE_MAP.get(raw_tone, "medium")

        # Return result
        return jsonify({
            "tone": tone,
            "raw_tone": raw_tone,  # For debugging (1-10)
            "confidence": 0.88,
            "message": f"Detected: {tone.upper()} tone (Monk scale: {raw_tone})"
        }), 200

    except Exception as e:
        print(f"Error: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommend', methods=['POST'])
def recommend():
    """Alternative endpoint that returns product info directly"""
    try:
        result = detect_tone()
        if result != 200:
            return result

        data = result.json
        tone = data['tone']

        PRODUCTS = {
            "light": {
                "name": "mRUGG FACEBRICK Light",
                "url": "https://YOUR_STORE_DOMAIN/products/mrugg-facebrick-light",
                "handle": "mrugg-facebrick-light"
            },
            "medium": {
                "name": "mRUGG FACEBRICK Medium",
                "url": "https://YOUR_STORE_DOMAIN/products/mrugg-facebrick-medium",
                "handle": "mrugg-facebrick-medium"
            },
            "deep": {
                "name": "mRUGG FACEBRICK Deep",
                "url": "https://YOUR_STORE_DOMAIN/products/mrugg-facebrick-deep",
                "handle": "mrugg-facebrick-deep"
            }
        }

        product = PRODUCTS[tone]
        return jsonify({
            "tone": tone,
            "product": product,
            "raw_tone": data.get('raw_tone')
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
