from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import traceback

app = Flask(__name__)
CORS(app)  # Allow Shopify to call this API

def analyze_skin_tone_opencv(image):
    """
    Analyze skin tone using OpenCV color space analysis
    Returns a value 1-10 (Monk Skin Tone Scale)
    """
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        bgr_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Convert to HSV for better skin tone analysis
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        
        # Define skin tone range in HSV
        # Skin tones typically have:
        # H: 0-20 or 340-360 (reddish-orange)
        # S: 10-40 (some saturation)
        # V: 50-255 (brightness)
        
        lower_skin1 = np.array([0, 10, 50], dtype=np.uint8)
        upper_skin1 = np.array([20, 40, 255], dtype=np.uint8)
        
        lower_skin2 = np.array([340, 10, 50], dtype=np.uint8)
        upper_skin2 = np.array([360, 40, 255], dtype=np.uint8)
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Extract skin regions
        skin_pixels = bgr_image[mask > 0]
        
        if len(skin_pixels) == 0:
            # If no skin detected, use overall image analysis
            skin_pixels = bgr_image.reshape(-1, 3)
        
        # Calculate average BGR values
        avg_b = np.mean(skin_pixels[:, 0])
        avg_g = np.mean(skin_pixels[:, 1])
        avg_r = np.mean(skin_pixels[:, 2])
        
        # Calculate luminance (brightness)
        luminance = (0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b) / 255.0
        
        # Calculate red intensity
        red_intensity = avg_r / 255.0
        
        # Map to Monk Scale (1-10)
        # Light: 1-3 (high luminance)
        # Medium: 4-7 (medium luminance)
        # Deep: 8-10 (low luminance)
        
        if luminance > 0.65:
            tone = 2  # Light
        elif luminance > 0.50:
            tone = 5  # Medium
        else:
            tone = 9  # Deep
        
        return tone, luminance, red_intensity
        
    except Exception as e:
        print(f"OpenCV analysis error: {e}")
        return 5, 0.5, 0.5  # Default to medium

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

        # Analyze skin tone using OpenCV
        raw_tone, luminance, red_intensity = analyze_skin_tone_opencv(img)
        
        # Map to your 3 products
        tone = TONE_MAP.get(raw_tone, "medium")
        
        # Calculate confidence (0-1)
        confidence = min(abs(luminance - 0.5) + 0.3, 1.0)

        # Return result
        return jsonify({
            "tone": tone,
            "raw_tone": raw_tone,  # For debugging (1-10)
            "confidence": round(confidence, 2),
            "luminance": round(luminance, 2),
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
