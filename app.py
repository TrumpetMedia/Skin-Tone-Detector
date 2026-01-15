from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import traceback

app = Flask(__name__)
CORS(app)


def _largest_face(faces):
    # faces are (x, y, w, h)
    return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]


def detect_tone_3class(image_pil):
    """
    Returns: (tone_str, confidence_float, debug_dict)
    tone_str in {"light","medium","deep"}
    """

    img_rgb = np.array(image_pil.convert("RGB"))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # ---- Face detection ----
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
    )

    used_face = False
    if len(faces) > 0:
        x, y, w, h = _largest_face(faces)
        used_face = True

        # Cheek ROI (tries to avoid eyes/hair; reduces background influence)
        x1 = int(x + 0.18 * w)
        x2 = int(x + 0.48 * w)
        y1 = int(y + 0.45 * h)
        y2 = int(y + 0.78 * h)
        roi = bgr[y1:y2, x1:x2]
    else:
        # Fallback: center crop (still avoids full-image background)
        H, W = bgr.shape[:2]
        roi = bgr[int(H * 0.25):int(H * 0.85), int(W * 0.30):int(W * 0.70)]

    if roi.size == 0:
        return "medium", 0.35, {"used_face": used_face, "reason": "empty_roi"}

    # ---- Skin pixels in YCrCb ----
    # Skin tends to cluster in Cb/Cr space (more stable than HSV for this use). [web:131][web:122]
    ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # Strict mask first
    mask = (Cr >= 135) & (Cr <= 180) & (Cb >= 85) & (Cb <= 135) & (Y >= 40)
    skin_Y = Y[mask]

    # Loosen if too few pixels
    if skin_Y.size < 250:
        mask = (Cr >= 130) & (Cr <= 190) & (Cb >= 75) & (Cb <= 145) & (Y >= 35)
        skin_Y = Y[mask]

    # Final fallback: ROI brightness only (never whole image)
    if skin_Y.size < 250:
        skin_Y = Y.flatten()

    # Robust brightness: percentile reduces shadows/beard influence
    y50 = float(np.percentile(skin_Y, 50))
    y70 = float(np.percentile(skin_Y, 70))
    luminance = y70 / 255.0  # 0..1

    # ---- 3-class thresholds (TUNE HERE) ----
    # These default thresholds are a starting point.
    # If you share 8-10 selfies, these can be tuned precisely.
    if luminance >= 0.64:
        tone = "light"
    elif luminance >= 0.48:
        tone = "medium"
    else:
        tone = "deep"

    # Confidence heuristic: higher if luminance is far from boundaries
    # distance to nearest boundary
    boundaries = [0.48, 0.64]
    dist = min(abs(luminance - b) for b in boundaries)
    confidence = max(0.35, min(1.0, 0.45 + dist * 1.8))

    debug = {
        "used_face": used_face,
        "skin_pixels_used": int(skin_Y.size),
        "y50": round(y50, 1),
        "y70": round(y70, 1),
        "luminance": round(float(luminance), 3),
    }

    return tone, float(confidence), debug


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/detect-tone", methods=["POST"])
def detect_tone():
    """
    POST /detect-tone
    Body: { "image": "<base64>" }
    Returns: { "tone": "light|medium|deep", "confidence": 0.xx }
    """
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        img_base64 = data["image"]

        try:
            img_data = base64.b64decode(img_base64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image: {str(e)}"}), 400

        tone, confidence, debug = detect_tone_3class(img)

        # If you want ONLY tone, remove confidence/debug from response.
        return jsonify({
            "tone": tone,
            "confidence": round(confidence, 2),
            "debug": debug
        }), 200

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
