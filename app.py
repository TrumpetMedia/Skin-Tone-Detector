import base64
import numpy as np
import cv2
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from sklearn.cluster import KMeans

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def white_patch_retinex(img):
    """Color constancy: neutralize dominant light cast."""
    img_float = img.astype(float)
    b, g, r = cv2.split(img_float)

    def get_max_val(channel):
        flat = np.sort(channel.flatten())[::-1]
        top_count = max(1, int(len(flat) * 0.01))
        return np.mean(flat[:top_count])

    b_max = max(1, get_max_val(b))
    g_max = max(1, get_max_val(g))
    r_max = max(1, get_max_val(r))

    b = np.minimum(b * (255.0 / b_max), 255)
    g = np.minimum(g * (255.0 / g_max), 255)
    r = np.minimum(r * (255.0 / r_max), 255)

    return cv2.merge([b, g, r]).astype(np.uint8)


def get_true_skin_tone_from_bgr(img_bgr):
    """
    Extract skin tone features from an already-loaded BGR image.
    Returns: (debug_info, chroma, hue_bias, face_base64)
    """
    if img_bgr is None:
        return None, None, None, None

    # 1) Face detection
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        h, w = img_bgr.shape[:2]
        roi = img_bgr[int(h * 0.2): int(h * 0.8), int(w * 0.2): int(w * 0.8)]
    else:
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        roi = img_bgr[
            y + int(h * 0.15): y + int(h * 0.85),
            x + int(w * 0.15): x + int(w * 0.85)
        ]

    # 2) Color correction
    corrected_roi = white_patch_retinex(roi)

    # Encode face crop for frontend preview/debug (not saved to disk)
    ok, buffer = cv2.imencode(".jpg", corrected_roi)
    face_base64 = base64.b64encode(buffer).decode("utf-8") if ok else None

    # 3) Skin masking (YCbCr)
    ycrcb_roi = cv2.cvtColor(corrected_roi, cv2.COLOR_BGR2YCrCb)
    min_ycrcb = np.array([80, 135, 85], np.uint8)
    max_ycrcb = np.array([255, 170, 115], np.uint8)

    mask = cv2.inRange(ycrcb_roi, min_ycrcb, max_ycrcb)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    skin_extracted = cv2.bitwise_and(corrected_roi, corrected_roi, mask=mask)
    pixels_bgr = skin_extracted[mask == 255]

    if pixels_bgr.shape[0] < 50:
        return None, None, None, face_base64

    # 4) K-means (kept for consistency; not used directly)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    kmeans.fit(pixels_bgr)

    # Distribution stats in YCbCr
    all_ycrcb = cv2.cvtColor(skin_extracted, cv2.COLOR_BGR2YCrCb)
    skin_y = all_ycrcb[mask == 255, 0]
    skin_cb = all_ycrcb[mask == 255, 1]
    skin_cr = all_ycrcb[mask == 255, 2]

    y_mean = float(np.mean(skin_y))
    y_p50 = float(np.percentile(skin_y, 50))
    y_p75 = float(np.percentile(skin_y, 75))

    cb_mean = float(np.mean(skin_cb))
    cr_mean = float(np.mean(skin_cr))

    chroma = float(np.sqrt((cb_mean - 128) ** 2 + (cr_mean - 128) ** 2))
    hue_bias = float(cr_mean / (cb_mean + 1e-6))

    debug_info = {
        "y_mean": y_mean,
        "y_p50": y_p50,
        "y_p75": y_p75,
        "cb_mean": cb_mean,
        "cr_mean": cr_mean,
        "chroma": chroma,
        "hue_bias": hue_bias,
    }

    return debug_info, chroma, hue_bias, face_base64


def classify_tone_v2_improved(y_mean, y_p75, chroma, hue_bias):
    # Warm-light detection
    warm_light_suspected = (hue_bias < 0.8) and (130 < y_mean < 165)

    score = 0.0

    # Luminance
    if y_mean >= 170:
        score += 3.0
    elif y_mean >= 160:
        score += 2.0
    elif y_mean >= 150:
        score += 1.0
    elif y_mean >= 140:
        score -= 0.5
    else:
        score -= 2.0

    # Chroma
    if chroma > 38:
        score -= 2.0
    elif chroma > 30:
        score -= 1.5
    elif chroma > 25:
        score -= 0.5
    elif chroma < 18:
        score += 2.5
    elif chroma < 22:
        score += 1.5

    # Hue bias (downweighted if warm light suspected)
    hue_weight = 0.5 if warm_light_suspected else 1.0

    if hue_bias < 0.75:
        score -= 1.0 * hue_weight
    elif hue_bias < 0.85:
        score -= 0.5 * hue_weight
    elif hue_bias > 1.15:
        score += 1.0 * hue_weight
    elif hue_bias > 1.05:
        score += 0.5 * hue_weight

    if score > 1:
        return "light"
    elif score < -1.2:
        return "deep"
    return "medium"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/detect-tone", methods=["POST"])
def detect_tone():
    try:
        data = request.get_json(silent=True)

        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        # Decode base64 to bytes
        img_bytes = base64.b64decode(data["image"])

        # Load with PIL, resize, convert to OpenCV BGR
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((800, 800))
        img_rgb = np.array(img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        debug_info, chroma, hue_bias, face_img = get_true_skin_tone_from_bgr(img_bgr)

        if debug_info is None:
            return jsonify({
                "error": "No skin detected",
                "face_crop": face_img
            }), 400

        y_mean = debug_info["y_mean"]
        y_p75 = debug_info["y_p75"]

        tone = classify_tone_v2_improved(y_mean, y_p75, chroma, hue_bias)

        return jsonify({
            "tone": tone,
            "debug": {
                "y_mean": round(y_mean, 2),
                "y_p75": round(y_p75, 2),
                "chroma": round(chroma, 2),
                "hue_bias": round(hue_bias, 2),
            },
            "face_crop": face_img,
        }), 200

    except Exception:
        # Avoid leaking internal details in production responses
        return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    # For local dev. In production, run via gunicorn/uvicorn.
    app.run(host="0.0.0.0", port=5000, debug=False)
