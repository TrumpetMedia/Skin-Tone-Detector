import base64
import numpy as np
import cv2
import logging
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# ---------------- CONFIG ---------------- #

MAX_IMAGE_SIZE = 800
MIN_SKIN_PIXELS = 120

# Logging (prints to Render / console)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ---------------- APP ---------------- #

app = Flask(__name__)
CORS(app)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- UTILITIES ---------------- #

def white_patch_retinex_safe(img):
    """
    Safer color constancy:
    uses top 0.5% instead of 1% to avoid highlight blowout.
    """
    img = img.astype(np.float32)
    out = np.zeros_like(img)

    for i in range(3):
        channel = img[:, :, i]
        top = np.percentile(channel, 99.5)
        scale = 255.0 / max(top, 1)
        out[:, :, i] = np.clip(channel * scale, 0, 255)

    return out.astype(np.uint8)


def extract_face_roi(img):
    """
    Detect face; fallback to center crop.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        h, w = img.shape[:2]
        log.info("Face not detected, using center fallback")
        return img[int(h * 0.25):int(h * 0.75),
                   int(w * 0.25):int(w * 0.75)]

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    # log.info(f"Face detected: x={x}, y={y}, w={w}, h={h}")

    return img[y:y + h, x:x + w]


def extract_cheeks(face):
    """
    Sample left & right cheek regions only.
    This dramatically improves tone stability.
    """
    h, w = face.shape[:2]

    left = face[int(h * 0.45):int(h * 0.65),
                int(w * 0.15):int(w * 0.35)]

    right = face[int(h * 0.45):int(h * 0.65),
                 int(w * 0.65):int(w * 0.85)]

    return cv2.vconcat([left, right])


def skin_mask_combined(bgr):
    """
    Combined HSV + YCrCb skin masking.
    Much more robust across ethnicities.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)

    hsv_mask = cv2.inRange(
        hsv,
        np.array([0, 30, 60]),
        np.array([25, 200, 255])
    )

    ycrcb_mask = cv2.inRange(
        ycrcb,
        np.array([0, 133, 77]),
        np.array([255, 173, 127])
    )

    mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)
    mask = cv2.medianBlur(mask, 5)

    return mask


# ---------------- CORE LOGIC ---------------- #

def analyze_skin_tone(img_bgr):
    """
    Returns:
    tone, debug_info, face_crop_base64
    """

    face = extract_face_roi(img_bgr)
    face = white_patch_retinex_safe(face)
    cheeks = extract_cheeks(face)

    # Encode face preview
    ok, buf = cv2.imencode(".jpg", face)
    face_b64 = base64.b64encode(buf).decode() if ok else None

    mask = skin_mask_combined(cheeks)
    skin_pixels = cheeks[mask > 0]

    if skin_pixels.shape[0] < MIN_SKIN_PIXELS:
        log.warning("Not enough skin pixels detected")
        return None, None, face_b64

    ycrcb = cv2.cvtColor(skin_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, 0, 0]
    cb = ycrcb[:, 0, 1]
    cr = ycrcb[:, 0, 2]

    # Robust luminance estimation
    y_sorted = np.sort(y)
    base_y = float(np.median(y_sorted[:int(len(y_sorted) * 0.35)]))

    chroma = float(np.sqrt(
        (np.mean(cb) - 128) ** 2 +
        (np.mean(cr) - 128) ** 2
    ))
    y_p75 = float(np.percentile(y, 75))
    rsbi = base_y / max(y_p75, 1)
    debug = {
        "skin_pixels": int(len(y)),
        "base_y": round(base_y, 2),
        "y_p75": round(y_p75, 2),
        "rsbi": round(rsbi, 3),
        "chroma": round(chroma, 2),
        "y_mean": round(float(np.mean(y)), 2),
        "cb_mean": round(float(np.mean(cb)), 2),
        "cr_mean": round(float(np.mean(cr)), 2)
    }

    # log.info(
    #     f"METRICS | base_y={debug['base_y']} | "
    #     f"y_p75={debug['y_p75']} | "
    #     f"rsbi={debug['rsbi']} | "
    #     f"chroma={debug['chroma']}"
    # )

    # Classification (non-ML)
    if rsbi > 0.82:
        tone = "light"
    elif rsbi < 0.68:
        tone = "deep"
    else:
        tone = "medium"

    # log.info(f"CLASSIFIED TONE: {tone}")

    return tone, debug, face_b64


# ---------------- ROUTES ---------------- #

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/detect-tone", methods=["POST"])
def detect_tone():
    try:
        data = request.get_json(silent=True)
        if not data or "image" not in data:
            return jsonify({"error": "Missing image"}), 400

        img_bytes = base64.b64decode(data["image"])
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        tone, debug, face_crop = analyze_skin_tone(img_bgr)

        if tone is None:
          return jsonify({
                "error_code": "NO_FACE_DETECTED",
                "message": "No face detected. Please upload a clear selfie with your face visible.",
                "face_crop": face_crop
            }), 400
        return jsonify({
            "tone": tone,
            "debug": debug,
            "face_crop": face_crop
        }), 200

    except Exception as e:
        log.exception("SERVER ERROR")
        return jsonify({"error": "Server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
