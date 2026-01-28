import os
import base64
import json
import logging
from io import BytesIO
from openai import OpenAI
from google import genai
from google.genai import types
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MAX_IMAGE_SIZE = 1024
MIN_DIM = 180  # if width or height below this, likely unusable selfie
# MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
Classify facial skin tone for cosmetic shade matching.

Return ONLY JSON:
{"tone":"light"} or {"tone":"medium"} or {"tone":"deep"} or {"tone":null}

If no clear human face is visible or image quality/lighting is bad: {"tone":null}
No extra keys. No extra text.
"""

VALID_TONES = {"light", "medium", "deep"}

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route("/detect-tone", methods=["POST"])
def detect_tone():
    log.info("REQUEST_RECEIVED")

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        log.warning("IMAGE_MISSING_IN_REQUEST")
        return jsonify({"tone": None}), 200

    # ---------- Decode image ----------
    try:
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        log.info("IMAGE_DECODED_SUCCESSFULLY")
    except Exception as e:
        log.error("IMAGE_DECODE_FAILED | %s", str(e))
        return jsonify({"tone": None}), 200

    # ---------- Resize (keep aspect) ----------
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    log.info("IMAGE_RESIZED | size=%s", img.size)

    # If the image is too small, return null (prevents model nonsense/refusal)
    if img.size[0] < MIN_DIM or img.size[1] < MIN_DIM:
        log.warning("IMAGE_TOO_SMALL_FOR_FACE | size=%s", img.size)
        return jsonify({"tone": None}), 200

    # ---------- Re-encode ----------
    out = BytesIO()
    img.save(out, format="JPEG", quality=95, subsampling=0)
    clean_b64 = base64.b64encode(out.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{clean_b64}"
    log.info("IMAGE_REENCODED")

    # ---------- Call OpenAI ----------
    log.info("OPENAI_REQUEST_START | model=%s", MODEL_NAME)

    # try:
    #     response = client.chat.completions.create(
    #         model=MODEL_NAME,
    #         # JSON mode: forces valid JSON output (prevents your parse crash)
    #         response_format={"type": "json_object"},  # [web:578]
    #         messages=[
    #             {"role": "system", "content": SYSTEM_PROMPT},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "text", "text": "Classify skin tone."},
    #                     {"type": "image_url", "image_url": {"url": data_url}},
    #                 ],
    #             },
    #         ],
    #         max_tokens=50,
    #         temperature=0,
    #     )

    try:

     response = client.models.generate_content(
         model=MODEL_NAME,
         contents=[
             types.Part.from_bytes(data=out.getvalue(), mime_type="image/jpeg"),
             SYSTEM_PROMPT
         ],
         config=types.GenerateContentConfig(
             temperature=0,
             response_mime_type="application/json",
             response_json_schema={
                 "type": "object",
                 "properties": {
                     "tone": {"type": ["string", "null"], "enum": ["light", "medium", "deep", None]}
                 },
                 "required": ["tone"],
                 "additionalProperties": False
             }
         )
     )

    except Exception as e:
        log.error("OPENAI_CALL_FAILED | %s", str(e))
        return jsonify({"tone": None}), 200

    log.info("OPENAI_RESPONSE_RECEIVED")

    # raw = (response.choices[0].message.content or "").strip()
    raw = (response.text or "").strip()

    log.info("RAW_MODEL_OUTPUT | %s", raw)

    # ---------- Parse JSON safely ----------
    try:
        parsed = json.loads(raw)
    except Exception as e:
        log.error("JSON_PARSE_FAILED | %s | raw=%s", str(e), raw)
        return jsonify({"tone": None}), 200

    tone = parsed.get("tone", None)

    # Normalize + validate
    if isinstance(tone, str):
        tone = tone.strip().lower()

    if tone not in VALID_TONES:
        tone = None

    log.info("CLASSIFICATION_FINAL | tone=%s", tone)
    return jsonify({"tone": tone}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
