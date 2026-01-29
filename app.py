import os
import base64
import json
import logging
import requests
import threading
from io import BytesIO
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()


MAX_IMAGE_SIZE = 1024
MIN_DIM = 180
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")


SHOPIFY_STORE = os.getenv("SHOPIFY_STORE")
SHOPIFY_ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
SHOPIFY_API_URL = f"https://{SHOPIFY_STORE}/admin/api/2024-01"


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


SYSTEM_PROMPT = """
Classify facial skin tone for cosmetic shade matching.


Return ONLY JSON:
{"tone":"light"} or {"tone":"medium"} or {"tone":"deep"} or {"tone":null}


If no clear human face is visible or image quality/lighting is bad: {"tone":null}
No extra keys. No extra text.
"""


VALID_TONES = {"light", "medium", "deep"}


# ------------------------------------------------------------------
# SHOPIFY CUSTOMER (UNIQUE EMAIL)
# ------------------------------------------------------------------
def create_or_get_customer(name, email, phone, tone):
    """Create or update Shopify customer with tone tags."""
    headers = {
        "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
        "Content-Type": "application/json"
    }

    # Check if customer exists
    search_url = f"{SHOPIFY_API_URL}/customers/search.json?query=email:{email}"
    resp = requests.get(search_url, headers=headers, timeout=10)

    if resp.status_code == 200:
        customers = resp.json().get("customers", [])
        if customers:
            customer = customers[0]
            customer_id = customer["id"]

            # Update tags
            tags = set((customer.get("tags") or "").split(","))
            tags.add("ai-recommendation")
            if tone:
                tags.add(f"tone-{tone}")

            requests.put(
                f"{SHOPIFY_API_URL}/customers/{customer_id}.json",
                json={"customer": {"id": customer_id, "tags": ",".join(tags)}},
                headers=headers,
                timeout=10
            )

            log.info("SHOPIFY_CUSTOMER_EXISTS | id=%s | email=%s", customer_id, email)
            return customer_id

    # Create new customer
    parts = name.split(" ", 1)
    payload = {
        "customer": {
            "first_name": parts[0],
            "last_name": parts[1] if len(parts) > 1 else "",
            "email": email,
            "phone": phone,
            "tags": f"ai-recommendation{',tone-' + tone if tone else ''}"
        }
    }

    resp = requests.post(
        f"{SHOPIFY_API_URL}/customers.json",
        json=payload,
        headers=headers,
        timeout=10
    )

    if resp.status_code in (200, 201):
        customer_id = resp.json()["customer"]["id"]
        log.info("SHOPIFY_CUSTOMER_CREATED | id=%s | email=%s", customer_id, email)
        return customer_id

    log.error("SHOPIFY_CUSTOMER_CREATE_FAILED | %s", resp.text)
    return None


# ------------------------------------------------------------------
# SHOPIFY FILE UPLOAD
# ------------------------------------------------------------------
def upload_image_to_shopify_files(image_bytes, name, tone):
    """Upload image to Shopify Files (3-step process)."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in "_-" else "_" for c in name)
        filename = f"selfie_{safe_name}_{timestamp}_{tone or 'unknown'}.jpg"

        graphql_url = f"https://{SHOPIFY_STORE}/admin/api/2024-01/graphql.json"
        headers = {
            "X-Shopify-Access-Token": SHOPIFY_ACCESS_TOKEN,
            "Content-Type": "application/json"
        }

        # STEP 1: Request staged upload
        staged_resp = requests.post(
            graphql_url,
            json={
                "query": """
                mutation stagedUploadsCreate($input: [StagedUploadInput!]!) {
                  stagedUploadsCreate(input: $input) {
                    stagedTargets {
                      url
                      resourceUrl
                    }
                  }
                }
                """,
                "variables": {
                    "input": [{
                        "filename": filename,
                        "mimeType": "image/jpeg",
                        "resource": "FILE",
                        "httpMethod": "PUT"
                    }]
                }
            },
            headers=headers,
            timeout=30
        )

        staged = staged_resp.json()["data"]["stagedUploadsCreate"]["stagedTargets"][0]
        upload_url = staged["url"]
        resource_url = staged["resourceUrl"]

        log.info("SHOPIFY_STAGED_UPLOAD | filename=%s", filename)

        # STEP 2: Upload binary to staging
        requests.put(
            upload_url,
            data=image_bytes,
            headers={"Content-Type": "image/jpeg"},
            timeout=30
        )

        log.info("SHOPIFY_FILE_UPLOADED | size=%s", len(image_bytes))

        # STEP 3: Finalize file in Shopify Files
        file_resp = requests.post(
            graphql_url,
            json={
                "query": """
                mutation fileCreate($files: [FileCreateInput!]!) {
                  fileCreate(files: $files) {
                    files {
                      id
                      preview {
                        image {
                          url
                        }
                      }
                    }
                    userErrors {
                      message
                    }
                  }
                }
                """,
                "variables": {
                    "files": [{
                        "originalSource": resource_url,
                        "contentType": "IMAGE",
                        "alt": f"Selfie - skin tone {tone or 'unknown'}"
                    }]
                }
            },
            headers=headers,
            timeout=30
        )

        file_json = file_resp.json()
        user_errors = file_json["data"]["fileCreate"]["userErrors"]

        if user_errors:
            log.warning("SHOPIFY_FILE_ERRORS | %s", user_errors)
            return resource_url

        file_obj = file_json["data"]["fileCreate"]["files"][0]
        preview = file_obj.get("preview")

        if preview and preview.get("image") and preview["image"].get("url"):
            file_url = preview["image"]["url"]
            log.info("SHOPIFY_FILE_SUCCESS | url=%s", file_url)
            return file_url

        # Fallback if preview not ready yet
        log.info("SHOPIFY_FILE_PENDING | using_resource_url")
        return resource_url

    except Exception as e:
        log.error("SHOPIFY_FILE_UPLOAD_ERROR | %s", str(e))
        return None


# ------------------------------------------------------------------
# BACKGROUND PROCESSING (ASYNC)
# ------------------------------------------------------------------
def background_processing(image_bytes, name, email, phone, tone):
    """
    Background thread: Save image to Shopify Files and create/update customer.
    Runs after API response is sent to user.
    """
    try:
        log.info("BACKGROUND_PROCESSING_START | email=%s | tone=%s", email, tone)
        
        # Process customer and file in parallel-like manner
        customer_id = create_or_get_customer(name, email, phone, tone) if email and name and phone else None
        file_url = upload_image_to_shopify_files(image_bytes, name, tone)
        
        log.info("BACKGROUND_PROCESSING_COMPLETE | customer_id=%s | file_url=%s", customer_id, file_url)
    
    except Exception as e:
        log.error("BACKGROUND_PROCESSING_ERROR | %s", str(e))


# ------------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/detect-tone", methods=["POST"])
def detect_tone():
    """
    Fast endpoint: Classifies tone and returns immediately.
    Background thread handles image upload and customer creation.
    """
    log.info("REQUEST_RECEIVED")

    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"tone": None}), 200

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip()
    phone = (data.get("phone") or "").strip()

    # Decode and validate image
    try:
        img_bytes = base64.b64decode(data["image"])
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return jsonify({"tone": None}), 200

    # Resize image
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
    if img.size[0] < MIN_DIM or img.size[1] < MIN_DIM:
        return jsonify({"tone": None}), 200

    # Encode to JPEG
    out = BytesIO()
    img.save(out, format="JPEG", quality=95, subsampling=0)

    # Call Gemini API
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

    # Parse response
    parsed = json.loads(response.text or "{}")
    tone = parsed.get("tone")

    log.info("CLASSIFICATION | tone=%s", tone)

    # ===== RETURN IMMEDIATELY TO USER =====
    response_to_user = {"tone": tone}
    
    # ===== THEN START BACKGROUND PROCESSING =====
    # Start background thread for customer creation and file upload
    # User doesn't have to wait for these to complete
    if name and email and phone:
        thread = threading.Thread(
            target=background_processing,
            args=(out.getvalue(), name, email, phone, tone),
            daemon=True
        )
        thread.start()
        log.info("BACKGROUND_THREAD_STARTED | email=%s", email)

    return jsonify(response_to_user), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)