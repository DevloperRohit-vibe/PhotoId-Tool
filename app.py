"""
PhotoID Studio — app.py
========================
Based on the original version, with targeted fixes for:

  1. ROTATION   — EXIF always fixed; tilt-correction only applies when
                  eyes are reliably detected AND angle is 2°–8°.
                  False positives from busy backgrounds (newspapers,
                  posters) are rejected via geometry validation.

  2. LIGHTING   — Analyses mean + std before touching anything.
                  Well-exposed photos (mean 90–175) are left alone.
                  Correction is subtle: lower CLAHE clip, no gamma on
                  normal photos.

  3. COLOR/SHARP — Vibrance-style saturation (not flat boost).
                   Sharpening reduced from radius=1.2/120% → 0.6/50%
                   so it adds presence without adding noise.
"""

import warnings, logging

# Silence noisy third-party loggers BEFORE any import
logging.getLogger("rembg").setLevel(logging.CRITICAL)
logging.getLogger("onnxruntime").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from main import process_image as pi


app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

# UPLOAD_FOLDER = "uploads"
# OUTPUT_FOLDER = "outputs"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
#  ROUTES
# ════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/app/api/process", methods=["POST"])
def process_image():
    response = pi()
    return response
# def process_image():


@app.route("/api/download/<sheet_id>")
def download_sheet(sheet_id):
    return send_from_directory(OUTPUT_FOLDER, f"sheet_{sheet_id}.jpg",
                               as_attachment=True,
                               download_name="id_photos.jpg")


if __name__ == "__main__":
    print("\n  📸  PhotoID Studio")
    print("  ────────────────────────────────────")
    print("Server Started ..")
    # print("  Press Ctrl+C to stop.\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
