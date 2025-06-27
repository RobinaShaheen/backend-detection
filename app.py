import os
import re
import cv2
import time
import logging
import easyocr
import psycopg2
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from inference import InferencePipeline  # type: ignore
from inference.core.interfaces.camera.entities import VideoFrame  # type: ignore
from roboflow import Roboflow  # type: ignore
import supervision as sv  # type: ignore
import requests
import subprocess
# from collections import defaultdict

# Load .env.local file
load_dotenv(dotenv_path=".env.local")

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# PostgreSQL DB connection setup
conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    sslmode='require'
)
cursor = conn.cursor()

# OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Track seen plates to avoid duplicates
plate_cooldown = {}
COOLDOWN_MINUTES = 30  # minimum time between same plate saves
COOLDOWN_SECONDS = COOLDOWN_MINUTES * 60  # minimum time between same plate saves

# Annotators for visualization
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()


# def download_video_from_vercel(url: str, output_path: str):
#     logging.info(f"Downloading video from {url}")
#     response = requests.get(url, stream=True)
#     with open(output_path, 'wb') as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#     logging.info(f"Video downloaded to {output_path}")

# Function to save plate to DB (skip if already exists today)
def save_plate_to_db(plate_text: str):
    try:
        timestamp = datetime.now()

        # Check if plate exists and get last detected time
        cursor.execute(
            "SELECT detected_at FROM number_plates WHERE plate_number = %s ORDER BY detected_at DESC LIMIT 1",
            (plate_text,)
        )
        result = cursor.fetchone()

        if result:
            last_detected_at = result[0]
            elapsed_minutes = (timestamp - last_detected_at).total_seconds() / 60

            if elapsed_minutes >= COOLDOWN_MINUTES:
                cursor.execute(
                    "INSERT INTO number_plates (plate_number, detected_at) VALUES (%s, %s)",
                    (plate_text, timestamp)
                )
                conn.commit()
                logging.info(f"Saved to DB (after {elapsed_minutes:.1f} min): {plate_text}")
            else:
                logging.info(f"Duplicate skipped (only {elapsed_minutes:.1f} min): {plate_text}")
        else:
            # First time this plate ever detected
            cursor.execute(
                "INSERT INTO number_plates (plate_number, detected_at) VALUES (%s, %s)",
                (plate_text, timestamp)
            )
            conn.commit()
            logging.info(f"Saved to DB (first time today): {plate_text}")

    except Exception as db_error:
        logging.error(f"Database error: {db_error}")

def clean_plate_text(text):
    text = re.sub(r'[^A-Za-z0-9]', '', text).upper()
    match = re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$', text)
    return match.group() if match else None

def preprocess_plate_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def should_save_plate(plate_text):
    now = time.time()
    last_seen = plate_cooldown.get(plate_text)
    if last_seen and (now - last_seen) < COOLDOWN_SECONDS:
        return False
    plate_cooldown[plate_text] = now
    return True

def cleanup_old_votes():
    now = time.time()
    expired = [plate for plate, t in plate_vote_time.items() if (now - t) > MAX_VOTE_AGE_SECONDS] # type: ignore
    for plate in expired:
        plate_vote_counter.pop(plate, None) # type: ignore
        plate_vote_time.pop(plate, None) # type: ignore
        
# Custom sink function to process predictions
def my_custom_sink(predictions: dict, video_frame: VideoFrame, video_writer: cv2.VideoWriter):
    try:
        detections = sv.Detections.from_inference(predictions)
        labels = [p["class"] for p in predictions["predictions"]]

        annotated_image = label_annotator.annotate(
            scene=video_frame.image.copy(), detections=detections, labels=labels
        )
        annotated_image = box_annotator.annotate(annotated_image, detections=detections)

        for det in predictions["predictions"]:
            x, y, width, height = int(det["x"]), int(det["y"]), int(det["width"]), int(det["height"])
            xmin, ymin, xmax, ymax = x - width // 2, y - height // 2, x + width // 2, y + height // 2

            plate_roi = video_frame.image[ymin:ymax, xmin:xmax]
            if plate_roi.size > 0:
                preprocessed = preprocess_plate_image(plate_roi)
                result = reader.readtext(preprocessed)

                if result:
                    raw_text = result[0][1]
                    plate_text = clean_plate_text(raw_text)

                    if plate_text and should_save_plate(plate_text):
                        logging.info(f"Valid plate detected: {plate_text}")
                        save_plate_to_db(plate_text)
                    else:
                        logging.info(f"Ignored or duplicate plate: {raw_text} => {plate_text}")

        cv2.imshow("Predictions", annotated_image)
        cv2.waitKey(1)
        video_writer.write(annotated_image)

    except Exception as e:
        logging.error(f"Error in custom sink: {e}")

def upload_to_vercel_blob(local_path: str, blob_path: str):
    logging.info(f"Uploading {local_path} to Vercel Blob: {blob_path}")
    result = subprocess.run(["vercel", "blob", "upload", f"{local_path}:{blob_path}"], capture_output=True, text=True)
    if result.returncode == 0:
        logging.info("Upload successful")
        logging.info(result.stdout)
    else:
        logging.error("Upload failed")
        logging.error(result.stderr)

# === Main execution ===
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/process-video", methods=["POST"])
def process_video():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")
    file.save(input_path)

    # Call your existing detection logic
    run_detection_pipeline(input_path, output_path)

    return jsonify({"processed_video_url": f"http://localhost:5000/{output_path}"})
# ({"processed_video_url": f"/{output_path}"})

def run_detection_pipeline(input_video_path, output_path):
    # --- Copy your detection logic here ---
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (frame_width, frame_height))

    pipeline = InferencePipeline.init(
        model_id="vehiclenumberplate/4",
        video_reference=input_video_path,
        on_prediction=lambda preds, frame: my_custom_sink(preds, frame, video_writer),
        api_key=os.getenv("ROBOFLOW_API_KEY")
    )

    pipeline.start()
    pipeline.join()
    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    logging.info("Detection pipeline complete.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))