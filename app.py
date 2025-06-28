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
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from roboflow import Roboflow
import supervision as sv
import requests
import subprocess

from flask import Flask, request, jsonify
from flask_cors import CORS

# Load env variables
load_dotenv(dotenv_path=".env.local")

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# PostgreSQL setup
conn = psycopg2.connect(
    host=os.getenv("PGHOST"),
    dbname=os.getenv("PGDATABASE"),
    user=os.getenv("PGUSER"),
    password=os.getenv("PGPASSWORD"),
    sslmode='require'
)
cursor = conn.cursor()

reader = easyocr.Reader(['en'], gpu=True)
plate_cooldown = {}
COOLDOWN_SECONDS = 1800

label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

def save_plate_to_db(plate_text: str):
    try:
        timestamp = datetime.now()
        cursor.execute(
            "SELECT detected_at FROM number_plates WHERE plate_number = %s ORDER BY detected_at DESC LIMIT 1",
            (plate_text,)
        )
        result = cursor.fetchone()
        if result:
            last_detected_at = result[0]
            elapsed_minutes = (timestamp - last_detected_at).total_seconds() / 60
            if elapsed_minutes >= 30:
                cursor.execute(
                    "INSERT INTO number_plates (plate_number, detected_at) VALUES (%s, %s)",
                    (plate_text, timestamp)
                )
                conn.commit()
        else:
            cursor.execute(
                "INSERT INTO number_plates (plate_number, detected_at) VALUES (%s, %s)",
                (plate_text, timestamp)
            )
            conn.commit()
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

def my_custom_sink(predictions: dict, video_frame: VideoFrame, video_writer: cv2.VideoWriter):
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
                    save_plate_to_db(plate_text)

    video_writer.write(annotated_image)

def run_detection_pipeline(input_video_path, output_path):
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

# Flask App
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Detection backend is working."})

@app.route("/process-video", methods=["POST"])
def process_video():
    file = request.files.get("video")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    output_path = os.path.join(PROCESSED_FOLDER, f"processed_{file.filename}")
    file.save(input_path)
    run_detection_pipeline(input_path, output_path)
    return jsonify({"processed_video_url": f"/{output_path}"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
