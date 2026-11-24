import base64
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from pose_engine import PoseProcessor

app = Flask(__name__)

# Satu instance MediaPipe dipakai bergantian oleh request frame.
processor = PoseProcessor()
processor_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_frame", methods=["POST"])
def process_frame():
    if "frame" not in request.files:
        return jsonify({"error": "Frame tidak ditemukan."}), 400

    file_bytes = request.files["frame"].read()
    np_buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Frame tidak dapat dibaca."}), 400

    # Batasi ukuran pemrosesan agar laptop tidak terbebani.
    height, width = frame.shape[:2]
    max_width = 640
    if width > max_width:
        new_height = int(height * max_width / width)
        frame = cv2.resize(frame, (max_width, new_height))

    try:
        with processor_lock:
            processed, people, labels, fps = processor.process_frame(frame)

        success, encoded = cv2.imencode(
            ".jpg", processed, [cv2.IMWRITE_JPEG_QUALITY, 75]
        )
        if not success:
            return jsonify({"error": "Gagal mengubah hasil menjadi JPEG."}), 500

        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        return jsonify({
            "image": image_b64,
            "people": people,
            "labels": labels,
            "fps": round(fps, 1)
        })

    except Exception as error:
        app.logger.exception("Kesalahan saat memproses frame")
        return jsonify({"error": str(error)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "online"})


if __name__ == "__main__":
    # Menggunakan sertifikat mkcert yang sudah dibuat pada tahap HTTPS.
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        ssl_context=("cert.pem", "key.pem")
    )
