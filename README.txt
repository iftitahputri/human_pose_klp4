TAHAP 3 - INTEGRASI KAMERA HP DENGAN MEDIAPIPE

Salin isi folder ini ke folder web_app project kamu (replace app.py, templates/index.html, static/script.js, static/style.css, dan tambahkan pose_engine.py).
Pastikan model tetap berada di ROOT PROJECT/models/pose_landmarker_lite.task.
Pastikan cert.pem dan key.pem berada di web_app/ (dibuat pada tahap HTTPS).
Install: pip install flask opencv-python mediapipe numpy
Jalankan dari folder web_app: python app.py
Buka di HP: https://IP-LAPTOP:5000

Catatan: proses menggunakan request HTTP berurutan, target awal sekitar 8-10 FPS. Jangan jalankan main.py bersamaan karena kamera laptop tidak digunakan dan instance model sudah dikelola oleh Flask.
