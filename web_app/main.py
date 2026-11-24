
import cv2
import mediapipe as mp
from pathlib import Path
import time

# ==========================================
# 1. KONFIGURASI PROYEK
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_PATH}\n"
        "Pastikan model berada di folder models."
    )

# ==========================================
# 2. INISIALISASI MEDIAPIPE
# ==========================================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=2,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

# Hubungan antar-titik tubuh yang akan digambar.
connections = mp.tasks.vision.PoseLandmarksConnections
POSE_CONNECTIONS = connections.POSE_LANDMARKS

# ==========================================
# 3. MEMBUKA WEBCAM
# ==========================================

# Indeks 0 biasanya webcam bawaan.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError(
        "Webcam tidak dapat dibuka. "
        "Periksa kamera dan izin akses Windows."
    )

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Pose Estimation berjalan.")
print("Tekan tombol Q pada jendela video untuk keluar.")

# ==========================================
# 4. PROSES DETEKSI POSE
# ==========================================

try:
    with PoseLandmarker.create_from_options(options) as landmarker:

        video_start_time = time.perf_counter()
        last_timestamp_ms = -1

        fps_start_time = time.perf_counter()
        fps_frame_count = 0
        fps = 0.0

        while True:
            success, frame = cap.read()

            if not success:
                print("Gagal membaca frame dari webcam.")
                break

            # Membalik gambar agar tampak seperti cermin.
            frame = cv2.flip(frame, 1)

            # OpenCV menggunakan BGR; MediaPipe menggunakan RGB.
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            # Konversi frame menjadi objek gambar MediaPipe.
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp harus meningkat pada setiap frame.
            timestamp_ms = int(
                (time.perf_counter() - video_start_time) * 1000
            )

            timestamp_ms = max(
                timestamp_ms,
                last_timestamp_ms + 1
            )

            last_timestamp_ms = timestamp_ms

            # Jalankan model untuk mendeteksi landmark.
            result = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )

            # ======================================
            # 5. GAMBAR TITIK DAN GARIS KERANGKA
            # ======================================

            
            if result.pose_landmarks:
                height, width = frame.shape[:2]

                for pose_landmarks in result.pose_landmarks:
                
                    # Gambar garis penghubung antar-landmark
                    for connection in POSE_CONNECTIONS:
                        start_idx = connection.start
                        end_idx = connection.end

                        start_point = pose_landmarks[start_idx]
                        end_point = pose_landmarks[end_idx]

                        if (
                            start_point.visibility < 0.5
                            or end_point.visibility < 0.5
                        ):
                            continue
                        
                        x1 = int(start_point.x * width)
                        y1 = int(start_point.y * height)
                        x2 = int(end_point.x * width)
                        y2 = int(end_point.y * height)

                        if (
                            0 <= x1 < width
                            and 0 <= y1 < height
                            and 0 <= x2 < width
                            and 0 <= y2 < height
                        ):
                            cv2.line(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )

                    # Gambar titik-titik landmark
                    for landmark in pose_landmarks:
                        if landmark.visibility < 0.5:
                            continue
                        
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(
                                frame,
                                (x, y),
                                4,
                                (0, 0, 255),
                                -1
                            )

                status = "Pose detected"
                status_color = (0, 255, 0)

            else:
                status = "No pose detected"
                status_color = (0, 0, 255)


            # ======================================
            # 6. INFORMASI PADA LAYAR
            # ======================================

            fps_frame_count += 1
            fps_elapsed = time.perf_counter() - fps_start_time
            
            if fps_elapsed >= 1.0:
                fps = fps_frame_count / fps_elapsed
                fps_frame_count = 0
                fps_start_time = time.perf_counter()

            cv2.putText(
                frame,
                status,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                status_color,
                2
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press Q to quit",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Human Pose Estimation", frame)

            # Keluar jika tombol Q ditekan.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

finally:
    # Selalu lepaskan webcam dan tutup jendela.
    cap.release()
    cv2.destroyAllWindows()

print("Program selesai.")