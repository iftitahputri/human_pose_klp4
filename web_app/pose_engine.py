
import cv2
import mediapipe as mp
from pathlib import Path
import time
import math
from collections import deque, Counter

# ==========================================
# 1. KONFIGURASI PROYEK
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "pose_landmarker_lite.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"File model tidak ditemukan: {MODEL_PATH}\n"
        "Pastikan model berada di folder models."
    )

MAX_POSES = 3
VISIBILITY_THRESHOLD = 0.5

# Warna skeleton dalam format BGRf
POSE_COLORS = [
    (0, 255, 0),      # Person 1: Hijau
    (255, 0, 0),      # Person 2: Biru
]

# Smoothing label agar tidak terlalu berubah-ubah
SMOOTHING_WINDOW = 5

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
    num_poses=MAX_POSES,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

connections = mp.tasks.vision.PoseLandmarksConnections
POSE_CONNECTIONS = connections.POSE_LANDMARKS

# ==========================================
# 3. LANDMARK INDEX
# ==========================================

# MediaPipe Pose memiliki 33 landmark.

NOSE = 0

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LEFT_WRIST = 15
RIGHT_WRIST = 16

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_KNEE = 25
RIGHT_KNEE = 26

LEFT_ANKLE = 27
RIGHT_ANKLE = 28

# ==========================================
# 4. FUNGSI GEOMETRI
# ==========================================

def calculate_angle(a, b, c):
    """
    Menghitung sudut ABC dalam derajat.

    Titik B merupakan titik pusat sudut.
    """

    ax, ay = a
    bx, by = b
    cx, cy = c

    vector_1 = (ax - bx, ay - by)
    vector_2 = (cx - bx, cy - by)

    dot_product = (
        vector_1[0] * vector_2[0]
        + vector_1[1] * vector_2[1]
    )

    magnitude_1 = math.sqrt(
        vector_1[0] ** 2 + vector_1[1] ** 2
    )

    magnitude_2 = math.sqrt(
        vector_2[0] ** 2 + vector_2[1] ** 2
    )

    if magnitude_1 == 0 or magnitude_2 == 0:
        return 0

    cosine = dot_product / (
        magnitude_1 * magnitude_2
    )

    # Menghindari error akibat floating point
    cosine = max(-1.0, min(1.0, cosine))

    angle = math.degrees(math.acos(cosine))

    return angle


def get_point(landmarks, landmark_index, aspect_ratio):
    landmark = landmarks[landmark_index]

    # Koreksi rasio gambar agar perhitungan sudut
    # tidak terdistorsi oleh aspect ratio webcam.
    return (
        landmark.x * aspect_ratio,
        landmark.y
    )


def get_visibility(landmarks, landmark_index):
    """
    Mengambil nilai visibility landmark.
    """

    return landmarks[landmark_index].visibility


def is_visible(landmarks, landmark_indices):
    """
    Memeriksa apakah semua landmark yang dibutuhkan
    memiliki visibility yang cukup.
    """

    for index in landmark_indices:
        if get_visibility(
            landmarks, index
        ) < VISIBILITY_THRESHOLD:
            return False

    return True

def get_pose_metrics(landmarks, frame_shape):
    height, width = frame_shape[:2]
    aspect_ratio = width / height

    required = [
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP,
        LEFT_KNEE, RIGHT_KNEE,
        LEFT_ANKLE, RIGHT_ANKLE
    ]

    if not is_visible(landmarks, required):
        return None

    ls = get_point(landmarks, LEFT_SHOULDER, aspect_ratio)
    rs = get_point(landmarks, RIGHT_SHOULDER, aspect_ratio)

    lh = get_point(landmarks, LEFT_HIP, aspect_ratio)
    rh = get_point(landmarks, RIGHT_HIP, aspect_ratio)

    lk = get_point(landmarks, LEFT_KNEE, aspect_ratio)
    rk = get_point(landmarks, RIGHT_KNEE, aspect_ratio)

    la = get_point(landmarks, LEFT_ANKLE, aspect_ratio)
    ra = get_point(landmarks, RIGHT_ANKLE, aspect_ratio)

    shoulder_center = (
        (ls[0] + rs[0]) / 2,
        (ls[1] + rs[1]) / 2
    )

    hip_center = (
        (lh[0] + rh[0]) / 2,
        (lh[1] + rh[1]) / 2
    )

    knee_center = (
        (lk[0] + rk[0]) / 2,
        (lk[1] + rk[1]) / 2
    )

    left_angle = calculate_angle(lh, lk, la)
    right_angle = calculate_angle(rh, rk, ra)

    avg_knee = (left_angle + right_angle) / 2

    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]

    torso_angle = math.degrees(
        math.atan2(abs(dx), abs(dy))
    )

    hip_knee_difference = (
        hip_center[1] - knee_center[1]
    )

    return {
        "left_knee": left_angle,
        "right_knee": right_angle,
        "average_knee": avg_knee,
        "torso": torso_angle,
        "hip_knee": hip_knee_difference
    }

# ==========================================
# 5. POSE CLASSIFICATION
# ==========================================

def classify_pose(landmarks, frame_shape):
    height, width = frame_shape[:2]
    aspect_ratio = width / height
    """
    Mengklasifikasikan pose berdasarkan posisi
    dan sudut landmark tubuh.

    Output:
    STANDING
    SITTING
    SQUATTING
    BENDING
    ARMS RAISED
    UNKNOWN
    """

    # --------------------------------------
    # Ambil koordinat landmark
    # --------------------------------------

    left_shoulder = get_point(
        landmarks, LEFT_SHOULDER, aspect_ratio
    )

    right_shoulder = get_point(
        landmarks, RIGHT_SHOULDER, aspect_ratio
    )

    left_hip = get_point(
        landmarks, LEFT_HIP, aspect_ratio
    )

    right_hip = get_point(
        landmarks, RIGHT_HIP, aspect_ratio
    )

    left_knee = get_point(
        landmarks, LEFT_KNEE, aspect_ratio
    )

    right_knee = get_point(
        landmarks, RIGHT_KNEE, aspect_ratio
    )

    left_ankle = get_point(
        landmarks, LEFT_ANKLE, aspect_ratio
    )

    right_ankle = get_point(
        landmarks, RIGHT_ANKLE, aspect_ratio
    )

    left_wrist = get_point(
        landmarks, LEFT_WRIST, aspect_ratio
    )

    right_wrist = get_point(
        landmarks, RIGHT_WRIST, aspect_ratio
    )

    # --------------------------------------
    # Hitung titik tengah tubuh
    # --------------------------------------

    shoulder_center = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )

    hip_center = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    )

    knee_center = (
        (left_knee[0] + right_knee[0]) / 2,
        (left_knee[1] + right_knee[1]) / 2
    )
        # Jarak vertikal pinggul terhadap lutut
    hip_knee_difference = (
        hip_center[1] - knee_center[1]
    )
    
    # Normalisasi terhadap panjang kaki
    leg_length = abs(
        (
            (left_ankle[1] + right_ankle[1]) / 2
        ) - hip_center[1]
    )
    
    if leg_length > 0:
        hip_knee_ratio = (
            hip_knee_difference / leg_length
        )
    else:
        hip_knee_ratio = 0
    
    # --------------------------------------
    # Periksa landmark yang diperlukan
    # --------------------------------------

    required_landmarks = [
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_HIP,
        RIGHT_HIP,
        LEFT_KNEE,
        RIGHT_KNEE,
        LEFT_ANKLE,
        RIGHT_ANKLE
    ]

    if not is_visible(
        landmarks, required_landmarks
    ):
        return "UNKNOWN"

    # --------------------------------------
    # Hitung sudut lutut
    # --------------------------------------

    left_knee_angle = calculate_angle(
        left_hip,
        left_knee,
        left_ankle
    )

    right_knee_angle = calculate_angle(
        right_hip,
        right_knee,
        right_ankle
    )

    average_knee_angle = (
        left_knee_angle + right_knee_angle
    ) / 2

    # --------------------------------------
    # Hitung kemiringan torso
    # --------------------------------------

    dx = shoulder_center[0] - hip_center[0]
    dy = shoulder_center[1] - hip_center[1]

    torso_angle = math.degrees(
        math.atan2(abs(dx), abs(dy))
    )

    # 0 derajat = torso tegak
    # Semakin besar = semakin condong

    # --------------------------------------
    # 1. ARMS RAISED
    # --------------------------------------

    arms_visible = is_visible(
        landmarks,
        [
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_WRIST,
            RIGHT_WRIST
        ]
    )

    if arms_visible:

        left_arm_raised = (
            left_wrist[1] < left_shoulder[1]
        )

        right_arm_raised = (
            right_wrist[1] < right_shoulder[1]
        )

        if left_arm_raised or right_arm_raised:
            return "ARMS RAISED"

    # --------------------------------------
    # 2. SQUATTING
    # --------------------------------------

    # Lutut menekuk dan pinggul turun
    # relatif terhadap posisi lutut.

    if (
        average_knee_angle < 125
        and hip_center[1] >= knee_center[1]
    ):
        return "SQUATTING"

    # --------------------------------------
    # 3. SITTING
    # --------------------------------------

    # Lutut menekuk dan pinggul masih
    # relatif berada di atas lutut.

    if (
        75 <= average_knee_angle <= 145
        and hip_center[1] < knee_center[1]
    ):
        return "SITTING"

    # --------------------------------------
    # 4. BENDING
    # --------------------------------------

    if (
        torso_angle > 35
        and average_knee_angle > 140
    ):
        return "BENDING"

    # --------------------------------------
    # 5. STANDING
    # --------------------------------------

    if (
        average_knee_angle > 150
        and torso_angle < 30
    ):
        return "STANDING"

    # --------------------------------------
    # 6. UNKNOWN
    # --------------------------------------

    return "UNKNOWN"


# ==========================================
# 6. SMOOTHING LABEL
# ==========================================

# Riwayat klasifikasi setiap index pose.
pose_histories = [
    deque(maxlen=SMOOTHING_WINDOW)
    for _ in range(MAX_POSES)
]


def smooth_pose(pose_label, person_index):
    """
    Mengambil label yang paling sering muncul
    dalam beberapa frame terakhir.
    """

    history = pose_histories[person_index]

    history.append(pose_label)

    most_common = Counter(history).most_common(1)

    return most_common[0][0]


# ==========================================
# 7. FUNGSI GAMBAR SKELETON
# ==========================================

def draw_skeleton(
    frame,
    landmarks,
    color
):
    """
    Menggambar garis dan titik landmark
    menggunakan warna yang diberikan.
    """

    height, width = frame.shape[:2]

    # --------------------------------------
    # Gambar garis skeleton
    # --------------------------------------

    for connection in POSE_CONNECTIONS:

        start_idx = connection.start
        end_idx = connection.end

        start_point = landmarks[start_idx]
        end_point = landmarks[end_idx]

        if (
            start_point.visibility
            < VISIBILITY_THRESHOLD
            or end_point.visibility
            < VISIBILITY_THRESHOLD
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
                color,
                2
            )

    # --------------------------------------
    # Gambar titik landmark
    # --------------------------------------

    for landmark in landmarks:

        if landmark.visibility < VISIBILITY_THRESHOLD:
            continue

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if (
            0 <= x < width
            and 0 <= y < height
        ):
            cv2.circle(
                frame,
                (x, y),
                4,
                color,
                -1
            )


# ==========================================
# 8. FUNGSI MENAMPILKAN LABEL
# ==========================================

def draw_label(
    frame,
    landmarks,
    label,
    person_index,
    color
):
    """
    Menampilkan nama person dan pose
    di dekat bagian atas tubuh.
    """

    height, width = frame.shape[:2]

    # Cari posisi bahu untuk titik label
    left_shoulder = landmarks[LEFT_SHOULDER]
    right_shoulder = landmarks[RIGHT_SHOULDER]

    center_x = int(
        (
            left_shoulder.x
            + right_shoulder.x
        ) / 2 * width
    )

    center_y = int(
        (
            left_shoulder.y
            + right_shoulder.y
        ) / 2 * height
    )

    label_text = (
        f"Person {person_index + 1}: {label}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2

    text_size = cv2.getTextSize(
        label_text,
        font,
        font_scale,
        thickness
    )[0]

    text_x = max(
        10,
        min(
            center_x - text_size[0] // 2,
            width - text_size[0] - 10
        )
    )

    text_y = max(
        30,
        center_y - 30
    )

    # Background label
    cv2.rectangle(
        frame,
        (
            text_x - 5,
            text_y - text_size[1] - 8
        ),
        (
            text_x + text_size[0] + 5,
            text_y + 5
        ),
        (0, 0, 0),
        -1
    )

    # Teks label
    cv2.putText(
        frame,
        label_text,
        (text_x, text_y),
        font,
        font_scale,
        color,
        thickness
    )




class PoseProcessor:
    """Mengelola satu instance MediaPipe dan memproses frame dari browser."""

    def __init__(self):
        self._landmarker_context = PoseLandmarker.create_from_options(options)
        self.landmarker = self._landmarker_context.__enter__()
        self.start_time = time.perf_counter()
        self.last_timestamp_ms = -1
        self.fps_start = time.perf_counter()
        self.fps_frames = 0
        self.fps = 0.0

    def process_frame(self, frame):
        # Frame diterima dalam format BGR dari OpenCV.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int((time.perf_counter() - self.start_time) * 1000)
        timestamp_ms = max(timestamp_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        detected_count = 0
        labels = []

        if result.pose_landmarks:
            detected_count = len(result.pose_landmarks)

            for person_index, pose_landmarks in enumerate(result.pose_landmarks):
                color = POSE_COLORS[person_index % len(POSE_COLORS)]

                draw_skeleton(frame, pose_landmarks, color)

                raw_pose = classify_pose(pose_landmarks, frame.shape)
                final_pose = smooth_pose(raw_pose, person_index)
                labels.append({
                    "person": person_index + 1,
                    "pose": final_pose
                })

                draw_label(
                    frame, pose_landmarks, final_pose, person_index, color
                )

        for index in range(detected_count, MAX_POSES):
            pose_histories[index].clear()

        self.fps_frames += 1
        elapsed = time.perf_counter() - self.fps_start
        if elapsed >= 1.0:
            self.fps = self.fps_frames / elapsed
            self.fps_frames = 0
            self.fps_start = time.perf_counter()

        cv2.putText(
            frame, f"People detected: {detected_count}", (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )
        cv2.putText(
            frame, f"FPS: {self.fps:.1f}", (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        return frame, detected_count, labels, self.fps

    def close(self):
        self._landmarker_context.__exit__(None, None, None)
        for history in pose_histories:
            history.clear()
