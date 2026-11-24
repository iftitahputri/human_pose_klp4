# Human Pose Estimation

Aplikasi Human Pose Estimation berbasis web untuk mendeteksi dan mengklasifikasikan gerakan tubuh manusia secara real-time menggunakan kamera perangkat. Sistem memanfaatkan MediaPipe Pose Landmarker untuk mendeteksi landmark tubuh, OpenCV untuk pengolahan frame dan visualisasi skeleton, serta Flask sebagai backend pemrosesan.

Kamera diakses melalui browser pada perangkat pengguna (misalnya HP), sedangkan proses deteksi dan klasifikasi dilakukan pada laptop yang menjalankan server Flask.

## Fitur Utama

- **Real-time Pose Detection** — Mendeteksi landmark tubuh manusia dari video kamera.
- **Multi-person Detection** — Mendukung pendeteksian hingga 3 orang dalam satu frame.
- **Skeleton Visualization** — Menampilkan titik landmark dan garis penghubung antartitik tubuh.
- **Pose Classification** — Mengklasifikasikan gerakan tubuh menjadi:
  - Standing (berdiri)
  - Squatting (jongkok)
  - Sitting (duduk)
  - Bending (membungkuk)
- **Detection Status** — Menampilkan status deteksi pose.
- **FPS Monitoring** — Menampilkan FPS untuk memantau performa pemrosesan.
- **Camera Switching** — Mendukung pergantian kamera depan dan belakang.
- **Web-based Interface** — Antarmuka dapat diakses melalui browser perangkat yang terhubung ke jaringan yang sama.

## Arsitektur Sistem

Sistem menggunakan pendekatan client-server. Perangkat HP berperan sebagai client yang menangkap video dan menampilkan hasil, sedangkan laptop berperan sebagai server yang menjalankan proses deteksi pose.

### Alur Pemrosesan

1. **Input:** Kamera perangkat diakses melalui browser menggunakan JavaScript `getUserMedia()`.
2. **Frame Transmission:** Frame video dikirim dari browser ke backend Flask melalui jaringan Wi-Fi.
3. **Preprocessing:** Frame diterima dan diproses menggunakan OpenCV, termasuk penyesuaian format warna jika diperlukan.
4. **Pose Detection:** MediaPipe Pose Landmarker mendeteksi landmark tubuh manusia.
5. **Pose Classification:** Landmark dianalisis untuk menentukan kategori gerakan tubuh.
6. **Visualization:** OpenCV menggambar titik landmark dan garis penghubung untuk membentuk skeleton.
7. **Output:** Hasil pemrosesan dikirim kembali ke browser untuk ditampilkan bersama status deteksi, klasifikasi gerakan, jumlah orang terdeteksi, dan FPS.

## Persyaratan

Sebelum menjalankan aplikasi, pastikan perangkat sudah memiliki:

- Python 3.10
- pip
- Browser modern yang mendukung akses kamera
- Koneksi Wi-Fi yang sama antara HP dan laptop
- `mkcert` untuk membuat sertifikat HTTPS lokal

## Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/iftitahputri/human_pose_klp4.git
```

Masuk ke direktori proyek:

```bash
cd human_pose_klp4
```

### 2. Buat Virtual Environment

```bash
python -m venv .venv
```

Aktifkan virtual environment pada Windows:

```powershell
.venv\Scripts\activate
```

### 3. Instal Dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan Model

Pastikan file model `pose_landmarker_lite.task` tersedia di dalam direktori:

```text
models/pose_landmarker_lite.task
```

Model ini digunakan oleh MediaPipe Pose Landmarker untuk mendeteksi landmark tubuh manusia.

## Konfigurasi HTTPS

Aplikasi menggunakan HTTPS agar browser dapat mengakses kamera melalui `getUserMedia()` ketika dibuka dari perangkat lain melalui jaringan lokal.

Sertifikat HTTPS dibuat secara lokal menggunakan `mkcert`.

### 1. Instalasi mkcert

Unduh dan instal mkcert sesuai sistem operasi melalui dokumentasi resminya:

https://github.com/FiloSottile/mkcert

### 2. Instal Local Certificate Authority

```bash
mkcert -install
```

### 3. Buat Sertifikat

Masuk ke direktori `web_app`:

```bash
cd web_app
```

Buat sertifikat menggunakan alamat IP laptop:

```bash
mkcert localhost 127.0.0.1 ::1 192.168.1.11
```

Ganti `192.168.1.11` dengan alamat IP laptop yang digunakan.

Simpan atau ubah nama file sertifikat yang dihasilkan menjadi:

```text
cert.pem
key.pem
```

Letakkan kedua file tersebut di dalam direktori `web_app`.

**Penting:** File sertifikat dan private key tidak disertakan dalam repository. Setiap pengguna perlu membuat sertifikat HTTPS sendiri untuk lingkungan lokalnya.

## Menjalankan Aplikasi

### 1. Masuk ke Direktori Web App

```bash
cd web_app
```

### 2. Jalankan Flask Server

```bash
python app.py
```

Server akan berjalan pada port `5000` dengan konfigurasi HTTPS.

### 3. Akses Melalui Browser

Pada laptop, buka:

```text
https://localhost:5000
```

Pada HP yang terhubung ke jaringan Wi-Fi yang sama, buka:

```text
https://192.168.1.11:5000
```

Ganti alamat IP tersebut sesuai alamat IP laptop.

Pastikan perangkat HP telah mempercayai sertifikat CA lokal yang digunakan. Jika browser menampilkan peringatan sertifikat, periksa kembali konfigurasi sertifikat dan kepercayaan CA.
