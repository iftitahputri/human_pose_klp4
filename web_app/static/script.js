
const video = document.getElementById("camera");
const processedFrame = document.getElementById("processed-frame");
const placeholder = document.getElementById("camera-placeholder");
const startButton = document.getElementById("start-btn");
const stopButton = document.getElementById("stop-btn");
const switchButton = document.getElementById("switch-btn");
const cameraStatus = document.getElementById("camera-status");

const peopleElement = document.getElementById("people");
const poseElement = document.getElementById("pose");
const fpsElement = document.getElementById("fps");

const canvas = document.createElement("canvas");
const context = canvas.getContext("2d");

let cameraStream = null;
let processing = false;
let stopped = true;
let lastRequestTime = 0;
let facingMode = "user";

const FRAME_INTERVAL_MS = 120;

async function startCamera() {
    try {
        stopped = false;
        startButton.disabled = true;
        stopButton.disabled = false;
        switchButton.disabled = true;

        cameraStatus.textContent = "Meminta izin kamera...";

        const newStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: facingMode },
                width: { ideal: 480 },
                height: { ideal: 360 },
                frameRate: { ideal: 15, max: 20 }
            },
            audio: false
        });

        // Jika kamera dihentikan saat izin masih diminta.
        if (stopped) {
            newStream.getTracks().forEach(track => track.stop());
            return;
        }

        cameraStream = newStream;
        video.srcObject = cameraStream;
        await video.play();

        processedFrame.style.display = "block";
        processedFrame.style.visibility = "hidden";
        placeholder.style.display = "none";

        switchButton.disabled = false;
        cameraStatus.textContent =
            facingMode === "user"
                ? "Kamera depan aktif. Memproses frame..."
                : "Kamera belakang aktif. Memproses frame...";

        processLoop();

    } catch (error) {
        console.error(error);
        cameraStatus.textContent = `Gagal mengakses kamera: ${error.message}`;
        alert(`Kamera tidak dapat diakses:\n${error.message}`);

        stopCamera();
    }
}

async function switchCamera() {
    if (stopped || !cameraStream) return;

    // Hentikan pengiriman frame sebelum mengganti kamera.
    stopped = true;
    processing = false;

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;
    processedFrame.style.visibility = "hidden";

    facingMode = facingMode === "user" ? "environment" : "user";

    cameraStatus.textContent = "Mengganti kamera...";

    // Mulai ulang dengan kamera yang baru.
    await startCamera();
}

async function processLoop() {
    if (stopped || !cameraStream || processing) return;

    const now = Date.now();

    if (now - lastRequestTime < FRAME_INTERVAL_MS) {
        setTimeout(processLoop, 30);
        return;
    }

    if (!video.videoWidth || !video.videoHeight) {
        setTimeout(processLoop, 100);
        return;
    }

    processing = true;
    lastRequestTime = now;

    try {
        const maxWidth = 480;
        const scale = Math.min(1, maxWidth / video.videoWidth);

        canvas.width = Math.round(video.videoWidth * scale);
        canvas.height = Math.round(video.videoHeight * scale);

        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const blob = await new Promise(resolve =>
            canvas.toBlob(resolve, "image/jpeg", 0.65)
        );

        if (!blob) throw new Error("Gagal mengambil frame kamera.");

        // Cegah pengiriman frame lama setelah kamera diganti.
        if (stopped) return;

        const formData = new FormData();
        formData.append("frame", blob, "frame.jpg");

        const response = await fetch("/process_frame", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Server gagal memproses frame.");
        }

        // Abaikan hasil request lama jika kamera sudah dihentikan.
        if (stopped) return;

        processedFrame.src = `data:image/jpeg;base64,${result.image}`;
        processedFrame.style.visibility = "visible";

        peopleElement.textContent = result.people;
        fpsElement.textContent = result.fps;

        poseElement.textContent = result.labels.length
            ? result.labels.map(item => `P${item.person}: ${item.pose}`).join(" · ")
            : "Tidak terdeteksi";

        cameraStatus.textContent =
            facingMode === "user"
                ? "Kamera depan aktif · diproses oleh laptop"
                : "Kamera belakang aktif · diproses oleh laptop";

    } catch (error) {
        console.error(error);

        if (!stopped) {
            cameraStatus.textContent =
                `Koneksi/pemrosesan bermasalah: ${error.message}`;
        }
    } finally {
        processing = false;

        if (!stopped) {
            setTimeout(processLoop, 0);
        }
    }
}

function stopCamera() {
    stopped = true;
    processing = false;

    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }

    video.srcObject = null;

    processedFrame.removeAttribute("src");
    processedFrame.style.display = "none";
    processedFrame.style.visibility = "hidden";
    placeholder.style.display = "flex";

    startButton.disabled = false;
    stopButton.disabled = true;
    switchButton.disabled = true;

    peopleElement.textContent = "0";
    poseElement.textContent = "—";
    fpsElement.textContent = "—";

    cameraStatus.textContent = "Kamera berhenti.";
}

startButton.addEventListener("click", startCamera);
stopButton.addEventListener("click", stopCamera);
switchButton.addEventListener("click", switchCamera);

window.addEventListener("beforeunload", stopCamera);