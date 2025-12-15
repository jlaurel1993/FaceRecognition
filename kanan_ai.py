#!/usr/bin/env python3
# Kanan AI – Faces + OCR + Objects (no "person", no confidence ratios)
# Local (Pi): camera + faces + TTS (Piper)
# Cloud (EC2): OCR + object detection via FastAPI (/analyze)

import os, cv2, time, json, queue, threading, requests, numpy as np, re, subprocess, shutil, tempfile
import face_recognition, datetime, sounddevice as sd
from vosk import Model as VoskModel, KaldiRecognizer

# ---------- CONFIG ----------
AWS_URL = None  # Cloud endpoint disabled in public repo
BASE_DIR = os.path.expanduser("~/FaceRecognition")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "uploads")
VOSK_DIR = os.path.join(BASE_DIR, "vosk-model")

# Piper config
PIPER_BIN = shutil.which("piper") or "piper"
PIPER_MODEL = os.path.join(BASE_DIR, "piper_models", "en_US-ryan-medium.onnx")
PIPER_LENGTH_SCALE = "1.1"
PIPER_SILENCE = "0.25"

OBJECT_INTERVAL = 4.0
TIMEOUT = 5.0
FACE_COOLDOWN = 5
OBJECT_COOLDOWN = 6
TEXT_COOLDOWN = 8

# ---------- BATTERY (PLACEHOLDER) ----------
def get_battery_status():
    """
    Placeholder battery function for testing.
    Replace later with real hardware logic (INA219, Pico, etc.).
    """
    return 75  # fixed test percentage

# ---------- TTS (Piper) ----------
_tts_q = queue.Queue()

def _piper_available() -> bool:
    if shutil.which(PIPER_BIN) is None:
        print("[TTS] Piper binary not found in PATH.")
        return False
    if not os.path.isfile(PIPER_MODEL):
        print(f"[TTS] Piper model missing: {PIPER_MODEL}")
        return False
    return True

def _piper_say(text: str):
    if not _piper_available():
        print(f"Kanan (text only): {text}")
        return
    text = text.strip()
    if not text:
        return

    sentences = re.split(r'(?<=[.!?]) +', text)
    for chunk in sentences:
        if not chunk.strip():
            continue
        os.makedirs("/tmp/kanan_tts", exist_ok=True)
        with tempfile.NamedTemporaryFile(dir="/tmp/kanan_tts", suffix=".wav", delete=False) as tmpf:
            wav_path = tmpf.name
        try:
            proc = subprocess.run(
                [PIPER_BIN, "--model", PIPER_MODEL,
                 "--output_file", wav_path,
                 "--length_scale", PIPER_LENGTH_SCALE,
                 "--sentence_silence", PIPER_SILENCE],
                input=chunk.encode("utf-8"),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if proc.returncode != 0:
                print(f"[TTS] Piper failed.")
                if os.path.exists(wav_path):
                    os.remove(wav_path)
                continue
            time.sleep(0.2)
            subprocess.run(
                ["aplay", wav_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except Exception as e:
            print(f"[TTS ERROR] {e}")
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)
        time.sleep(0.4)
    time.sleep(0.5)

def _tts_loop():
    if _piper_available():
        print(f"🔊 Piper ready • Model: {os.path.basename(PIPER_MODEL)}")
    else:
        print("🔇 Piper not available")
    while True:
        msg = _tts_q.get()
        if msg is None:
            break
        _piper_say(msg)

threading.Thread(target=_tts_loop, daemon=True).start()

def speak(msg: str):
    msg = msg.strip()
    if msg:
        print(f"Kanan: {msg}")
        _tts_q.put(msg)

# ---------- CAMERA ----------
class Camera:
    def __init__(self):
        self.cap = None
        self.frame = None
        self.running = False
        self.device_index = None

    def find_camera(self):
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self.device_index = i
                    print(f"✅ Camera {i} active (/dev/video{i})")
                    return cap
                cap.release()
        print("❌ No working camera found.")
        return None

    def open(self):
        self.cap = self.find_camera()
        return self.cap is not None

    def start(self):
        self.running = True

        def loop():
            fail = 0
            while self.running:
                if not self.cap:
                    self.cap = self.find_camera()
                    time.sleep(1)
                    continue
                ret, f = self.cap.read()
                if not ret:
                    fail += 1
                    if fail >= 5:
                        print("⚠️ Camera feed lost. Reconnecting...")
                        self.cap.release()
                        time.sleep(3)
                        self.cap = self.find_camera()
                        if self.cap:
                            speak("Camera reconnected")
                        fail = 0
                        continue
                    continue
                fail = 0
                self.frame = f
                time.sleep(0.01)

        threading.Thread(target=loop, daemon=True).start()

    def read(self):
        return self.frame

    def close(self):
        self.running = False
        if self.cap:
            self.cap.release()

cam = Camera()
if not cam.open():
    speak("No camera detected.")
    raise SystemExit(1)
cam.start()

# ---------- FACES ----------
known_faces = []
known_names = []
faces_lock = threading.Lock()

def rebuild_faces():
    global known_faces, known_names
    with faces_lock:
        known_faces = []
        known_names = []
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        for fn in os.listdir(KNOWN_FACES_DIR):
            if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                img = face_recognition.load_image_file(
                    os.path.join(KNOWN_FACES_DIR, fn)
                )
                enc = face_recognition.face_encodings(img)
                if enc:
                    known_faces.append(enc[0])
                    known_names.append(os.path.splitext(fn)[0])
        print(f"📂 {len(known_names)} faces loaded.")

rebuild_faces()

def check_faces(frame):
    try:
        with faces_lock:
            if not known_faces:
                return []
            small = cv2.resize(frame, (320, 240))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)
            out = []
            for e, loc in zip(encs, locs):
                matches = face_recognition.compare_faces(known_faces, e, 0.5)
                if True in matches:
                    out.append((known_names[matches.index(True)], loc))
            return out
    except Exception as e:
        print("[FaceDetection ERROR]", e)
        return []

# ---------- CLOUD ----------
last_send = 0
object_recognition_enabled = True

def cloud_detect(frame):
    global last_send
    now = time.time()
    if now - last_send < OBJECT_INTERVAL:
        return None
    last_send = now
    try:
        _, buf = cv2.imencode(".jpg", cv2.resize(frame, (480, 360)))
        r = requests.post(
            AWS_URL,
            files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            timeout=TIMEOUT
        )
        if r.status_code == 200:
            return r.json()
        else:
            print("[Cloud] HTTP", r.status_code)
    except Exception as e:
        print("[Cloud]", e)
    return None

# ---------- SNAPSHOT ----------
def take_snapshot(frame, name=None):
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    filename = (
        f"{name.strip().replace(' ', '_')}.jpg"
        if name else f"picture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    )
    path = os.path.join(KNOWN_FACES_DIR, filename)
    cv2.imwrite(path, frame)
    time.sleep(0.3)
    speak(f"Picture saved as {filename}")
    time.sleep(0.5)
    rebuild_faces()
    if name:
        speak(f"New face {name.strip()} added successfully")
    return path

# ---------- VOICE ----------
vosk_model = VoskModel(VOSK_DIR)
rec = KaldiRecognizer(vosk_model, 16000)
_audio_q = queue.Queue()

def _audio_cb(indata, frames, time_info, status):
    if status:
        print(status)
    _audio_q.put(bytes(indata))

# ---------- VOICE THREAD ----------
def _voice_thread():
    global object_recognition_enabled
    with sd.RawInputStream(
        samplerate=16000,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=_audio_cb
    ):
        speak("Hi my name is Kanan, I am your visual assistant.")
        while True:
            data = _audio_q.get()
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "").strip().lower()
                if not text:
                    continue
                cmd = text

                if "what time" in cmd:
                    speak(datetime.datetime.now().strftime("%I:%M %p"))

                elif "rebuild" in cmd:
                    rebuild_faces()
                    speak("Faces refreshed")

                # ----- BATTERY STATUS -----
                # IMPORTANT: no plain "battery" here, to avoid self-trigger!
                elif ("battery status" in cmd or
                      "battery level" in cmd or
                      "what is my battery" in cmd or
                      "whats my battery" in cmd or
                      "what's my battery" in cmd or
                      "power level" in cmd):
                    level = get_battery_status()
                    speak(f"Your battery is at {level} percent.")

                # ---------- ONLY CHANGE YOU REQUESTED ----------
                # removed "hey kanan" entirely
                elif ("picture" in cmd and ("name it" in cmd or "name" in cmd)):
                    frame = cam.read()
                    if frame is None:
                        speak("Camera not ready")
                        continue

                    if "name it" in cmd:
                        parts = cmd.split("name it")
                    else:
                        parts = cmd.split("name")

                    name = parts[1].strip() if len(parts) > 1 else "picture"
                    take_snapshot(frame, name)
                # ------------------------------------------------

                elif "stop object recognition" in cmd:
                    object_recognition_enabled = False
                    speak("Object recognition disabled, but I will continue reading text.")

                elif "start object recognition" in cmd:
                    object_recognition_enabled = True
                    speak("Object recognition enabled.")

                elif "shutdown" in cmd or "exit" in cmd:
                    speak("Shutting down")
                    _tts_q.put(None)
                    os._exit(0)

threading.Thread(target=_voice_thread, daemon=True).start()

# ---------- MAIN LOOP ----------
last_face_announce = {}
last_object_announce = 0
last_text_announce = 0

print("🚀 Kanan AI started (Faces + OCR + Objects except 'person'). Press 'q' to quit.")
try:
    while True:
        f = cam.read()
        if f is None:
            time.sleep(0.01)
            continue

        faces = check_faces(f)
        for name, loc in faces:
            top, right, bottom, left = [v * 4 for v in loc]
            cv2.rectangle(f, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                f, name,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            now = time.time()
            if now - last_face_announce.get(name, 0) > FACE_COOLDOWN:
                speak(f"{name} is there")
                last_face_announce[name] = now

        cloud_data = cloud_detect(f)
        if cloud_data:
            objs = cloud_data.get("objects", [])
            text = cloud_data.get("text", "")
            now = time.time()

            if object_recognition_enabled:
                clean = [
                    re.sub(r'\s*\([^)]*\)', '', o).strip()
                    for o in objs
                    if o and "person" not in o.lower()
                ]
                if clean and now - last_object_announce > OBJECT_COOLDOWN:
                    speak("I see " + ", ".join(clean))
                    last_object_announce = now

            if text and now - last_text_announce > TEXT_COOLDOWN:
                clean = text.replace('\n', ' ').strip()
                if clean:
                    speak("Text says: " + clean)
                    last_text_announce = now

        cv2.imshow("Kanan View", f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass
finally:
    cam.close()
    _tts_q.put(None)
    cv2.destroyAllWindows()
