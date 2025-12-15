# Kanan Vision AI – Face Recognition for AI Glasses

Kanan Vision AI is a senior design project focused on assisting visually impaired users using on-device artificial intelligence.  
The system is designed to run locally on a Raspberry Pi and provides real-time face recognition, voice interaction, and system feedback.

## Features
- Real-time face recognition using a camera
- Offline voice commands using Vosk
- Text-to-Speech feedback using Piper
- Battery status voice command
- Modular architecture for vision, audio, and system components

## Architecture
- `kanan_ai.py` – Main orchestration layer coordinating vision, audio, and system modules
- `image_processing.py` – Image and face processing logic
- `battery_monitor.py` – Battery monitoring functionality
- `bluetooth_audio.py` – Audio output via Bluetooth
- `server.py` – Local API / service layer

## Tech Stack
- Python
- OpenCV
- face_recognition
- Vosk (offline speech recognition)
- Piper (offline text-to-speech)
- Linux / Raspberry Pi

## Notes
- The system prioritizes **local processing** for reliability and privacy
- Cloud-based components and indoor navigation are intentionally excluded from this public repository
