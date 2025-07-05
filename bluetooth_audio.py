import pyttsx3
import platform

# Initialize Text-to-Speech (TTS) engine
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  # Adjust speech speed
tts_engine.setProperty('volume', 1.0)  # Max volume

def speak(text):
    """Convert text to speech on Windows."""
    print(f"Speaking: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# Test speech output
if __name__ == "__main__":
    if platform.system() == "Windows":
        speak("Text-to-Speech is working on Windows.")
    else:
        speak("Bluetooth audio enabled on Raspberry Pi.")
