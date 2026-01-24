import wave

path = "assets/voice-references/tars/tars-voice-sample-01.wav"

try:
    with wave.open(path, "rb") as f:
        print(f"File: {path}")
        print(f"Channels: {f.getnchannels()}")
        print(f"Sample Width: {f.getsampwidth()} bytes")
        print(f"Frame Rate: {f.getframerate()}")
        print(f"Frames: {f.getnframes()}")
        print(f"Comp Type: {f.getcomptype()} {f.getcompname()}")
except Exception as e:
    print(f"Error opening wav: {e}")
