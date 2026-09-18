# import numpy as np
# import sounddevice as sd
# import librosa

# from kokoro import KPipeline

# # ==========================
# # KOKORO SETUP
# # ==========================

# pipeline = KPipeline(lang_code="a")

# VOICE = "af_heart"
# SAMPLE_RATE = 24000


# # ==========================
# # EMOTION DETECTION
# # ==========================

# def detect_emotion(text):
#     text = text.lower()

#     if any(word in text for word in [
#         "success", "completed", "downloaded",
#         "installed", "done", "finished"
#     ]):
#         return "happy"

#     if any(word in text for word in [
#         "wow", "awesome", "amazing",
#         "congratulations"
#     ]):
#         return "excited"

#     if any(word in text for word in [
#         "battery", "warning", "careful",
#         "attention", "low"
#     ]):
#         return "gentle"

#     if any(word in text for word in [
#         "error", "failed", "unable"
#     ]):
#         return "serious"

#     return "neutral"


# # ==========================
# # AUDIO PROCESSING
# # ==========================

# def apply_emotion(audio, emotion):

#     presets = {
#         "happy": {
#             "speed": 1.08,
#             "pitch": 2,
#             "volume": 1.05
#         },

#         "excited": {
#             "speed": 1.15,
#             "pitch": 4,
#             "volume": 1.20
#         },

#         "gentle": {
#             "speed": 0.95,
#             "pitch": -1,
#             "volume": 0.90
#         },

#         "serious": {
#             "speed": 0.98,
#             "pitch": -1,
#             "volume": 1.00
#         },

#         "neutral": {
#             "speed": 1.00,
#             "pitch": 0,
#             "volume": 1.00
#         }
#     }

#     cfg = presets.get(emotion, presets["neutral"])

#     # Speed change
#     audio = librosa.effects.time_stretch(
#         audio,
#         rate=cfg["speed"]
#     )

#     # Pitch shift
#     if cfg["pitch"] != 0:
#         audio = librosa.effects.pitch_shift(
#             audio,
#             sr=SAMPLE_RATE,
#             n_steps=cfg["pitch"]
#         )

#     # Volume
#     audio *= cfg["volume"]

#     # Prevent clipping
#     peak = np.max(np.abs(audio))
#     if peak > 1:
#         audio /= peak

#     return audio.astype(np.float32)


# # ==========================
# # MAYA SPEAK
# # ==========================

# def maya_speak(text, emotion=None):

#     if emotion is None:
#         emotion = detect_emotion(text)

#     print(f"\n[MAYA]")
#     print(f"Emotion: {emotion}")
#     print(f"Text: {text}")

#     generator = pipeline(
#         text,
#         voice=VOICE
#     )

#     full_audio = []

#     for _, _, audio in generator:
#         full_audio.append(np.array(audio))

#     audio = np.concatenate(full_audio)

#     audio = apply_emotion(
#         audio,
#         emotion
#     )

#     sd.play(
#         audio,
#         samplerate=SAMPLE_RATE
#     )

#     sd.wait()


# # ==========================
# # TESTS
# # ==========================

# if __name__ == "__main__":

#     maya_speak(
#         "Download completed successfully."
#     )

#     maya_speak(
#         "Battery level is twenty percent. Please connect the charger."
#     )

#     maya_speak(
#         "Wow! Congratulations on solving the problem."
#     )

#     maya_speak(
#         "An error occurred while opening the file."
#     )


import time
import torch
from kokoro import KPipeline

pipeline = KPipeline(lang_code="a")

texts = [
    "Hello, I am Maya.",
    "How can I help you today?",
    "This is a warm GPU inference test.",
    "The system is ready."
]

for text in texts:
    torch.cuda.synchronize()
    start = time.perf_counter()

    chunks = list(pipeline(text, voice="af_sky"))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print(f"{elapsed:.3f}s | {text}")

print(f"VRAM allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
print(f"VRAM reserved:  {torch.cuda.memory_reserved()/1024**2:.1f} MB")