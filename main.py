import os
import torch
import torchaudio
from transformers import MusicgenForConditionalGeneration, AutoProcessor
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-stereo-medium",
    torch_dtype=torch.float16
).to(device)

processor = AutoProcessor.from_pretrained("facebook/musicgen-stereo-medium")

prompt = "Chaotic trap beat, glitchy risers, comedic bass hits, fast pacing, 8 seconds."

inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt"
).to(device)

with torch.no_grad():
    audio_values = model.generate(
        **inputs,
        max_new_tokens=512
    )

# Convert float16 → float32 and transpose
audio = audio_values[0].cpu().float().numpy().T

os.makedirs("outputs", exist_ok=True)

sf.write("outputs/wtfoutput3.wav", audio, 32000)
print("Saved output.wav")

#
# # Make a soundfile with a provided melody
# melody, sr = torchaudio.load("melody.wav")
# melody = melody.to(device)
#
# inputs = processor(
#     text=[prompt],
#     audio=melody,
#     sampling_rate=sr,
#     return_tensors="pt"
# ).to(device)
#
# with torch.no_grad():
#     audio_values = model.generate(
#         **inputs,
#         max_new_tokens=256
#     )
#
# audio = audio_values[0].cpu().float().numpy().T
# sf.write("output_melody.wav", audio, 32000)
# print("Saved output_melody.wav")
