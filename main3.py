import os
import torch
import soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor

# -----------------------------
# Device selection
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# -----------------------------
# Load model + processor
# -----------------------------
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-stereo-medium",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

processor = AutoProcessor.from_pretrained("facebook/musicgen-stereo-medium")

# -----------------------------
# Prompt for your viral WTF audio
# -----------------------------
prompt = (
    "lofi electronic hybrid, crisp drums, dreamy synths, subtle vinyl texture, 90 bpm, energetic"
)

# -----------------------------
# Prepare input
# -----------------------------
inputs = processor(
    text=[prompt],
    padding=True,
    return_tensors="pt"
).to(device)

# -----------------------------
# Generate audio
# -----------------------------
with torch.no_grad():
    audio_values = model.generate(
        **inputs,
        # max_new_tokens=1024,   # ~8 seconds for stereo-medium
        max_new_tokens=1024,
        do_sample=True,
        temperature=1.2,
        top_k=250,
        top_p=0.95
    )

# -----------------------------
# Convert float16 → float32
# MusicGen outputs: (batch, channels, samples)
# -----------------------------
audio = audio_values[0].cpu().float().numpy().T  # shape: (samples, channels)

# -----------------------------
# Save output
# -----------------------------
os.makedirs("outputs", exist_ok=True)
output_path = "outputs/wtfoutput12.wav"

sf.write(output_path, audio, 32000)
print(f"Saved {output_path}")
