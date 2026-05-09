import torch
import soundfile as sf
from transformers import MusicgenForConditionalGeneration, AutoProcessor
from stereo import stereo_upmix


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_name: str = "facebook/musicgen-stereo-medium", device: str = "cuda"):
    print(f"Loading model: {model_name} on {device} (fp16)…")
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def generate_music(
    model,
    processor,
    prompt: str,
    duration_seconds: int = 12,
    temperature: float = 1.2,
    top_k: int = 250,
    top_p: float = 0.95,
):
    print(f"Prompt: {prompt}")
    device = next(model.parameters()).device

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # 32kHz, ~32k samples per second → duration * 32000 / 4 ≈ tokens
    max_new_tokens = min(int(duration_seconds * 32000 / 4), 16000)


    print(f"Generating… (duration ~{duration_seconds}s, max_new_tokens={max_new_tokens})")
    audio_values = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # audio_values: (batch, channels, samples)
    audio = audio_values[0, 0].detach().cpu().numpy()
    sampling_rate = model.config.audio_encoder.sampling_rate
    return audio, sampling_rate


def save_audio(audio, sr: int, path: str = "output.wav"):
    sf.write(path, audio, sr)
    print(f"Saved {path} ({len(audio) / sr:.2f}s @ {sr} Hz)")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # 🔥 Viral “what did I just watch?” style prompt
    prompt = (
        "fast chaotic electronic beat with glitch effects, distorted bass, random comedic sound cues, "
        "unpredictable rhythm, high energy, perfect for absurd meme content"
    )

    model, processor = load_model(device=device)

    audio, sr = generate_music(
        model,
        processor,
        prompt=prompt,
        duration_seconds=12,
        temperature=1.2,
        top_k=250,
        top_p=0.95,
    )

    # stereo = stereo_upmix(audio, sr)
    sf.write("output_stereo.wav", audio, sr)
    print("Saved output_stereo.wav (stereo)")


if __name__ == "__main__":
    main()