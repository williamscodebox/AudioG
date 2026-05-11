import os
import requests
from pathlib import Path
from typing import List, Tuple
from moviepy import VideoFileClip, AudioFileClip, CompositeAudioClip


FREESOUND_API_BASE = "https://freesound.org/apiv2"


class SFXEngine:
    def __init__(self, api_key: str | None = None, sfx_dir: str = "sfx_cache"):
        self.api_key = api_key or os.getenv("FREESOUND_API_KEY")
        if not self.api_key:
            raise ValueError(
                "FREESOUND_API_KEY not set. "
                "Set it in env or pass api_key to SFXEngine()."
            )

        self.sfx_dir = Path(sfx_dir)
        self.sfx_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # TEXT → SFX
    # -----------------------------
    def text_to_sfx(
        self,
        query: str,
        max_duration: float = 3.0,
        min_duration: float = 0.2,
        license_filter: str = "license:(\"Creative Commons 0\" OR \"Attribution\")",
    ) -> Path:
        """
        Search Freesound for a short SFX and download it as WAV.
        Returns local file path.
        """

        print(f"[SFX] Searching Freesound for: {query!r}")

        params = {
            "query": query,
            "filter": f"duration:[{min_duration} TO {max_duration}] {license_filter}",
            "fields": "id,name,previews,duration",
            "sort": "score",
            "page_size": 1,
            "token": self.api_key,
        }

        r = requests.get(f"{FREESOUND_API_BASE}/search/text/", params=params, timeout=15)
        r.raise_for_status()
        data = r.json()

        if not data.get("results"):
            raise RuntimeError(f"No SFX found for query: {query}")

        result = data["results"][0]
        sound_id = result["id"]
        name = result["name"]

        print(f"[SFX] Found: {name} (id={sound_id})")

        # Get sound details to access high‑quality preview
        r2 = requests.get(
            f"{FREESOUND_API_BASE}/sounds/{sound_id}/",
            params={"token": self.api_key},
            timeout=15,
        )
        r2.raise_for_status()
        sound_data = r2.json()

        # Prefer HQ OGG preview (small but good)
        preview_url = sound_data["previews"].get("preview-hq-ogg") or sound_data["previews"].get("preview-hq-mp3")
        if not preview_url:
            raise RuntimeError("No suitable preview URL found for sound.")

        out_path = self.sfx_dir / f"{sound_id}.ogg"
        if not out_path.exists():
            print(f"[SFX] Downloading: {preview_url}")
            with requests.get(preview_url, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        else:
            print(f"[SFX] Using cached file: {out_path}")

        return out_path

    # -----------------------------
    # VIDEO → FOLEY
    # -----------------------------
    def video_add_sfx(
        self,
        video_path: str | Path,
        cues: List[Tuple[float, str]],
        output_path: str | Path = "video_with_sfx.mp4",
        sfx_gain_db: float = 0.0,
    ) -> Path:
        """
        Add SFX to a video at given timestamps.

        cues: list of (time_in_seconds, text_query_for_sfx)
        """

        video_path = Path(video_path)
        output_path = Path(output_path)

        print(f"[FOLEY] Loading video: {video_path}")
        clip = VideoFileClip(str(video_path))

        audio_layers = [clip.audio] if clip.audio is not None else []

        for t, query in cues:
            print(f"[FOLEY] Cue at {t:.2f}s → {query!r}")
            sfx_file = self.text_to_sfx(query)
            sfx_clip = AudioFileClip(str(sfx_file)).set_start(t)

            if sfx_gain_db != 0.0:
                factor = 10 ** (sfx_gain_db / 20.0)
                sfx_clip = sfx_clip.volumex(factor)

            audio_layers.append(sfx_clip)

        final_audio = CompositeAudioClip(audio_layers)
        final_clip = clip.set_audio(final_audio)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[FOLEY] Writing output: {output_path}")
        final_clip.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
        )

        clip.close()
        final_clip.close()
        return output_path


if __name__ == "__main__":
    # Example usage
    engine = SFXEngine()

    # A) Text → SFX
    sfx_path = engine.text_to_sfx("cartoon boing")
    print(f"[DEMO] Downloaded SFX: {sfx_path}")

    # B) Video → Foley
    # Example cues: (time_in_seconds, description)
    cues = [
        (0.5, "whoosh fast"),
        (1.2, "metallic clang"),
        (2.0, "cartoon pop"),
    ]
    engine.video_add_sfx("input.mp4", cues, "output_with_sfx.mp4")
