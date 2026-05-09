import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter


def stereo_upmix(audio, sr, delay_ms=12, gain_l=1.0, gain_r=0.98):
    """
    Convert mono audio into wide stereo using Haas effect + subtle EQ difference.
    """
    # Ensure mono
    if audio.ndim > 1:
        audio = audio.mean(axis=0)

    # Convert delay to samples
    delay_samples = int(sr * (delay_ms / 1000.0))

    # Create left and right channels
    left = audio * gain_l

    # Right channel delayed
    right = np.concatenate([np.zeros(delay_samples), audio])[: len(audio)]
    right *= gain_r

    # Slight EQ difference for width
    def lowpass(sig, cutoff=8000):
        b, a = butter(2, cutoff / (sr / 2), btype='low')
        return lfilter(b, a, sig)

    right = lowpass(right, cutoff=9000)

    # Stack into stereo
    stereo = np.stack([left, right], axis=1)
    return stereo
