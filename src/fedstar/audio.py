"""One-second, 16 kHz log-Mel inputs used by the paper audio CNN."""

from functools import lru_cache
import math

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf
import torch


def read_wave(path):
    """Decode audio, average channels and resample to mono float32 at 16 kHz."""
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        wave = resample_poly(wave, 16000 // divisor, rate // divisor)
    if not len(wave) or not np.isfinite(wave).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    return wave.astype(np.float32)


@lru_cache(maxsize=1)
def _mel_weights():
    # Match tf.signal.linear_to_mel_weight_matrix: HTK scale, triangles in
    # Mel space, no area normalization, and a zeroed DC bin.
    frequencies = torch.linspace(0.0, 8000.0, 513, dtype=torch.float32, device="cpu")
    mel = 1127.0 * torch.log1p(frequencies / 700.0)
    edges = torch.linspace(
        1127.0 * math.log1p(60.0 / 700.0),
        1127.0 * math.log1p(7800.0 / 700.0),
        66,
        dtype=torch.float32,
        device="cpu",
    )
    lower = (mel[:, None] - edges[:-2]) / (edges[1:-1] - edges[:-2])
    upper = (edges[2:] - mel[:, None]) / (edges[2:] - edges[1:-1])
    weights = torch.minimum(lower, upper).clamp_min(0)
    weights[0] = 0
    return weights


def features(wave, training, rng=None):
    """Return [1,98,64] for training or [segments,1,98,64] for evaluation.

    Training normalizes the whole clip before randomly cropping one second.
    Evaluation normalizes each complete, non-overlapping second separately;
    incomplete trailing audio is discarded. Short clips are center padded.
    The epsilon follows TensorFlow l2_normalize's squared-norm convention.
    """
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Expected a nonempty finite mono waveform")
    if len(wave) < 16000:
        extra = 16000 - len(wave)
        wave = np.pad(wave, (extra // 2, extra - extra // 2))
    if training:
        rng = rng if rng is not None else np.random.default_rng()
        wave = wave / np.sqrt(max(float(np.dot(wave, wave)), 1e-9))
        start = int(rng.integers(len(wave) - 16000 + 1))
        segments = wave[start : start + 16000][None]
    else:
        segments = wave[: len(wave) // 16000 * 16000].reshape(-1, 16000)
        norm = np.sqrt(np.maximum(np.sum(segments * segments, axis=1, keepdims=True), 1e-9))
        segments = segments / norm
    samples = torch.from_numpy(np.ascontiguousarray(segments, dtype=np.float32))
    # Explicit 400-sample frames avoid torch.stft's n_fft-based framing.
    frames = samples.unfold(-1, 400, 160)
    window = torch.hann_window(400, periodic=True, dtype=torch.float32, device="cpu")
    spectrum = torch.fft.rfft(frames * window, n=1024).abs()
    result = (spectrum @ _mel_weights()).clamp(1e-5, 1e8).log().unsqueeze(1)
    return result[0] if training else result
