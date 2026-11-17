"""Audio I/O and channel simulation: additive noise at a set SNR, reverberation, telephone band-limit, codecs (via ffmpeg)."""
import os, subprocess, tempfile, numpy as np, soundfile as sf, soxr
from scipy.signal import fftconvolve, butter, sosfiltfilt
from . import config as C

FFMPEG = os.environ.get("VBD_FFMPEG", "ffmpeg")
FFMPEG_AMR = f"{C.WORK}/ffmpeg_static"      # static build with libopencore_amrnb (copied from Drive tools/)

# name -> (sample rate on the wire, ffmpeg encoder args, container)
CODECS = {
    "gsm_fr":  (8000, ["-c:a", "libgsm", "-b:a", "13k"], "gsm"),
    "amr_nb":  (8000, ["-c:a", "libopencore_amrnb", "-b:a", "12.2k"], "amr"),
    "g711a":   (8000, ["-c:a", "pcm_alaw"], "wav"),
    "g711u":   (8000, ["-c:a", "pcm_mulaw"], "wav"),
    "opus_nb": (8000, ["-c:a", "libopus", "-b:a", "12k", "-application", "voip"], "ogg"),
    "opus_wb": (16000, ["-c:a", "libopus", "-b:a", "16k", "-application", "voip"], "ogg"),
}


def load(path, sr=C.SR):
    """Read any libsndfile-supported file (wav/flac/mp3) as mono float32 at `sr`."""
    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if fs != sr:
        x = soxr.resample(x, fs, sr)
    return x.astype(np.float32)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def _fit(noise, n, rng):
    if len(noise) < n:
        noise = np.tile(noise, int(np.ceil(n / len(noise))))
    s = rng.integers(0, len(noise) - n + 1)
    return noise[s:s + n]


def add_noise(x, noise, snr_db, rng):
    """Mix `noise` into `x` at exactly `snr_db` (energy ratio over the whole utterance)."""
    n = _fit(noise, len(x), rng)
    g = rms(x) / (rms(n) * 10 ** (snr_db / 20))
    return (x + g * n).astype(np.float32)


def reverb(x, rir):
    rir = rir / (np.abs(rir).max() + 1e-9)
    d = int(np.argmax(np.abs(rir)))
    y = fftconvolve(x, rir)[d:d + len(x)]
    return (y * rms(x) / rms(y)).astype(np.float32)


def bandlimit(x, lo=300, hi=3400, sr=C.SR):
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x).astype(np.float32)


def codec(x, name, sr=C.SR):
    """Encode + decode through a real codec with ffmpeg; returns audio at `sr`, same length."""
    wire_sr, args, ext = CODECS[name]
    with tempfile.TemporaryDirectory() as d:
        a, b, c = f"{d}/a.wav", f"{d}/b.{ext}", f"{d}/c.wav"
        sf.write(a, x, sr, subtype="PCM_16")
        FF = FFMPEG_AMR if name == "amr_nb" else FFMPEG
        subprocess.run([FF, "-loglevel", "error", "-y", "-i", a, "-ac", "1", "-ar", str(wire_sr), *args, b], check=True)
        subprocess.run([FF, "-loglevel", "error", "-y", "-i", b, "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", c], check=True)
        y = load(c, sr)
    y = y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def telephone(x, name="gsm_fr", noise=None, snr_db=None, rng=None):
    """Simulated call: optional background noise at the talker side -> 300-3400 Hz handset band -> codec."""
    if noise is not None:
        x = add_noise(x, noise, snr_db, rng)
    return codec(bandlimit(x), name)


def peak_norm(x, peak=0.95):
    m = np.abs(x).max()
    return x if m < 1e-6 else (x * min(1.0, peak / m) if m > peak else x)


def codec_batch(wavs, name, gap=0.25, chunk_sec=1800, sr=C.SR):
    """Same as codec() for many clips but with one ffmpeg round-trip per ~30 min of audio:
    clips are joined with short silences, coded, then cut back at the original offsets."""
    out = [None] * len(wavs); g = np.zeros(int(gap * sr), np.float32)
    i = 0
    while i < len(wavs):
        j, tot = i, 0
        while j < len(wavs) and (tot < chunk_sec * sr or j == i):
            tot += len(wavs[j]) + len(g); j += 1
        y = codec(np.concatenate([np.concatenate([w, g]) for w in wavs[i:j]]), name, sr)
        p = 0
        for k in range(i, j):
            out[k] = y[p:p + len(wavs[k])].copy(); p += len(wavs[k]) + len(g)
        i = j
    return out


def mulaw_8k(x, sr=C.SR):
    """Fast in-process telephone approximation for training-time augmentation: 300-3400 Hz, 8 kHz, G.711 mu-law."""
    y = soxr.resample(bandlimit(x, sr=sr), sr, 8000)
    mu = 255.0
    q = np.sign(y) * np.log1p(mu * np.abs(np.clip(y, -1, 1))) / np.log1p(mu)
    q = np.round((q + 1) / 2 * mu) / mu * 2 - 1
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    y = soxr.resample(y.astype(np.float32), 8000, sr)
    return (y[:len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))).astype(np.float32)
