"""S5a: simulated replay attacks for FA-EVAL-120 (1,200 clips = 120 speakers x 10).
The proposal asked for physical replays through 3 playback devices; this is a software substitute:
  genuine recording of the target (a clip that is neither an enrolment nor a test clip)
  -> loudspeaker model (device band-pass + resonance + soft clipping) -> room impulse response -> ambient noise
  -> recording-microphone high-pass. Three device profiles: phone speaker, laptop speaker, external (bluetooth-class) speaker.
Output: /content/work/spoof/replay/*.flac, tar on Drive (data/spoof/replay.tar), index data/protocols/fa_spoof_replay.csv.
    python -m vbd.spoof_replay
"""
import os, time, subprocess, numpy as np, pandas as pd, soundfile as sf
from scipy.signal import butter, sosfilt, iirpeak, lfilter
from . import config as C, augment as A
from .train_sv import noise_files, read_segment

PER_SPK = 10
#            band-pass Hz, resonance (Hz, Q), clipping drive (dB range), room-to-direct mix, ambient SNR dB range
DEVICES = {"phone":    dict(band=(450, 7200), peak=(2800, 2.0), drive=(6, 14), snr=(18, 32)),
           "laptop":   dict(band=(220, 7600), peak=(1100, 1.5), drive=(3, 10), snr=(18, 32)),
           "external": dict(band=(90, 7900),  peak=(160, 1.2),  drive=(0, 6),  snr=(20, 35))}
OUT = f"{C.WORK}/spoof/replay"


def loudspeaker(x, dev, rng):
    d = DEVICES[dev]
    x = sosfilt(butter(3, d["band"], btype="band", fs=C.SR, output="sos"), x)
    b, a = iirpeak(d["peak"][0], d["peak"][1], fs=C.SR); x = x + 0.6 * lfilter(b, a, x)
    g = 10 ** (rng.uniform(*d["drive"]) / 20); x = x / (np.abs(x).max() + 1e-9)
    return (np.tanh(g * x) / np.tanh(g)).astype(np.float32)


def replay(x, dev, bank, rng):
    y = loudspeaker(x, dev, rng)
    y = A.reverb(y, read_segment(bank["rir"][rng.integers(len(bank["rir"]))], 10 ** 9, rng))
    nz = read_segment(bank["noise"][rng.integers(len(bank["noise"]))], len(y), rng)
    y = A.add_noise(y, nz, rng.uniform(*DEVICES[dev]["snr"]), rng)
    y = sosfilt(butter(2, 70, btype="high", fs=C.SR, output="sos"), y)
    return A.peak_norm(y.astype(np.float32))


def main():
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
    os.makedirs(OUT, exist_ok=True); os.makedirs(f"{C.ROOT}/data/spoof", exist_ok=True)
    utt = pd.read_csv(f"{C.PROTOCOLS}/fa_eval120_utts.csv")
    spare = utt[~utt.role.isin(["enrol", "test"]) & utt.sec.between(3, 10)]
    bank = {k: v for k, v in noise_files("train").items()}; b2 = noise_files("eval")
    bank = {k: bank[k] + b2[k] for k in bank}                      # replay uses the whole RIR / noise bank; CM folds are split by speaker
    rng = np.random.default_rng(C.SEED + 50); devs = list(DEVICES); rows = []
    for si, (spk, g) in enumerate(spare.groupby("spk")):
        g = g.sample(min(PER_SPK, len(g)), random_state=C.SEED % 100000 + si)
        for i, r in enumerate(g.itertuples()):
            dev = devs[(si + i) % 3]; x = A.load(f"{C.CV_AUDIO}/{r.rel}"); y = replay(x, dev, bank, rng)
            name = f"{spk}_replay_{dev}_{i:02d}.flac"; sf.write(f"{OUT}/{name}", y, C.SR, subtype="PCM_16")
            rows.append(dict(spk=spk, fold=r.fold, attack="replay", system=f"replay_{dev}", file=name, src_rel=r.rel, sec=len(y) / C.SR))
        if si % 20 == 0: log("speaker", si, "/ 120")
    df = pd.DataFrame(rows); df.to_csv(f"{C.PROTOCOLS}/fa_spoof_replay.csv", index=False)
    subprocess.run(f"tar -cf {C.ROOT}/data/spoof/replay.tar -C {C.WORK}/spoof replay", shell=True, check=True)
    log("clips", len(df), "speakers", df.spk.nunique(), "per device", df.system.value_counts().to_dict(), "min per speaker", int(df.groupby("spk").size().min()),
        "hours", round(df.sec.sum() / 3600, 2)); log("REPLAY_DONE")


if __name__ == "__main__":
    main()
