"""Build the FA-EVAL-120 test conditions (deterministic) and evaluate an encoder on all of them.

main15     : the proposal's 15 genuine tests per speaker = 5 clean + 5 noisy (MUSAN, SNR 5-20 dB) + 5 telephone (light noise, 300-3400 Hz, AMR-NB)
clean      : the 5 clean clips                      noisy / tel : the two other groups on their own
snr10      : clean clips + MUSAN noise/babble at exactly 10 dB
gsm_fr, amr_nb, g711a, opus_nb : clean clips through band-limit + codec
ti5s       : clean clips extended to >= 5 s by appending another clean clip of the same speaker
Noise comes only from the MUSAN files reserved for evaluation (never used in training augmentation)."""
import os, pickle, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from . import config as C, augment as A, sv as SV, metrics as M
from .train_sv import noise_files, read_segment

CODECS = ["gsm_fr", "amr_nb", "g711a", "opus_nb"]
PKL = f"{C.WORK}/eval_conditions.pkl"


def _noise(bank, n, rng, cats=("noise", "babble", "music"), p=(0.5, 0.35, 0.15)):
    cat = rng.choice(cats, p=p)
    if cat == "babble":
        k = int(rng.integers(3, 7))
        return np.sum([A._fit(read_segment(bank["speech"][rng.integers(len(bank["speech"]))], n, rng), n, rng) for _ in range(k)], 0)
    return read_segment(bank[cat][rng.integers(len(bank[cat]))], n, rng)


def build(utt, audio_root=C.CV_AUDIO, force=False):
    """Returns (items df [key, spk, fold, cond], dict key -> wav). Cached on the Colab disk."""
    if os.path.exists(PKL) and not force:
        return pickle.load(open(PKL, "rb"))
    ev = utt[utt.role.isin(["enrol", "test"])].reset_index(drop=True)
    with ThreadPoolExecutor(4) as ex:
        wav = dict(zip(ev.path, ex.map(lambda r: A.load(f"{audio_root}/{r}"), ev.rel)))
    bank = noise_files("eval"); rng = np.random.default_rng(C.SEED + 1)
    test = ev[ev.role == "test"]; items, out = [], {}
    def add(key, r, cond, x): out[key] = x.astype(np.float32); items.append((key, r.spk, r.fold, cond))
    for r in ev[ev.role == "enrol"].itertuples(): out[f"enrol|{r.path}"] = wav[r.path]
    cl, nz, tl = (test[test.cond == c] for c in ["clean", "noisy", "tel"])
    for r in cl.itertuples():
        add(f"clean|{r.path}", r, "clean", wav[r.path])
    for r in nz.itertuples():
        x = wav[r.path]; add(f"noisy|{r.path}", r, "noisy", A.peak_norm(A.add_noise(x, _noise(bank, len(x), rng), rng.uniform(5, 20), rng)))
    pre = [A.bandlimit(A.add_noise(wav[p], _noise(bank, len(wav[p]), rng, ("noise", "babble"), (0.6, 0.4)), rng.uniform(15, 30), rng)) for p in tl.path]
    for r, y in zip(tl.itertuples(), A.codec_batch(pre, "amr_nb")): add(f"tel|{r.path}", r, "tel", y)
    for r in cl.itertuples():
        x = wav[r.path]; add(f"snr10|{r.path}", r, "snr10", A.peak_norm(A.add_noise(x, _noise(bank, len(x), rng, ("noise", "babble"), (0.6, 0.4)), 10.0, rng)))
    bl = [A.bandlimit(wav[p]) for p in cl.path]
    for cn in CODECS:
        for r, y in zip(cl.itertuples(), A.codec_batch(bl, cn)): add(f"{cn}|{r.path}", r, cn, y)
    for s, g in cl.groupby("spk"):
        ps = g.path.tolist()
        for i, r in enumerate(g.itertuples()):
            x = wav[r.path]; j = i
            while len(x) < 5 * C.SR:
                j = (j + 1) % len(ps); x = np.concatenate([x, wav[ps[j]]])
            add(f"ti5s|{r.path}", r, "ti5s", x)
    items = pd.DataFrame(items, columns=["key", "spk", "fold", "cond"])
    pickle.dump((items, out), open(PKL, "wb"), protocol=4)
    return items, out


def evaluate(enc, utt, items, wavs, tag, save=True, snorm=None):
    """Embed everything with `enc`, score all conditions, return (trials, summary, operating points)."""
    keys = list(wavs); E = SV.embed(enc, [wavs[k] for k in keys]); emb = dict(zip(keys, E))
    u = utt[utt.role == "enrol"].assign(key=lambda d: "enrol|" + d.path)
    tr = SV.score_trials(u, emb, items)
    main = tr[tr.cond.isin(["clean", "noisy", "tel"])].assign(cond="main15")
    tr = pd.concat([main, tr], ignore_index=True)
    summ = SV.summarize(tr)
    ops = []
    for far_t in (0.008, 0.001):
        far, frr, pf, _ = SV.operating_point(tr[tr.cond == "main15"], far_t)
        ops.append(dict(target_far=100 * far_t, far=100 * far, frr=100 * frr, far_fold_std=100 * pf.far.std(), frr_fold_std=100 * pf.frr.std()))
    ops = pd.DataFrame(ops)
    if save:
        tr.to_csv(f"{C.SCORES}/{tag}_trials.csv.gz", index=False); summ.to_csv(f"{C.REPORTS}/{tag}_summary.csv", index=False)
        ops.to_csv(f"{C.REPORTS}/{tag}_operating_points.csv", index=False)
        np.savez_compressed(f"{C.FEATURES}/{tag}_emb.npz", keys=np.array(keys), emb=E)
    return tr, summ, ops, emb
