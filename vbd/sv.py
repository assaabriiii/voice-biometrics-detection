"""Speaker-verification embedding extraction (SpeechBrain pretrained encoders) and trial scoring."""
import numpy as np, pandas as pd, torch
from . import config as C, metrics as M

SOURCES = {"ecapa": "speechbrain/spkrec-ecapa-voxceleb", "xvector": "speechbrain/spkrec-xvect-voxceleb"}


def load_encoder(name="ecapa", device=None):
    from speechbrain.inference.speaker import EncoderClassifier
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = EncoderClassifier.from_hparams(source=SOURCES[name], savedir=f"{C.WORK}/pretrained/{name}", run_opts={"device": device})
    enc.eval()
    return enc


@torch.no_grad()
def embed(enc, wavs, batch_size=32):
    """wavs: list of 1-D float32 arrays at 16 kHz -> (N, D) L2-normalised embeddings."""
    out = np.zeros((len(wavs), 0), np.float32); res = [None] * len(wavs)
    order = np.argsort([len(w) for w in wavs])
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        L = max(len(wavs[j]) for j in idx)
        x = torch.zeros(len(idx), L)
        for k, j in enumerate(idx): x[k, :len(wavs[j])] = torch.from_numpy(wavs[j])
        lens = torch.tensor([len(wavs[j]) / L for j in idx])
        e = enc.encode_batch(x.to(enc.device), lens.to(enc.device)).squeeze(1)
        e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        for k, j in enumerate(idx): res[j] = e[k]
    return np.stack(res) if res else out


def templates(enrol_df, emb):
    """Mean of normalised enrolment embeddings per speaker -> dict spk -> unit vector."""
    t = {}
    for s, g in enrol_df.groupby("spk"):
        v = np.mean([emb[k] for k in g.key], axis=0)
        t[s] = v / np.linalg.norm(v)
    return t


def score_trials(utt, emb, test_keys):
    """All test items against every template of the same fold.
    utt: protocol rows with a `key` column into `emb`; test_keys: df with columns spk, fold, cond, key."""
    rows = []
    for fold, tests in test_keys.groupby("fold"):
        tpl = templates(utt[(utt.fold == fold) & (utt.role == "enrol")], emb)
        spks = sorted(tpl); T = np.stack([tpl[s] for s in spks])
        E = np.stack([emb[k] for k in tests.key]); S = E @ T.T
        for (r, sc) in zip(tests.itertuples(), S):
            for s, v in zip(spks, sc):
                rows.append((fold, s, r.spk, r.cond, r.key, int(s == r.spk), float(v)))
    return pd.DataFrame(rows, columns=["fold", "model_spk", "test_spk", "cond", "key", "target", "score"])


def summarize(tr, n_boot=500):
    """EER (+95% speaker-bootstrap CI) per condition."""
    out = []
    for cond, g in tr.groupby("cond"):
        tar, non = g.score[g.target == 1].values, g.score[g.target == 0].values
        e, _ = M.eer(tar, non)
        lo, hi = M.bootstrap_ci(lambda a, b: M.eer(a, b)[0], M.group_by_speaker(g), n_boot=n_boot)
        folds = [M.eer(f.score[f.target == 1].values, f.score[f.target == 0].values)[0] for _, f in g.groupby("fold")]
        out.append(dict(cond=cond, n_tar=len(tar), n_non=len(non), eer=100 * e, ci_lo=100 * lo, ci_hi=100 * hi,
                        fold_mean=100 * np.mean(folds), fold_std=100 * np.std(folds)))
    return pd.DataFrame(out)


def operating_point(tr, target_far=0.008):
    """Per rotation: threshold from the validation fold at target FAR, applied unchanged to the test fold. Pooled FAR/FRR."""
    from .data import rotation
    acc = []
    for k in sorted(tr.fold.unique()):
        r = rotation(k)
        v, t = tr[tr.fold == r["val"]], tr[tr.fold == r["test"]]
        thr = M.thr_at_far(v.score[v.target == 0].values, target_far)
        acc.append(t.assign(thr=thr, accept=t.score >= thr))
    a = pd.concat(acc)
    far = float(a.accept[a.target == 0].mean()); frr = float(1 - a.accept[a.target == 1].mean())
    per_fold = a.groupby("fold").apply(lambda g: pd.Series(dict(far=g.accept[g.target == 0].mean(), frr=1 - g.accept[g.target == 1].mean(), thr=g.thr.iat[0])), include_groups=False)
    return far, frr, per_fold, a


def cohort_embeddings(enc, store, speakers, max_clips=8, seed=0, variants=("orig", "tel")):
    """One mean embedding per (cohort speaker, channel variant). Cohort speakers come from the fine-tuning pool,
    which is disjoint from FA-EVAL-120, so score normalisation uses no evaluation speaker."""
    from . import augment as A
    rng = np.random.default_rng(seed); idx = store.idx; out = []
    for s in speakers:
        r = rng.permutation(np.flatnonzero(idx.spk.values == s))[:max_clips]
        w = [store.get(int(i)) for i in r]
        for v in variants:
            E = embed(enc, w if v == "orig" else [A.mulaw_8k(x) for x in w])
            m = E.mean(0); out.append(m / np.linalg.norm(m))
    return np.stack(out)


def _top_stats(S, topn):
    top = np.sort(S, axis=1)[:, -topn:]
    return top.mean(1), top.std(1) + 1e-6


def score_trials_asnorm(utt, emb, test_keys, cohort, topn=200):
    """Adaptive symmetric score normalisation (AS-norm): each raw cosine is normalised by the statistics of the
    enrolment template's and the test utterance's top-N scores against the cohort."""
    rows = []
    for fold, tests in test_keys.groupby("fold"):
        tpl = templates(utt[(utt.fold == fold) & (utt.role == "enrol")], emb)
        spks = sorted(tpl); T = np.stack([tpl[s] for s in spks])
        E = np.stack([emb[k] for k in tests.key]); S = E @ T.T
        mu_e, sd_e = _top_stats(T @ cohort.T, topn); mu_t, sd_t = _top_stats(E @ cohort.T, topn)
        Sn = 0.5 * ((S - mu_e[None]) / sd_e[None] + (S - mu_t[:, None]) / sd_t[:, None])
        for (r, sc) in zip(tests.itertuples(), Sn):
            for s, v in zip(spks, sc):
                rows.append((fold, s, r.spk, r.cond, r.key, int(s == r.spk), float(v)))
    return pd.DataFrame(rows, columns=["fold", "model_spk", "test_spk", "cond", "key", "target", "score"])


def full_report(tr, tag=None):
    """Add the pooled main15 condition, summarise EERs and both operating points; optionally save."""
    main = tr[tr.cond.isin(["clean", "noisy", "tel"])].assign(cond="main15")
    tr = pd.concat([main, tr[tr.cond != "main15"]], ignore_index=True)
    summ = summarize(tr); ops = []
    for far_t in (0.008, 0.001):
        far, frr, pf, _ = operating_point(tr[tr.cond == "main15"], far_t)
        ops.append(dict(target_far=100 * far_t, far=100 * far, frr=100 * frr, far_fold_std=100 * pf.far.std(), frr_fold_std=100 * pf.frr.std()))
    ops = pd.DataFrame(ops)
    if tag:
        tr.to_csv(f"{C.SCORES}/{tag}_trials.csv.gz", index=False); summ.to_csv(f"{C.REPORTS}/{tag}_summary.csv", index=False)
        ops.to_csv(f"{C.REPORTS}/{tag}_operating_points.csv", index=False)
    return tr, summ, ops
