"""Biometric metrics: EER, FAR/FRR at a threshold, threshold at target FAR, speaker bootstrap CIs, DET points.
Convention: higher score = more likely genuine / bona fide; accept if score >= threshold."""
import numpy as np


def eer(tar, non):
    """Equal error rate and its threshold."""
    tar, non = np.asarray(tar, float), np.asarray(non, float)
    scores = np.concatenate([tar, non])
    labels = np.concatenate([np.ones(len(tar)), np.zeros(len(non))])
    order = np.argsort(scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    # threshold placed just above scores[i]: FRR = targets <= s, FAR = nontargets > s
    frr = np.concatenate([[0.0], np.cumsum(labels) / len(tar)])
    far = np.concatenate([[1.0], 1.0 - np.cumsum(1 - labels) / len(non)])
    i = int(np.argmin(np.abs(frr - far)))
    thr = scores[min(i, len(scores) - 1)]
    return float((frr[i] + far[i]) / 2), float(thr)


def far_frr(tar, non, thr):
    tar, non = np.asarray(tar, float), np.asarray(non, float)
    return float(np.mean(non >= thr)), float(np.mean(tar < thr))


def thr_at_far(non, target_far):
    """Smallest threshold whose FAR on `non` is <= target_far."""
    non = np.sort(np.asarray(non, float))[::-1]
    k = int(np.floor(target_far * len(non)))  # number of false accepts allowed
    if k >= len(non):
        return float(non[-1])
    return float(np.nextafter(non[k], np.inf)) if k < len(non) else float(non[-1])


def frr_at_far(tar, non, target_far):
    thr = thr_at_far(non, target_far)
    return far_frr(tar, non, thr)[1], thr


def det_points(tar, non, n=400):
    tar, non = np.asarray(tar, float), np.asarray(non, float)
    thrs = np.quantile(np.concatenate([tar, non]), np.linspace(0, 1, n))
    return np.array([np.mean(non >= t) for t in thrs]), np.array([np.mean(tar < t) for t in thrs]), thrs


def bootstrap_ci(fn, groups, n_boot=1000, seed=0, alpha=0.05):
    """Percentile CI by resampling groups (speakers) with replacement.
    groups: dict group_id -> dict(tar=array, non=array); fn(tar, non) -> float."""
    rng = np.random.default_rng(seed)
    keys = list(groups)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(keys), len(keys), replace=True)
        tar = np.concatenate([groups[keys[i]]["tar"] for i in pick])
        non = np.concatenate([groups[keys[i]]["non"] for i in pick])
        if len(tar) and len(non):
            vals.append(fn(tar, non))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def group_by_speaker(df, score_col="score", label_col="target", spk_col="model_spk"):
    return {k: dict(tar=g.loc[g[label_col] == 1, score_col].values, non=g.loc[g[label_col] == 0, score_col].values)
            for k, g in df.groupby(spk_col)}
