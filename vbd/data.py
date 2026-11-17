"""Common Voice fa manifest, FA-EVAL-120 selection, speaker-disjoint folds, enrol/test lists, fine-tuning pool."""
import numpy as np, pandas as pd
from . import config as C


def build_manifest(raw_dir=f"{C.RAW}/cv22_fa"):
    dur = pd.read_csv(f"{raw_dir}/clip_durations.tsv", sep="\t"); dur.columns = ["path", "ms"]
    parts = [pd.read_csv(f"{raw_dir}/{s}.tsv", sep="\t", quoting=3, low_memory=False).assign(split=s)
             for s in ["train", "dev", "test", "other"]]
    df = pd.concat(parts).drop_duplicates("path").merge(dur, on="path", how="left")
    df["sec"] = df.ms / 1000.0
    df["cid"] = df.path.str.extract(r"_(\d+)\.mp3").astype(int)
    ids = {c: f"S{i:05d}" for i, c in enumerate(sorted(df.client_id.unique()))}
    df["spk"] = df.client_id.map(ids)
    df["gender"] = df.gender.map({"male_masculine": "M", "female_feminine": "F"}).fillna("U")
    cols = ["spk", "path", "split", "sec", "cid", "gender", "age", "accents", "sentence", "up_votes", "down_votes"]
    return df[cols].sort_values(["spk", "cid"]).reset_index(drop=True)


def _sessions(cids):
    cids = np.asarray(cids)
    return np.concatenate([[0], np.cumsum(np.diff(cids) > C.SESSION_GAP)])


def select_eval(man, seed=C.SEED):
    """Pick 120 eval speakers; return (speakers df, utterance df with role/cond/fold)."""
    rng = np.random.default_rng(seed)
    el = man[(man.sec >= C.MIN_SEC) & (man.sec <= C.MAX_SEC)].copy()
    el["sess"] = el.groupby("spk").cid.transform(_sessions)
    st = el.groupby("spk").agg(n=("path", "size"), gender=("gender", lambda x: x.mode().iat[0]),
                               n_sess=("sess", "nunique")).reset_index()
    cand = st[st.n >= C.MIN_ELIGIBLE].copy()
    cand["r"] = rng.random(len(cand))
    cand["multi"] = (cand.n_sess >= 2).astype(int)
    fem = cand[cand.gender == "F"].sort_values(["multi", "r"], ascending=[False, True])
    rest = cand[cand.gender != "F"].sort_values(["multi", "r"], ascending=[False, True])
    n_f = min(len(fem), C.N_EVAL_SPK // 2)
    spk = pd.concat([fem.head(n_f), rest.head(C.N_EVAL_SPK - n_f)]).drop(columns="r")
    assert len(spk) == C.N_EVAL_SPK, len(spk)
    # folds: round-robin within gender so every fold has a similar gender mix
    spk = spk.sample(frac=1, random_state=seed).sort_values("gender", kind="mergesort").reset_index(drop=True)
    spk["fold"] = np.arange(len(spk)) % C.N_FOLDS
    rows = []
    conds = ["clean"] * 5 + ["noisy"] * 5 + ["tel"] * 5
    for s, fold in zip(spk.spk, spk.fold):
        u = el[el.spk == s].sort_values("cid")
        enrol = u.head(C.N_ENROL)
        later = u.iloc[C.N_ENROL:]
        other_sess = later[later.sess > enrol.sess.max()]
        src = other_sess if len(other_sess) >= C.N_TEST else later   # prefer a later "session" than enrolment
        test = src.iloc[np.sort(rng.choice(len(src), C.N_TEST, replace=False))]
        cc = rng.permutation(conds)
        used = set(enrol.path) | set(test.path)
        for _, r in enrol.iterrows(): rows.append((s, fold, "enrol", "clean", r.path, r.sec, r.sess))
        for (_, r), c in zip(test.iterrows(), cc): rows.append((s, fold, "test", c, r.path, r.sec, r.sess))
        for _, r in u[~u.path.isin(used)].iterrows(): rows.append((s, fold, "adapt", "clean", r.path, r.sec, r.sess))
    utt = pd.DataFrame(rows, columns=["spk", "fold", "role", "cond", "path", "sec", "sess"])
    cross = utt[utt.role != "adapt"].groupby("spk").apply(
        lambda g: g[g.role == "test"].sess.min() > g[g.role == "enrol"].sess.max(), include_groups=False)
    spk["cross_session"] = spk.spk.map(cross).astype(bool)
    return spk, utt


def build_pool(man, eval_spk):
    """Fine-tuning pool: every non-eval speaker with enough clips."""
    p = man[~man.spk.isin(set(eval_spk)) & (man.sec >= 1.5) & (man.sec <= 15)]
    n = p.groupby("spk").path.transform("size")
    return p[n >= C.POOL_MIN_CLIPS].reset_index(drop=True)


def rotation(k):
    """Fold roles for rotation k: test fold, validation fold, adaptation folds."""
    test, val = k, (k + 1) % C.N_FOLDS
    return dict(test=test, val=val, adapt=[f for f in range(C.N_FOLDS) if f not in (test, val)])
