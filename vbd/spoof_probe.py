"""S5 probe: try candidate spoofing systems on a few FA-EVAL-120 speakers before bulk generation.
For each system: speaker-verification cosine of the spoof against the target's enrolment template (does it fool SV?)
and the liveness score of the ASVspoof-trained countermeasures (is it caught without Persian fine-tuning?).
    python -m vbd.spoof_probe [n_speakers]
Systems here: kNN-VC (voice conversion from another human speaker), MMS-TTS Persian (single voice), MMS-TTS -> kNN-VC (cloned TTS).
"""
import os, sys, time, numpy as np, pandas as pd, torch, soundfile as sf
from . import config as C, sv as SV, cm as CM, augment as A

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = f"{C.ROOT}/data/spoof/probe"; TMP = f"{C.WORK}/probe"


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def main(n_spk=6):
    os.makedirs(OUT, exist_ok=True); os.makedirs(TMP, exist_ok=True)
    rng = np.random.default_rng(C.SEED)
    utt = pd.read_csv(f"{C.PROTOCOLS}/fa_eval120_utts.csv"); man = pd.read_csv(f"{C.MANIFESTS}/cv22_fa_manifest.csv", usecols=["rel", "sentence", "gender"])
    utt = utt.merge(man, on="rel", how="left"); log("roles", utt.role.value_counts().to_dict())
    spks = rng.choice(sorted(utt.spk.unique()), n_spk, replace=False)
    ld = lambda rel: A.load(f"{C.CV_AUDIO}/{rel}")
    enc = SV.load_encoder("ecapa", device=DEV); enc.mods.embedding_model.load_state_dict(torch.load(f"{C.CKPT}/sv/stage2v2/best.pt", map_location=DEV)); enc.eval()
    cms = {}
    for n in ["lcnn", "rawnet2"]:
        m = CM.build(n).to(DEV); m.load_state_dict(torch.load(f"{C.CKPT}/cm/{n}_asv19la/best.pt", map_location=DEV)); cms[n] = m.eval()
    knn = torch.hub.load("bshall/knn-vc", "knn_vc", prematched=True, trust_repo=True, pretrained=True, device=DEV); log("knn-vc loaded")
    from transformers import VitsModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/mms-tts-fas"); tts = VitsModel.from_pretrained("facebook/mms-tts-fas").to(DEV).eval()
    log("mms loaded, sr", tts.config.sampling_rate, "uroman", getattr(tok, "is_uroman", None))

    def synth(text):
        ids = tok(text, return_tensors="pt").to(DEV)
        with torch.no_grad(): w = tts(**ids).waveform[0].float().cpu().numpy()
        if tts.config.sampling_rate != C.SR:
            import soxr; w = soxr.resample(w, tts.config.sampling_rate, C.SR)
        return A.peak_norm(w.astype(np.float32))

    def convert(src_wav, ref_paths, tag):
        p = f"{TMP}/{tag}_src.wav"; sf.write(p, src_wav, C.SR)
        with torch.no_grad():
            q = knn.get_features(p); ms = knn.get_matching_set(ref_paths); y = knn.match(q, ms, topk=4)
        return A.peak_norm(y.float().cpu().numpy().astype(np.float32))

    rows, wavs = [], []
    def add(spk, system, name, x, tpl, text=""):
        sf.write(f"{OUT}/{spk}_{system}_{name}.wav", x, C.SR); rows.append(dict(spk=spk, system=system, name=name, sec=len(x) / C.SR, text=text)); wavs.append((x, tpl))

    for spk in spks:
        g = utt[utt.spk == spk]; gender = g.gender.dropna().iloc[0] if g.gender.notna().any() else "?"
        tpl = SV.embed(enc, [ld(r) for r in g[g.role == "enrol"].rel]).mean(0); tpl /= np.linalg.norm(tpl)
        spare = g[~g.role.isin(["enrol", "test"])]
        if len(spare) < 4: log(spk, "no spare clips, skipping"); continue
        spare = spare.sample(min(12, len(spare)), random_state=1)
        ref_paths = []
        for i, r in enumerate(spare.rel):
            p = f"{TMP}/{spk}_ref{i}.wav"; sf.write(p, ld(r), C.SR); ref_paths.append(p)
        log(spk, gender, "refs", len(ref_paths), "ref sec", round(spare.sec.sum(), 1))
        for i, r in enumerate(g[(g.role == "test") & (g.cond == "clean")].rel[:2]): add(spk, "genuine", f"t{i}", ld(r), tpl)
        other = utt[(utt.spk != spk) & (utt.fold != g.fold.iloc[0]) & (utt.role == "test") & (utt.cond == "clean")].sample(2, random_state=int(rng.integers(1e6)))
        for i, r in enumerate(other.itertuples()):
            x = ld(r.rel); add(spk, "impostor", f"s{i}", x, tpl)
            add(spk, "knnvc", f"s{i}", convert(x, ref_paths, f"{spk}_vc{i}"), tpl)
        sents = man.sentence.dropna(); sents = sents[sents.str.len().between(25, 70)].sample(2, random_state=int(rng.integers(1e6)))
        for i, t in enumerate(sents):
            x = synth(t); add(spk, "mms", f"x{i}", x, tpl, t)
            add(spk, "mms_knnvc", f"x{i}", convert(x, ref_paths, f"{spk}_tts{i}"), tpl, t)
    df = pd.DataFrame(rows); E = SV.embed(enc, [w for w, _ in wavs]); df["sv_cos"] = [float(e @ t) for e, (_, t) in zip(E, wavs)]
    with torch.no_grad():
        for n, m in cms.items(): df[f"live_{n}"] = CM.score(m, [w for w, _ in wavs], device=DEV)
    df.to_csv(f"{C.REPORTS}/s5_probe_scores.csv", index=False)
    pd.set_option("display.width", 200); print(df.drop(columns="text").round(3).to_string(index=False))
    print(df.groupby("system")[["sv_cos", "live_lcnn", "live_rawnet2", "sec"]].agg(["mean", "min", "max"]).round(3).to_string())
    log("PROBE_DONE")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
