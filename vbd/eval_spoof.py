"""Score a Persian spoof set before any Persian CM training ("zero-shot" check).
  - speaker engine (stage2v2): cosine of each spoof against its target's enrolment template, next to genuine clean test trials
    and impostor trials of the same folds; share of spoofs above the pooled threshold that gives FAR 0.8% on impostors
  - ASVspoof-trained countermeasures: EER of spoof vs genuine Persian clean test clips, and spoof acceptance at the ASVspoof dev-EER threshold
    python -m vbd.eval_spoof replay
"""
import sys, time, numpy as np, pandas as pd, torch, soundfile as sf
from . import config as C, sv as SV, cm as CM, metrics as M, augment as A


def main(name):
    log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
    idx = pd.read_csv(f"{C.PROTOCOLS}/fa_spoof_{name}.csv"); utt = pd.read_csv(f"{C.PROTOCOLS}/fa_eval120_utts.csv")
    sp = [sf.read(f"{C.WORK}/spoof/{name}/{f}", dtype="float32")[0] for f in idx.file]
    en = utt[utt.role == "enrol"]; te = utt[(utt.role == "test") & (utt.cond == "clean")].reset_index(drop=True)
    ld = lambda rel: A.load(f"{C.CV_AUDIO}/{rel}")
    enc = SV.load_encoder("ecapa", device="cuda"); enc.mods.embedding_model.load_state_dict(torch.load(f"{C.CKPT}/sv/stage2v2/best.pt", map_location="cuda")); enc.eval()
    Ee = SV.embed(enc, [ld(r) for r in en.rel]); tw = [ld(r) for r in te.rel]; Et = SV.embed(enc, tw); Es = SV.embed(enc, sp)
    spks = sorted(en.spk.unique()); T = np.stack([Ee[(en.spk == s).values].mean(0) for s in spks]); T /= np.linalg.norm(T, axis=1, keepdims=True)
    pos = {s: i for i, s in enumerate(spks)}; fold = en.drop_duplicates("spk").set_index("spk").fold
    S = Et @ T.T; own = np.array([pos[s] for s in te.spk]); gen = S[np.arange(len(te)), own]
    same_fold = fold.loc[te.spk].values[:, None] == fold.loc[spks].values[None, :]; imp_mask = same_fold.copy(); imp_mask[np.arange(len(te)), own] = False
    imp = S[imp_mask]; thr = M.thr_at_far(imp, 0.008)
    spc = np.einsum("ij,ij->i", Es, T[[pos[s] for s in idx.spk]]); idx["sv_cos"] = spc
    rows = []
    for sysn, g in list(idx.groupby("system")) + [("all", idx)]:
        rows.append(dict(set=name, system=sysn, n=len(g), sv_cos_median=float(g.sv_cos.median()), sv_accept_pct=100 * float(np.mean(g.sv_cos >= thr))))
    log("SV pooled threshold %.3f | genuine clean median %.3f accept %.1f%% | impostor median %.3f" % (thr, np.median(gen), 100 * np.mean(gen >= thr), np.median(imp)))
    ref = pd.read_csv(f"{C.REPORTS}/s6a_cm_persian_bonafide_rejection.csv")
    with torch.no_grad():
        for n in ["lcnn", "rawnet2"]:
            m = CM.build(n).cuda(); m.load_state_dict(torch.load(f"{C.CKPT}/cm/{n}_asv19la/best.pt", map_location="cuda")); m.eval()
            bona = CM.score(m, tw); idx[f"live_{n}"] = CM.score(m, sp); dthr = float(ref[ref.model == n].dev_eer_threshold.iloc[0])
            for r in rows:
                g = idx if r["system"] == "all" else idx[idx.system == r["system"]]
                r[f"{n}_eer"] = 100 * M.eer(bona, g[f"live_{n}"].values)[0]; r[f"{n}_spoof_accept_at_asv_thr"] = 100 * float(np.mean(g[f"live_{n}"] >= dthr))
                r[f"{n}_pass_both_pct"] = 100 * float(np.mean((g[f"live_{n}"] >= dthr) & (g.sv_cos >= thr)))
    out = pd.DataFrame(rows); out.to_csv(f"{C.REPORTS}/s5_{name}_zeroshot.csv", index=False); idx.to_csv(f"{C.SCORES}/s5_{name}_zeroshot_scores.csv.gz", index=False)
    pd.set_option("display.width", 250); print(out.round(2).to_string(index=False)); log("EVAL_SPOOF_DONE")


if __name__ == "__main__":
    main(sys.argv[1])
