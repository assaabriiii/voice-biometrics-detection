"""Evaluate the ASVspoof-trained countermeasures:
  1) ASVspoof 2019 LA evaluation split (71,237 utterances, attacks A07-A19, none seen in training): pooled and per-attack EER;
  2) Persian bona fide audio from FA-EVAL-120 in every test condition: how often would genuine Persian speech be flagged as spoof
     at the threshold that gives the EER on the ASVspoof validation split? (domain / channel mismatch check, no Persian spoofs yet).
    python -m vbd.eval_cm
"""
import json, os, time, numpy as np, pandas as pd, torch
from . import config as C, cm as CM, metrics as M, cache, conditions as CD
from .train_cm import build_asv_cache, CMSet


@torch.no_grad()
def score_store(model, name, bs=128):
    ds = CMSet(name, train=False); dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=2)
    model.eval(); out = []
    for x, _ in dl:
        with torch.autocast("cuda"):
            lg = model(x.cuda()).float()
        out.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    return np.concatenate(out), ds.idx


def main():
    t0 = time.time(); log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
    dev = build_asv_cache("validation", log); ev = build_asv_cache("test", log); log("caches ready", round(time.time() - t0), "s")
    utt = pd.read_csv(f"{C.PROTOCOLS}/fa_eval120_utts.csv"); items, wavs = CD.build(utt); log("persian conditions ready")
    rows, pers = [], []
    for name in ["lcnn", "rawnet2"]:
        ck = f"{C.CKPT}/cm/{name}_asv19la/best.pt"
        if not os.path.exists(ck): log("missing", ck); continue
        model = CM.build(name).cuda(); model.load_state_dict(torch.load(ck, map_location="cuda")); model.eval()
        sd, idx_d = score_store(model, dev); eer_d, thr = M.eer(sd[idx_d.label.values == 1], sd[idx_d.label.values == 0])
        se, idx_e = score_store(model, ev); lab = idx_e.label.values
        eer_e, _ = M.eer(se[lab == 1], se[lab == 0])
        far_e, frr_e = M.far_frr(se[lab == 1], se[lab == 0], thr)          # at the dev-EER threshold: spoof accepted / bona fide rejected
        rows.append(dict(model=name, split="dev", attack="all", n_bona=int((idx_d.label == 1).sum()), n_spoof=int((idx_d.label == 0).sum()), eer=100 * eer_d))
        rows.append(dict(model=name, split="eval", attack="all", n_bona=int((lab == 1).sum()), n_spoof=int((lab == 0).sum()), eer=100 * eer_e,
                         spoof_accept_at_dev_thr=100 * far_e, bona_reject_at_dev_thr=100 * frr_e))
        for a in sorted(set(idx_e.system) - {"-"}):
            m = (idx_e.system.values == a)
            rows.append(dict(model=name, split="eval", attack=a, n_bona=int((lab == 1).sum()), n_spoof=int(m.sum()), eer=100 * M.eer(se[lab == 1], se[m])[0],
                             spoof_accept_at_dev_thr=100 * float(np.mean(se[m] >= thr))))
        np.savez_compressed(f"{C.SCORES}/s6a_{name}_asv19la_eval_scores.npz", score=se, label=lab, system=idx_e.system.values, key=idx_e.rel.values)
        keys = items.key.tolist(); sp = CM.score(model, [wavs[k] for k in keys]); it = items.assign(live=sp)
        it.to_csv(f"{C.SCORES}/s6a_{name}_persian_bonafide_scores.csv.gz", index=False)
        for cnd, g in it.groupby("cond"):
            pers.append(dict(model=name, cond=cnd, n=len(g), dev_eer_threshold=thr, bona_reject_pct=100 * float(np.mean(g.live < thr)),
                             live_median=float(g.live.median()), live_p05=float(g.live.quantile(0.05))))
        log(name, "dev EER %.2f eval EER %.2f" % (100 * eer_d, 100 * eer_e)); del model; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(f"{C.REPORTS}/s6a_cm_asvspoof2019la.csv", index=False)
    pd.DataFrame(pers).to_csv(f"{C.REPORTS}/s6a_cm_persian_bonafide_rejection.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False)); print(pd.DataFrame(pers).round(2).to_string(index=False)); log("EVAL_CM_DONE")


if __name__ == "__main__":
    main()
