"""Stage 3 + 4 of the proposal, per cross-validation rotation k:
  - start from the stage-2 model (shared, trained without any evaluation speaker);
  - adapt ONLY the last layers (MFA, attentive pooling, embedding layer) with a low learning rate on the adaptation folds
    (72 speakers, their non-protocol clips, incl. real-codec copies), mixed 50/50 with the Persian pool to avoid forgetting;
  - early stopping on the validation fold (main15 EER on the real test conditions); "no adaptation" is one of the candidates;
  - save validation-fold and test-fold embeddings of the selected model. Thresholds are set later on the validation fold.
Test-fold speakers are never used. Adaptation speakers suspected to be the same person as a validation/test speaker are dropped.
    python -m vbd.train_sv3 --base stage2v2 --name stage3 --rot 0
"""
import argparse, json, math, os, pickle, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from . import config as C, sv as SV, metrics as M, conditions as CD
from .data import rotation
from .train_sv import TrainSet, AAM, forward_emb

LAST = ("mfa", "asp", "asp_bn", "fc")


class MixSet(torch.utils.data.Dataset):
    """Each item comes from the adaptation set with probability p_adapt, otherwise from the pool. Labels are offset."""
    def __init__(self, pool_set, adapt_set, n_items, p_adapt=0.5):
        self.a, self.b, self.n, self.p = pool_set, adapt_set, n_items, p_adapt; self.off = len(pool_set.spk)

    def __len__(self): return self.n

    def __getitem__(self, i):
        if np.random.default_rng((os.getpid() * 31 + i + int(time.time() * 1e3)) % 2 ** 32).random() < self.p:
            x, y = self.b[i]; return x, y + self.off
        return self.a[i]


def val_eer(enc, utt, items, wavs, fold):
    """main15 EER of one fold (its own enrolment templates and test items)."""
    it = items[(items.fold == fold) & items.cond.isin(["clean", "noisy", "tel"])]
    en = utt[(utt.fold == fold) & (utt.role == "enrol")].assign(key=lambda d: "enrol|" + d.path)
    keys = en.key.tolist() + it.key.tolist(); emb = dict(zip(keys, SV.embed(enc, [wavs[k] for k in keys])))
    tr = SV.score_trials(en, emb, it)
    return 100 * M.eer(tr.score[tr.target == 1].values, tr.score[tr.target == 0].values)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="stage2v2"); ap.add_argument("--name", default="stage3"); ap.add_argument("--rot", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--iters", type=int, default=300); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--head_lr", type=float, default=1e-3); ap.add_argument("--p_adapt", type=float, default=0.5)
    a = ap.parse_args(); k = a.rot; r = rotation(k)
    out = f"{C.CKPT}/sv/{a.name}/rot{k}"; os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/done.json"): print("rotation already done", k); return
    def log(**kw):
        kw["t"] = time.strftime("%H:%M:%S"); kw["rot"] = k; print(json.dumps(kw), flush=True); open(f"{out}/log.jsonl", "a").write(json.dumps(kw) + "\n")
    torch.manual_seed(C.SEED + k); np.random.seed(C.SEED + k)
    utt = pd.read_csv(f"{C.PROTOCOLS}/fa_eval120_utts.csv"); spk = pd.read_csv(f"{C.PROTOCOLS}/fa_eval120_speakers.csv")
    items, wavs = CD.build(utt)
    held = set(spk.spk[spk.fold.isin([r["val"], r["test"]])]); dup = pd.read_csv(f"{C.PROTOCOLS}/eval_dup_suspect_pairs.csv")
    drop = set(dup.spk_a[dup.spk_b.isin(held)]) | set(dup.spk_b[dup.spk_a.isin(held)])
    adapt_spk = sorted(set(spk.spk[spk.fold.isin(r["adapt"])]) - drop)
    base = json.load(open(f"{C.CKPT}/sv/{a.base}/split.json")); pool_spk = base["train_spk"]
    assert not (set(adapt_spk) & held) and not (set(pool_spk) & set(spk.spk))
    log(event="start", val_fold=r["val"], test_fold=r["test"], n_adapt=len(adapt_spk), dropped_dup=sorted(drop & set(spk.spk[spk.fold.isin(r["adapt"])])), n_pool=len(pool_spk))
    codecs = ["amr_nb", "gsm_fr"]
    ds = MixSet(TrainSet("pool", pool_spk, a.iters * a.bs, codecs=codecs, p_codec=0.3, p_mulaw=0.2),
                TrainSet("adapt", adapt_spk, a.iters * a.bs, codecs=codecs, p_codec=0.4, p_mulaw=0.2), a.iters * a.bs, a.p_adapt)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=2, drop_last=True, persistent_workers=True, prefetch_factor=4)
    enc = SV.load_encoder("ecapa"); net = enc.mods.embedding_model
    net.load_state_dict(torch.load(f"{C.CKPT}/sv/{a.base}/best.pt", map_location="cuda"))
    for n, p in net.named_parameters(): p.requires_grad_(n.split(".")[0] in LAST)
    head = AAM(192, len(pool_spk) + len(adapt_spk)).cuda()
    ck = torch.load(f"{C.CKPT}/sv/{a.base}/last.pt", map_location="cuda")
    with torch.no_grad(): head.W[:len(pool_spk)] = ck["head"]["W"]          # classes of the pool keep their trained weights
    del ck
    def set_mode(train):
        net.eval()
        if train:
            for n, m in net.named_children():
                if n in LAST: m.train()
    net.eval(); e0 = val_eer(enc, utt, items, wavs, r["val"]); log(event="val_base", val_eer=e0)
    best, best_ep = e0, -1; torch.save(net.state_dict(), f"{out}/best.pt")
    opt = torch.optim.AdamW([{"params": [p for p in net.parameters() if p.requires_grad], "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=1e-4)
    total = a.epochs * a.iters
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 100) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda")
    for ep in range(a.epochs):
        set_mode(True); t0 = time.time(); ls, ac = [], []
        for it, (x, y) in enumerate(dl):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            loss, acc = head(forward_emb(enc, x), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 5.0); scaler.step(opt); scaler.update(); sched.step()
            ls.append(loss.item()); ac.append(acc.item())
        set_mode(False); e = val_eer(enc, utt, items, wavs, r["val"])
        log(event="epoch", epoch=ep, loss=float(np.mean(ls)), acc=float(np.mean(ac)), val_eer=e, sec=round(time.time() - t0))
        if e < best:
            best, best_ep = e, ep; torch.save(net.state_dict(), f"{out}/best.pt.tmp"); os.replace(f"{out}/best.pt.tmp", f"{out}/best.pt")
    # embeddings of validation + test folds with the selected model (no test information was used to select it)
    net.load_state_dict(torch.load(f"{out}/best.pt", map_location="cuda")); net.eval()
    en = utt[utt.fold.isin([r["val"], r["test"]]) & (utt.role == "enrol")]
    keys = ["enrol|" + p for p in en.path] + items[items.fold.isin([r["val"], r["test"]])].key.tolist()
    E = SV.embed(enc, [wavs[q] for q in keys]); np.savez_compressed(f"{C.FEATURES}/s4_{a.name}_rot{k}_emb.npz", keys=np.array(keys), emb=E)
    json.dump(dict(rot=k, val_base=e0, val_best=best, best_epoch=best_ep, n_adapt=len(adapt_spk)), open(f"{out}/done.json", "w"))
    log(event="done", val_base=e0, val_best=best, best_epoch=best_ep)


if __name__ == "__main__":
    main()
