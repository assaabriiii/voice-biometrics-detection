"""Stage 1 for the countermeasure: train LCNN or RawNet2 on ASVspoof 2019 LA (train split), select on the dev split.
    python -m vbd.train_cm --model lcnn --epochs 30
Label convention: 1 = bona fide, 0 = spoof. Checkpoints to Drive after every epoch; --resume continues.
Augmentation is applied to BOTH classes (noise, telephone approximation) so that channel effects are not a spoof cue:
the Persian bona fide test audio is noisy / telephone-coded and must not be rejected for that reason."""
import argparse, io, json, math, os, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from . import config as C, augment as A, cache, cm as CM, metrics as M
from .train_sv import noise_files, make_noise

ASV = f"{C.RAW}/asvspoof2019_la/data"


def build_asv_cache(split, log=print):
    """Decode one ASVspoof parquet split into the int16 memmap store `asv_<split>` (label column: bona fide = 1)."""
    name = f"asv_{split}"; out = f"{C.WORK}/cache"; os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/{name}.csv"): return name
    pf = pq.ParquetFile(f"{ASV}/{split}-00000-of-00001.parquet"); off, rows = 0, []
    with open(f"{out}/{name}.i16.part", "wb") as f:
        for g in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(g).to_pandas()
            for r in t.itertuples():
                x, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="int16"); assert sr == 16000
                f.write(x.tobytes()); rows.append((r.speaker_id, r.audio_file_name, off, len(x), r.system_id, int(r.system_id == "-"))); off += len(x)
            if g % 50 == 0: log(f"{name}: row group {g}/{pf.metadata.num_row_groups}")
    os.replace(f"{out}/{name}.i16.part", f"{out}/{name}.i16")
    pd.DataFrame(rows, columns=["spk", "rel", "off", "len", "system", "label"]).to_csv(f"{out}/{name}.csv", index=False)
    return name


class CMSet(torch.utils.data.Dataset):
    def __init__(self, name, train=True, p_noise=0.3, p_tel=0.3):
        self.name, self.train, self.p_noise, self.p_tel = name, train, p_noise, p_tel
        self.idx = pd.read_csv(f"{C.WORK}/cache/{name}.csv"); self.store = None

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        if self.store is None:
            self.store = cache.Store(self.name); self.bank = noise_files("train")
            self.rng = np.random.default_rng((os.getpid() * 7919 + int(time.time() * 1000)) % 2 ** 32)
        x = self.store.get(i)
        if self.train:
            x = CM.fix_len(x, rng=self.rng)
            if self.bank["noise"] and self.rng.random() < self.p_noise:
                nz, snr = make_noise(self.bank, len(x), self.rng); x = A.add_noise(x, nz, max(snr, 5.0), self.rng)
            if self.rng.random() < self.p_tel: x = A.mulaw_8k(x)
            x = A.peak_norm(x)
        else:
            x = CM.fix_len(x)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), int(self.idx.label.iat[i])


@torch.no_grad()
def evaluate(model, name, bs=128, max_items=None):
    ds = CMSet(name, train=False)
    if max_items and len(ds) > max_items:
        ds.idx = ds.idx.sample(max_items, random_state=0).sort_index(); keep = ds.idx.index.values
        sub = torch.utils.data.Subset(CMSet(name, train=False), keep.tolist()); labels = ds.idx.label.values; ds = sub
    else:
        labels = ds.idx.label.values
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=2)
    model.eval(); sc = []
    for x, _ in dl:
        with torch.autocast("cuda"):
            lg = model(x.cuda()).float()
        sc.append((lg[:, 1] - lg[:, 0]).cpu().numpy())
    sc = np.concatenate(sc)
    return 100 * M.eer(sc[labels == 1], sc[labels == 0])[0], sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lcnn"); ap.add_argument("--name", default="")
    ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(); name = a.name or f"{a.model}_asv19la"
    out = f"{C.CKPT}/cm/{name}"; os.makedirs(out, exist_ok=True)
    def log(**kw):
        kw["t"] = time.strftime("%H:%M:%S"); print(json.dumps(kw), flush=True); open(f"{out}/log.jsonl", "a").write(json.dumps(kw) + "\n")
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    tr = build_asv_cache("train", lambda m: log(event="cache", msg=m)); dv = build_asv_cache("validation", lambda m: log(event="cache", msg=m))
    ds = CMSet(tr); lab = ds.idx.label.values
    w = np.where(lab == 1, 0.5 / (lab == 1).mean(), 0.5 / (lab == 0).mean())      # balanced sampling of bona fide / spoof
    sampler = torch.utils.data.WeightedRandomSampler(torch.from_numpy(w), num_samples=len(ds), replacement=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, sampler=sampler, num_workers=2, drop_last=True, persistent_workers=True, prefetch_factor=4)
    model = CM.build(a.model).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4); total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 300) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda"); ep0, best = 0, 1e9
    if a.resume and os.path.exists(f"{out}/last.pt"):
        ck = torch.load(f"{out}/last.pt", map_location="cuda")
        model.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); ep0, best = ck["epoch"] + 1, ck["best"]
        log(event="resumed", epoch=ep0)
    for ep in range(ep0, a.epochs):
        model.train(); t0 = time.time(); ls = []
        for it, (x, y) in enumerate(dl):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.autocast("cuda"):
                loss = F.cross_entropy(model(x).float(), y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); scaler.step(opt); scaler.update(); sched.step(); ls.append(loss.item())
            if it % 100 == 0: log(event="it", epoch=ep, it=it, n_it=len(dl), loss=float(np.mean(ls[-100:])), sec=round(time.time() - t0))
        eer, _ = evaluate(model, dv)
        log(event="epoch", epoch=ep, loss=float(np.mean(ls)), dev_eer=eer, sec=round(time.time() - t0))
        if eer < best:
            best = eer; torch.save(model.state_dict(), f"{out}/best.pt.tmp"); os.replace(f"{out}/best.pt.tmp", f"{out}/best.pt"); log(event="saved_best", epoch=ep, dev_eer=eer)
        torch.save(dict(net=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, best=best), f"{out}/last.pt.tmp"); os.replace(f"{out}/last.pt.tmp", f"{out}/last.pt")
    log(event="done", best=best)


if __name__ == "__main__":
    main()
