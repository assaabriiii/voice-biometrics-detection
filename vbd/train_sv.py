"""Stage 2: fine-tune pretrained ECAPA-TDNN on Persian (Common Voice pool) with AAM-Softmax + noise/reverb/telephone augmentation.
Run:  python -m vbd.train_sv --name stage2 --epochs 12
Checkpoints and a JSON-lines log go to Drive, so a dropped Colab session can resume with --resume."""
import argparse, glob, json, math, os, sys, time
import numpy as np, pandas as pd, soundfile as sf, torch, torch.nn as nn, torch.nn.functional as F
from . import config as C, augment as A, cache, sv as SV, metrics as M


def noise_files(split):
    """MUSAN / RIR file lists. Every 5th file (sorted) is reserved for evaluation conditions; training never sees it."""
    out = {}
    for cat, pat in [("noise", f"{C.WORK}/musan/noise/**/*.wav"), ("music", f"{C.WORK}/musan/music/**/*.wav"),
                     ("speech", f"{C.WORK}/musan/speech/**/*.wav"), ("rir", f"{C.WORK}/RIRS_NOISES/simulated_rirs/**/*.wav")]:
        fs = sorted(glob.glob(pat, recursive=True))
        out[cat] = [f for i, f in enumerate(fs) if (i % 5 == 0) == (split == "eval")]
    return out


def read_segment(path, n, rng):
    info = sf.info(path)
    if info.frames <= n:
        x, _ = sf.read(path, dtype="float32", always_2d=True)
    else:
        s = int(rng.integers(0, info.frames - n))
        x, _ = sf.read(path, start=s, frames=n, dtype="float32", always_2d=True)
    return x[:, 0]


def make_noise(bank, n, rng):
    """Returns (noise segment, snr_db) from one of: environmental noise, music, babble."""
    cat = rng.choice(["noise", "music", "babble"], p=[0.5, 0.2, 0.3])
    if cat == "babble":
        k = int(rng.integers(3, 7))
        segs = [A._fit(read_segment(bank["speech"][rng.integers(len(bank["speech"]))], n, rng), n, rng) for _ in range(k)]
        return np.sum(segs, 0), float(rng.uniform(8, 20))
    return read_segment(bank[cat][rng.integers(len(bank[cat]))], n, rng), float(rng.uniform(0, 20) if cat == "noise" else rng.uniform(5, 20))


def augment(x, bank, rng, p_rev=0.25, p_noise=0.6, p_tel=0.35):
    if bank["rir"] and rng.random() < p_rev:
        x = A.reverb(x, read_segment(bank["rir"][rng.integers(len(bank["rir"]))], 10 ** 9, rng))
    if bank["noise"] and rng.random() < p_noise:
        nz, snr = make_noise(bank, len(x), rng); x = A.add_noise(x, nz, snr, rng)
    if rng.random() < p_tel:
        x = A.mulaw_8k(x)
    return A.peak_norm(x)


class TrainSet(torch.utils.data.Dataset):
    """Speaker-uniform sampling: every item draws a random speaker, then a random clip and crop.
    With `codecs`, a share `p_codec` of items is taken from pre-computed real-codec copies of the same audio
    (vbd.codec_cache); the rest goes through the normal reverb/noise/mu-law augmentation."""
    def __init__(self, store_name, spk_list, n_items, crop_sec=3.0, aug=True, codecs=(), p_codec=0.3, p_mulaw=0.35):
        self.name, self.n_items, self.crop, self.aug = store_name, n_items, int(crop_sec * C.SR), aug
        self.codecs, self.p_codec, self.p_mulaw = list(codecs), p_codec, p_mulaw
        idx = pd.read_csv(f"{C.WORK}/cache/{store_name}.csv")
        self.spk = list(spk_list); self.lab = {s: i for i, s in enumerate(self.spk)}
        self.rows = {s: np.flatnonzero(idx.spk.values == s) for s in self.spk}
        self.len, self.off = idx.len.values, idx.off.values; self.store = None; self.bank = None

    def __len__(self): return self.n_items

    def _crop(self, mm, r, start, L):
        x = mm[self.off[r] + start: self.off[r] + start + min(self.crop, L)].astype(np.float32) / 32767.0
        return x if len(x) == self.crop else np.resize(x, self.crop)

    def __getitem__(self, i):
        if self.store is None:
            self.store = cache.Store(self.name); self.bank = noise_files("train")
            self.cmm = {c: np.memmap(f"{C.WORK}/cache/{self.name}_{c}.i16", dtype=np.int16, mode="r") for c in self.codecs}
            self.rng = np.random.default_rng((os.getpid() * 7919 + int(time.time() * 1000)) % 2 ** 32)
        rng = self.rng
        s = self.spk[rng.integers(len(self.spk))]; r = int(rng.choice(self.rows[s])); L = int(self.len[r])
        start = int(rng.integers(0, L - self.crop)) if L > self.crop else 0
        if self.codecs and rng.random() < self.p_codec:
            x = self._crop(self.cmm[self.codecs[rng.integers(len(self.codecs))]], r, start, L)
            if self.aug and rng.random() < 0.3:
                nz, snr = make_noise(self.bank, len(x), rng); x = A.add_noise(x, nz, max(snr, 10.0), rng)
            x = A.peak_norm(x)
        else:
            x = self._crop(self.store.mm, r, start, L)
            if self.aug: x = augment(x, self.bank, rng, p_tel=self.p_mulaw)
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)), self.lab[s]


class AAM(nn.Module):
    def __init__(self, dim, n_cls, m=0.2, s=30.0):
        super().__init__(); self.W = nn.Parameter(torch.randn(n_cls, dim) * 0.01); self.m, self.s = m, s

    def forward(self, e, y):
        cos = F.linear(F.normalize(e.float()), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        phi = torch.cos(torch.acos(cos) + self.m)
        logits = self.s * torch.where(F.one_hot(y, cos.size(1)).bool(), phi, cos)
        return F.cross_entropy(logits, y), (logits.argmax(1) == y).float().mean()


def forward_emb(enc, wav, amp=True):
    lens = torch.ones(wav.size(0), device=wav.device)
    with torch.no_grad():
        feats = enc.mods.mean_var_norm(enc.mods.compute_features(wav), lens)
    with torch.autocast("cuda", enabled=amp):
        return enc.mods.embedding_model(feats, lens).squeeze(1)


def dev_eval(enc, store, dev_spk, n_enrol=3, max_clips=12, seed=0):
    """Held-out pool speakers: EER on original audio and on the fast telephone approximation."""
    rng = np.random.default_rng(seed); idx = store.idx; rows = []
    for s in dev_spk:
        r = np.flatnonzero(idx.spk.values == s); r = rng.permutation(r)[:max_clips]
        rows += [(s, int(i), "enrol" if k < n_enrol else "test") for k, i in enumerate(r)]
    df = pd.DataFrame(rows, columns=["spk", "row", "role"])
    wav = [store.get(i) for i in df.row]
    res = {}
    for cond, w in [("orig", wav), ("tel", [A.mulaw_8k(x) for x in wav])]:
        E = SV.embed(enc, [wav[k] if df.role[k] == "enrol" else w[k] for k in range(len(df))])
        spks = sorted(df.spk.unique())
        T = np.stack([E[(df.spk == s) & (df.role == "enrol")].mean(0) for s in spks]); T /= np.linalg.norm(T, axis=1, keepdims=True)
        te = df[df.role == "test"]; S = E[te.index] @ T.T
        tgt = te.spk.values[:, None] == np.array(spks)[None]
        res[cond] = 100 * M.eer(S[tgt], S[~tgt])[0]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="stage2"); ap.add_argument("--store", default="pool")
    ap.add_argument("--epochs", type=int, default=12); ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_lr", type=float, default=2e-3); ap.add_argument("--n_dev", type=int, default=60)
    ap.add_argument("--init", default=""); ap.add_argument("--exclude", default=""); ap.add_argument("--codecs", default=""); ap.add_argument("--p_codec", type=float, default=0.3); ap.add_argument("--p_mulaw", type=float, default=0.35); ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    out = f"{C.CKPT}/sv/{a.name}"; os.makedirs(out, exist_ok=True)
    def log(**kw):
        kw["t"] = time.strftime("%H:%M:%S"); print(json.dumps(kw), flush=True)
        open(f"{out}/log.jsonl", "a").write(json.dumps(kw) + "\n")
    torch.manual_seed(C.SEED); np.random.seed(C.SEED)
    store = cache.Store(a.store)
    counts = store.idx.groupby("spk").size()
    rng = np.random.default_rng(C.SEED)
    dev_spk = sorted(rng.choice(counts[counts >= 10].index.values, a.n_dev, replace=False))
    excl = set(pd.read_csv(a.exclude).spk) if a.exclude else set()
    train_spk = sorted(set(counts.index) - set(dev_spk) - excl)
    json.dump(dict(dev_spk=dev_spk, train_spk=train_spk, args=vars(a)), open(f"{out}/split.json", "w"))
    ds = TrainSet(a.store, train_spk, a.iters * a.bs, codecs=[c for c in a.codecs.split(",") if c], p_codec=a.p_codec, p_mulaw=a.p_mulaw)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=2, drop_last=True, persistent_workers=True, prefetch_factor=4)
    enc = SV.load_encoder("ecapa"); net = enc.mods.embedding_model
    for p in net.parameters(): p.requires_grad_(True)
    if a.init: net.load_state_dict(torch.load(a.init, map_location="cuda"))
    head = AAM(192, len(train_spk)).cuda()
    opt = torch.optim.AdamW([{"params": net.parameters(), "lr": a.lr}, {"params": head.parameters(), "lr": a.head_lr}], weight_decay=1e-4)
    total = a.epochs * a.iters
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    scaler = torch.amp.GradScaler("cuda"); ep0, best = 0, 1e9
    if a.resume and os.path.exists(f"{out}/last.pt"):
        ck = torch.load(f"{out}/last.pt", map_location="cuda")
        net.load_state_dict(ck["net"]); head.load_state_dict(ck["head"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        ep0, best = ck["epoch"] + 1, ck["best"]; log(event="resumed", epoch=ep0)
    else:
        net.eval(); d = dev_eval(enc, store, dev_spk); log(event="dev_pretrained", **d)
    for ep in range(ep0, a.epochs):
        frozen = ep == 0                      # epoch 0: train only the new classifier head
        net.train(not frozen); t0 = time.time(); ls, ac = [], []
        for it, (x, y) in enumerate(dl):
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            with torch.set_grad_enabled(not frozen):
                e = forward_emb(enc, x)
            loss, acc = head(e, y)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); scaler.step(opt); scaler.update(); sched.step()
            ls.append(loss.item()); ac.append(acc.item())
            if it % 100 == 0: log(event="it", epoch=ep, it=it, loss=float(np.mean(ls[-100:])), acc=float(np.mean(ac[-100:])), sec=round(time.time() - t0))
        net.eval(); d = dev_eval(enc, store, dev_spk); score = (d["orig"] + d["tel"]) / 2
        log(event="epoch", epoch=ep, loss=float(np.mean(ls)), acc=float(np.mean(ac)), dev_orig=d["orig"], dev_tel=d["tel"], sec=round(time.time() - t0))
        if score < best and not frozen:
            best = score; torch.save(net.state_dict(), f"{out}/best.pt.tmp"); os.replace(f"{out}/best.pt.tmp", f"{out}/best.pt"); log(event="saved_best", epoch=ep, score=score)
        torch.save(dict(net=net.state_dict(), head=head.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep, best=best), f"{out}/last.pt.tmp"); os.replace(f"{out}/last.pt.tmp", f"{out}/last.pt")   # atomic: a disconnect mid-write cannot corrupt the resume point
    log(event="done", best=best)


if __name__ == "__main__":
    main()
