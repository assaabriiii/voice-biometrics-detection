"""Decode many clips once into a flat int16 memmap (16 kHz) so training does not decode mp3 on the fly."""
import os, numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from . import config as C, augment as A


def build(df, name, audio_root=C.CV_AUDIO, out_dir=f"{C.WORK}/cache", workers=4, log=print):
    """df needs columns: rel (path under audio_root), spk. Writes <name>.i16 and <name>.csv (offset, length)."""
    os.makedirs(out_dir, exist_ok=True)
    binp, idxp = f"{out_dir}/{name}.i16", f"{out_dir}/{name}.csv"
    if os.path.exists(idxp):
        return idxp
    off, rows = 0, []
    with open(binp + ".part", "wb") as f, ThreadPoolExecutor(workers) as ex:
        for i, (r, x) in enumerate(zip(df.itertuples(), ex.map(lambda p: A.load(f"{audio_root}/{p}"), df.rel))):
            y = (np.clip(x, -1, 1) * 32767).astype(np.int16); f.write(y.tobytes())
            rows.append((r.spk, r.rel, off, len(y))); off += len(y)
            if i % 5000 == 0: log(f"cache {name}: {i}/{len(df)}")
    os.replace(binp + ".part", binp)
    pd.DataFrame(rows, columns=["spk", "rel", "off", "len"]).to_csv(idxp, index=False)
    return idxp


class Store:
    def __init__(self, name, out_dir=f"{C.WORK}/cache"):
        self.idx = pd.read_csv(f"{out_dir}/{name}.csv")
        self.mm = np.memmap(f"{out_dir}/{name}.i16", dtype=np.int16, mode="r")

    def get(self, i, start=0, n=None):
        r = self.idx.iloc[i]
        n = r.len - start if n is None else min(n, r.len - start)
        return self.mm[r.off + start: r.off + start + n].astype(np.float32) / 32767.0
