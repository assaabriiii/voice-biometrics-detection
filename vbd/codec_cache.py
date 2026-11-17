"""Pre-compute a telephone-codec copy of a cached audio store (same index/offsets), for training-time augmentation.
    python -m vbd.codec_cache pool amr_nb
Writes <WORK>/cache/<store>_<codec>.i16 and copies it to Drive (features/cache/) so a new runtime can reuse it."""
import os, shutil, sys, time, numpy as np
from . import config as C, augment as A, cache


def main(store_name, codec, block=400):
    st = cache.Store(store_name); out = f"{C.WORK}/cache/{store_name}_{codec}.i16"; drv = f"{C.FEATURES}/cache/{store_name}_{codec}.i16"
    os.makedirs(os.path.dirname(drv), exist_ok=True)
    if os.path.exists(out): print("exists", out); return
    if os.path.exists(drv): shutil.copy(drv, out); print("restored from Drive", out); return
    mm = np.memmap(out + ".part", dtype=np.int16, mode="w+", shape=st.mm.shape); t0 = time.time()
    for b in range(0, len(st.idx), block):
        rows = st.idx.iloc[b:b + block]
        ys = A.codec_batch([A.bandlimit(st.get(i)) for i in range(b, b + len(rows))], codec)
        for r, y in zip(rows.itertuples(), ys):
            mm[r.off:r.off + r.len] = (np.clip(y, -1, 1) * 32767).astype(np.int16)
        print(f"{time.strftime('%H:%M:%S')} {codec}: {b + len(rows)}/{len(st.idx)} clips, {round(time.time() - t0)} s", flush=True)
    mm.flush(); del mm; os.replace(out + ".part", out)
    shutil.copy(out, drv + ".part"); os.replace(drv + ".part", drv); print("CODEC_DONE", codec, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
