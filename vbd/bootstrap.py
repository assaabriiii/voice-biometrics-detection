"""Rebuild the Colab working directory from the archives on Drive. Safe to re-run; skips what is already there.

On a fresh Colab runtime (after a disconnect or usage limit):
    from google.colab import drive; drive.mount('/content/drive')
    !cd /content/drive/MyDrive/voice-biometrics-detection && python -m vbd.bootstrap
    !cd /content/drive/MyDrive/voice-biometrics-detection && nohup python -m vbd.train_sv --name stage2 --epochs 12 --resume >> logs/train_sv_stage2.log 2>&1 &
Nothing is downloaded again: all archives are read from data/raw on Drive."""
import glob, os, shutil, subprocess, sys, time

ROOT = "/content/drive/MyDrive/voice-biometrics-detection"
RAW, W = f"{ROOT}/data/raw", "/content/work"


def log(*a): print(time.strftime("%H:%M:%S"), *a, flush=True)


def sh(cmd): subprocess.run(cmd, shell=True, check=True)


def main(with_noise=True, with_pool_cache=True):
    os.makedirs(W, exist_ok=True)
    try:
        import speechbrain, soxr  # noqa
    except ImportError:
        sh(f"{sys.executable} -m pip -q install speechbrain soundfile soxr"); log("pip packages installed")
    if len(glob.glob(f"{W}/cv_fa/*/*.mp3")) < 85000:
        os.makedirs(f"{W}/cv_fa", exist_ok=True)
        for sp in ["train", "dev", "test", "other"]:
            sh(f"tar -xf {RAW}/cv22_fa/fa_{sp}_0.tar -C {W}/cv_fa"); log("extracted common voice", sp)
    if with_noise and not os.path.isdir(f"{W}/RIRS_NOISES/simulated_rirs"):
        sh(f"unzip -q -o {RAW}/rirs_noises.zip -d {W}"); log("extracted RIRS_NOISES")
    if with_noise and not os.path.isdir(f"{W}/musan/noise"):
        sh(f"tar -xzf {RAW}/musan.tar.gz -C {W}"); log("extracted MUSAN")
    if os.path.exists(f"{ROOT}/tools/ffmpeg") and not os.path.exists(f"{W}/ffmpeg_static"):
        shutil.copy(f"{ROOT}/tools/ffmpeg", f"{W}/ffmpeg_static"); os.chmod(f"{W}/ffmpeg_static", 0o755); log("static ffmpeg ready")
    if with_pool_cache and not os.path.exists(f"{W}/cache/pool.csv"):
        sys.path.insert(0, ROOT)
        import pandas as pd
        from vbd import config as C, cache
        pool = pd.read_csv(f"{C.MANIFESTS}/fa_pool.csv")
        files = {os.path.basename(p): os.path.relpath(p, C.CV_AUDIO) for p in glob.glob(f"{C.CV_AUDIO}/**/*.mp3", recursive=True)}
        pool["rel"] = pool.path.map(files); cache.build(pool, "pool", log=log); log("pool audio cache built")
    log("BOOTSTRAP_DONE")


if __name__ == "__main__":
    main()
