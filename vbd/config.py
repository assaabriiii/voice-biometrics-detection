"""Central paths and constants."""
import os
ROOT = "/content/drive/MyDrive/voice-biometrics-detection"
WORK = "/content/work"                      # fast local scratch, rebuilt from Drive archives
RAW = f"{ROOT}/data/raw"
MANIFESTS = f"{ROOT}/data/manifests"
PROTOCOLS = f"{ROOT}/data/protocols"
SPOOF = f"{ROOT}/data/spoof"
FEATURES = f"{ROOT}/features"
CKPT = f"{ROOT}/checkpoints"
SCORES = f"{ROOT}/results/scores"
REPORTS = f"{ROOT}/results/reports"
FIGURES = f"{ROOT}/results/figures"
LOGS = f"{ROOT}/logs"
CV_AUDIO = f"{WORK}/cv_fa"                  # extracted Common Voice mp3s
SR = 16000
SEED = 20261116
N_EVAL_SPK, N_FOLDS = 120, 5
N_ENROL, N_TEST = 5, 15                     # clips per eval speaker
MIN_SEC, MAX_SEC = 3.0, 12.0                # eligible clip duration
MIN_ELIGIBLE = 30                           # eligible clips needed to be an eval-speaker candidate
SESSION_GAP = 20000                         # clip-id gap treated as a new "session" (proxy)
POOL_MIN_CLIPS = 8                          # min clips for a speaker to enter the fine-tuning pool
for _d in (MANIFESTS, PROTOCOLS, SPOOF, FEATURES, CKPT, SCORES, REPORTS, FIGURES, LOGS):
    os.makedirs(_d, exist_ok=True)
