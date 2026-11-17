# Resuming after a Colab disconnect or usage limit

Everything needed to continue is on Drive. The Colab disk (`/content/work`) is disposable and is rebuilt from the archives in `data/raw`; nothing is downloaded again.

## What is saved, and when

| Item | Where on Drive | Written |
|---|---|---|
| Resume point of a training run (model, classifier head, optimiser, schedule, epoch, best score) | `checkpoints/sv/<run>/last.pt` | end of **every epoch**, atomically |
| Best model so far | `checkpoints/sv/<run>/best.pt` | whenever the dev score improves |
| Per-iteration and per-epoch log | `checkpoints/sv/<run>/log.jsonl` | continuously |
| Train/dev speaker split and run arguments | `checkpoints/sv/<run>/split.json` | at start |
| Code | `vbd/` | — |
| Protocols, manifests, scores, reports | `data/protocols`, `data/manifests`, `results/` | when produced |

At most one epoch (about 4-5 minutes) of work is lost on a disconnect.

## Steps on a fresh runtime (GPU runtime)

```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
cd /content/drive/MyDrive/voice-biometrics-detection && python -m vbd.bootstrap      # about 10-15 min: extracts Common Voice, RIRs, MUSAN, builds the audio cache
cd /content/drive/MyDrive/voice-biometrics-detection && nohup python -m vbd.train_sv --name stage2 --epochs 12 --iters 400 --bs 64 --resume >> logs/train_sv_stage2.log 2>&1 &
```

`--resume` continues from `last.pt`; if the run already finished it exits immediately after logging `done`.

## Check progress

```bash
grep -E '"event": "(epoch|saved_best|done|resumed)"' /content/drive/MyDrive/voice-biometrics-detection/checkpoints/sv/stage2/log.jsonl
pgrep -af "[v]bd.train_sv"
```

## Not on Drive (rebuilt when needed)

- `/content/work/eval_conditions.pkl` — test-condition audio for FA-EVAL-120; `vbd.conditions.build` regenerates it deterministically (a few minutes).
- `/content/work/cache/pool.i16` — decoded training audio; rebuilt by `vbd.bootstrap`.


## Current run: stage2v2 (started 2026-11-16 about 18:52 UTC)

Needs the pre-computed codec copies of the pool. They are copied to Drive (`features/cache/pool_amr_nb.i16`, `pool_gsm_fr.i16`, 3.1 GB each) when finished; `vbd.codec_cache` restores them from Drive instead of recomputing.

```bash
cd /content/drive/MyDrive/voice-biometrics-detection
python -m vbd.bootstrap
python -m vbd.codec_cache pool amr_nb && python -m vbd.codec_cache pool gsm_fr
nohup python -m vbd.train_sv --name stage2v2 --epochs 12 --iters 400 --bs 64 \
  --exclude data/protocols/pool_exclude_dup_suspects.csv --codecs amr_nb,gsm_fr --p_codec 0.3 --p_mulaw 0.2 --resume >> logs/train_sv_stage2v2.log 2>&1 &
grep -E '"event": "(epoch|saved_best|done|resumed)"' checkpoints/sv/stage2v2/log.jsonl
```
