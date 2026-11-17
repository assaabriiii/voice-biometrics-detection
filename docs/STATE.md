# Where we left off (read this first in a new session)

Last updated: 2026-11-17 01:10 UTC. Nothing is running. Full history: `PROGRESS.md`. Plan and rules: `PROJECT_FLOW.md`. Runtime rebuild: `RESUME.md`.

## Speaker verification (S0-S4): finished for now

| Model | Where | FA-EVAL-120 main15: FAR / FRR at target FAR 0.8% |
|---|---|---|
| ECAPA zero-shot | pretrained | 0.84 / 29.11 |
| stage 2 (superseded, leak) | `checkpoints/sv/stage2/best.pt` | 0.82 / 11.11 |
| **stage2v2 (reported model)** | `checkpoints/sv/stage2v2/best.pt` | 0.82 / 10.22 |
| stage 3 (per-rotation adaptation) | `checkpoints/sv/stage3/rot{0..4}/best.pt` | 0.82 / 10.17 |

- Committed targets: FAR <= 1 met; **FRR <= 10 not met** (10.17-10.22, CI 7.4-13.5); EER targets (clean <= 4, telephone/noise <= 7, >= 5 s <= 6) met on point estimates, GSM-FR and simulated-call CIs cross 7.
- Stage 3 gave almost nothing; stage2v2 is the simpler choice for the system unless a later decision says otherwise.
- All measurements are text-independent (text-dependent mode is not measurable with public data).

## Countermeasure (S6a): partly done
- LCNN on ASVspoof 2019 LA: 11 of 30 epochs done, best dev EER 0.63% (epoch 9). `checkpoints/cm/lcnn_asv19la/{best,last}.pt`. Resume continues from epoch 11.
- RawNet2: not started.

## To continue (needs a GPU runtime)
```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
cd /content/drive/MyDrive/voice-biometrics-detection
python -m vbd.bootstrap                         # ~9 min; rebuilds /content/work (Common Voice, RIRs, MUSAN, pool cache)
nohup sh -c "python -m vbd.train_cm --model lcnn --epochs 30 --bs 64 --lr 3e-4 --resume && python -m vbd.train_cm --model rawnet2 --epochs 20 --bs 32 --lr 1e-4 --resume" >> logs/chain_cm.log 2>&1 &
grep -E '"event": "(epoch|saved_best|done|resumed)"' checkpoints/cm/*/log.jsonl
```
`train_cm` rebuilds the ASVspoof caches from the parquet files on Drive by itself (a few minutes). LCNN needs about 50 more minutes; RawNet2 time is unknown (not yet run; expect a few hours).

## Remaining work
1. S6a: finish LCNN, train RawNet2; report ASVspoof 2019 LA evaluation-split EER for both. ASVspoof 2019 PA source still missing (official host returned 403).
2. S5: generate 3,600 Persian spoofs for the 120 speakers — simulated replay (CPU), cloned TTS and voice conversion (GPU). Candidate models are listed in PROGRESS.md; none tested.
3. S6b: fine-tune both countermeasures on Persian spoofs inside the 5-fold protocol; leave-one-attack-out; check that noisy/telephone bona fide audio is not rejected.
4. S7: fuse speaker and liveness scores; integrated FAR / FRR / spoof acceptance (criteria 7, 8).
5. S8: CPU latency p95 over 200 requests (criterion 9); embedding-only template store (criterion 10). CPU runtime is enough.
6. S9: demo UI, final evaluation report, reproduction scripts. CPU runtime is enough.
7. Optional, for the FRR target: more Persian training speakers (lower the 8-clip pool minimum), a stronger pretrained backbone, or multi-utterance test input.

## Known open issues
- Criterion 2 (FRR <= 10%) missed by about 0.2 points.
- Text-dependent mode not measurable; accent stratification not possible; female share 27% (proposal: 40%).
- Duplicate-identity screen is a cosine > 0.5 heuristic on embeddings.
- `vbd/bootstrap.py` does not rebuild the `adapt` cache or its codec copies (only needed to re-run stage 3; commands are in PROGRESS.md / `logs/chain_s3_cm.sh`).
