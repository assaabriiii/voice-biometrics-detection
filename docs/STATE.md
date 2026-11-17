# Where we left off (read this first in a new session)

Last updated: 2026-11-17 03:55 UTC. Nothing is running. Full history: `PROGRESS.md`. Plan and rules: `PROJECT_FLOW.md`. Runtime rebuild: `RESUME.md`.

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

## Countermeasure (S6a): finished

| Model | Checkpoint | Dev EER | ASVspoof 2019 LA eval EER | Genuine Persian speech rejected (clean / tel / gsm_fr) |
|---|---|---|---|---|
| LFCC-LCNN (primary candidate) | `checkpoints/cm/lcnn_asv19la/best.pt` | 0.58 | 6.47 | 6.0 / 6.2 / 22.8 |
| RawNet2 | `checkpoints/cm/rawnet2_asv19la/best.pt` | 1.38 | 7.54 | 13.0 / 13.0 / 17.0 |

- Both fail on voice-conversion attacks A17 and A18 of the evaluation split (LCNN EER 10.3 and 26.3); the other 11 attacks are below 5% for LCNN.
- Both reject too much genuine Persian speech at the ASVspoof threshold (domain shift). They need Persian fine-tuning and a Persian validation threshold (S6b) before fusion.
- Re-run the scoring: `python -m vbd.eval_cm` (about 10 minutes on a T4, after `python -m vbd.bootstrap`). Tables: `results/reports/s6a_cm_*.csv`.

## To continue
```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
cd /content/drive/MyDrive/voice-biometrics-detection
python -m vbd.bootstrap        # ~9 min; rebuilds /content/work (Common Voice, RIRs, MUSAN, pool cache)
```
Next job is S5 (no code written yet): test candidate Persian voice-cloning TTS and voice-conversion models on a few FA-EVAL-120 speakers, then write `vbd/spoof_gen.py` to generate the full set into `data/spoof/` on Drive.

## Remaining work
1. S6a: done (see above). ASVspoof 2019 PA source still missing (official host returned 403).
2. S5: generate 3,600 Persian spoofs for the 120 speakers — simulated replay (CPU), cloned TTS and voice conversion (GPU). Candidate models are listed in PROGRESS.md; none tested.
3. S6b: fine-tune both countermeasures on Persian spoofs inside the 5-fold protocol; leave-one-attack-out; check that noisy/telephone bona fide audio is not rejected.
4. S7: fuse speaker and liveness scores; integrated FAR / FRR / spoof acceptance (criteria 7, 8).
5. S8: first pass done (see below); still open: latency optimisation or an 8-core measurement, encryption of stored vectors.
6. S9: demo UI, final evaluation report, reproduction scripts. CPU runtime is enough.
7. Optional, for the FRR target: more Persian training speakers (lower the 8-clip pool minimum), a stronger pretrained backbone, or multi-utterance test input.

## Known open issues
- Criterion 2 (FRR <= 10%) missed by about 0.2 points.
- Text-dependent mode not measurable; accent stratification not possible; female share 27% (proposal: 40%).
- Duplicate-identity screen is a cosine > 0.5 heuristic on embeddings.
- `vbd/bootstrap.py` does not rebuild the `adapt` cache or its codec copies (only needed to re-run stage 3; commands are in PROGRESS.md / `logs/chain_s3_cm.sh`).

## Added 2026-11-17 01:25 UTC
- S8 first pass done on CPU: `vbd/system.py` (enrol / verify / embedding-only SQLite store). Latency p95 2.30 s on 2 vCPUs (target <= 2 s on 8 cores: unverified on the specified hardware; ECAPA embedding is 94% of the time, no optimisation tried). Storage inspection passed. Still open in S8: latency optimisation or an 8-core measurement, encryption of stored vectors.
- The git repository (github.com/assaabriiii/voice-biometrics-detection) now mirrors code, docs, protocol files, small result tables and training logs; `CLAUDE.md` in the repo root is the handoff guide for any Claude model. Sync method: a Colab cell zips those files and calls `google.colab.files.download`, the zip lands in `~/Downloads` on the Mac, then copy into the repo, commit, push. Do this at the end of every session.

## Added 2026-11-17 03:45 UTC
- S6a finished on the second Google account's GPU (shared-folder shortcut; see `CLAUDE.md` in the git repo for the two-account setup). Details and per-attack table: PROGRESS.md entry of 2026-11-17 03:40 UTC.
- New code: `vbd/eval_cm.py`. New launcher: `logs/chain_cm.sh`.
- S5a done: 1,200 simulated replay clips (`python -m vbd.spoof_replay`; Drive `data/spoof/replay.tar`, index `data/protocols/fa_spoof_replay.csv`). To use them on a fresh runtime: `mkdir -p /content/work/spoof && tar -xf data/spoof/replay.tar -C /content/work/spoof`. Before Persian training, 85% of replays pass the speaker engine and the ASVspoof-trained detectors are near chance on them (LCNN EER 39%); re-run with `python -m vbd.eval_spoof replay`.
- S5b (cloned TTS) and S5c (voice conversion) not started. `vbd/spoof_probe.py` is written but was never run: launching it was blocked by the Claude Code permission system on 2026-11-17 and needs the user's approval. Run with `python -m vbd.spoof_probe 6` (GPU); output `results/reports/s5_probe_scores.csv` and sample audio in `data/spoof/probe/`.
- Not yet in git (Drive only, the browser did not deliver the sync archive): `vbd/spoof_replay.py`, `vbd/eval_spoof.py`, `vbd/spoof_probe.py`, `data/protocols/fa_spoof_replay.csv`, `results/reports/s5_replay_zeroshot.csv`, `logs/spoof_replay.log`, `logs/eval_spoof_replay.log`. Include them in the next sync.
