# Where we left off (read this first in a new session)

Last updated: 2026-11-17 06:20 UTC. Nothing is running. Full history: `PROGRESS.md`. Plan and rules: `PROJECT_FLOW.md`. Runtime rebuild: `RESUME.md`.

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

## Countermeasure base models (S6a): finished

| Model | Checkpoint | Dev EER | ASVspoof 2019 LA eval EER | Genuine Persian speech rejected (clean / tel / gsm_fr) |
|---|---|---|---|---|
| LFCC-LCNN (primary candidate) | `checkpoints/cm/lcnn_asv19la/best.pt` | 0.58 | 6.47 | 6.0 / 6.2 / 22.8 |
| RawNet2 | `checkpoints/cm/rawnet2_asv19la/best.pt` | 1.38 | 7.54 | 13.0 / 13.0 / 17.0 |

- Both fail on voice-conversion attacks A17 and A18 of the evaluation split (LCNN EER 10.3 and 26.3).
- Both reject too much genuine Persian speech at the ASVspoof threshold (domain shift). They need Persian fine-tuning and a threshold set on Persian validation folds (S6b) before fusion.
- Re-run: `python -m vbd.eval_cm` (about 10 minutes on a T4). Tables: `results/reports/s6a_cm_*.csv`.

## Persian spoof set (S5): one third done

| Part | Status | Where |
|---|---|---|
| Replay, 1,200 clips (simulated, 3 device profiles) | done | Drive `data/spoof/replay.tar`; index `data/protocols/fa_spoof_replay.csv`; `python -m vbd.spoof_replay` |
| Voice conversion, 1,200 clips | not generated; method chosen and probed | `vbd/spoof_probe.py` |
| Cloned TTS, 1,200 clips | not generated; method chosen and probed | `vbd/spoof_probe.py` |

- Replay before Persian training: 85% pass the speaker engine; ASVspoof-trained detectors near chance (LCNN EER 39%). `python -m vbd.eval_spoof replay`, table `results/reports/s5_replay_zeroshot.csv`.
- Probe on 6 speakers (`results/reports/s5_probe_scores.csv`, audio in Drive `data/spoof/probe/`): kNN-VC conversions passed the speaker engine 12 of 12 times, both from human speech and from MMS Persian TTS; LCNN let through 9 of 12 and 5 of 12.
- The user approved running the cloning probe on 2026-11-17. Bulk generation is the same activity at larger scale; if the permission system blocks it, ask the user.

## Next job: S5b / S5c bulk generation (needs writing; GPU strongly preferred)

Write `vbd/spoof_gen.py` modelled on `vbd/spoof_probe.py` and `vbd/spoof_replay.py`:
1. For each of the 120 speakers build the kNN-VC matching set once from about 12 `adapt`-role clips of that speaker (never enrolment or test clips).
2. `vc_knn`: 10 clips per speaker. Source: clips of other speakers. Choose sources from the same fold as the victim or from the non-evaluation pool, so that a fold's spoofs contain no speech of other folds' speakers.
3. `tts_mms_knn`: 10 clips per speaker. `facebook/mms-tts-fas` reading Common Voice sentences (25-70 characters), then kNN-VC to the victim.
4. Write FLAC to `/content/work/spoof/<set>/`, tar to Drive `data/spoof/<set>.tar`, index to `data/protocols/fa_spoof_<set>.csv` with the same columns as the replay index (spk, fold, attack, system, file, src_rel, sec). Make it resumable (skip speakers already done) and log per speaker.
5. Score with `python -m vbd.eval_spoof <set>` (it reads `data/protocols/fa_spoof_<set>.csv` and `/content/work/spoof/<set>/`).
6. For the unseen-attack test add a second system of each kind on a smaller scale (for example 2 clips per speaker): candidates are a piper `fa_IR` voice or XTTS/F5 Persian for TTS, and FreeVC, OpenVoice v2 or Seed-VC for conversion. None is tested; probe on a few speakers first.

Time: on CPU the probe needed about 20 s per converted clip (2,400 clips would take about 13 hours), so use a T4. kNN-VC downloads 1.2 GB of weights from GitHub on every fresh runtime (about 1 minute).

## To start a session
```python
from google.colab import drive; drive.mount('/content/drive')
```
```bash
cd /content/drive/MyDrive/voice-biometrics-detection
python -m vbd.bootstrap        # ~9 min full; python -c "from vbd import bootstrap; bootstrap.main(with_noise=False, with_pool_cache=False)" takes 1 min and is enough for spoof generation and scoring
mkdir -p /content/work/spoof && tar -xf data/spoof/replay.tar -C /content/work/spoof     # only if the replay clips are needed
```

## Remaining work, in order
1. S5b/S5c: bulk voice-conversion and cloned-TTS generation (above), plus small second systems for the unseen-attack test.
2. S6b: fine-tune LCNN (and RawNet2 for comparison) on Persian spoofs inside the 5-fold protocol: train on the 3 adaptation folds (spoofs + bona fide with noise and codecs, GSM-FR included), select and set the threshold on the validation fold, test on the test fold. Report CM EER on 3,600 spoofs vs 1,800 bona fide (criterion 6), leave-one-attack-out, and bona fide rejection per condition. The spoof clips should also be passed through the telephone conditions.
3. S7: fuse speaker and liveness scores; integrated FAR / FRR / spoof acceptance (criteria 7, 8); replace the placeholder thresholds in `vbd/system.py`.
4. S8 leftovers: latency optimisation or an 8-core measurement (criterion 9); encryption of stored vectors.
5. S9: demo UI (Streamlit or FastAPI), DET curves, final evaluation report, reproduction scripts; the gaps listed in `REQUIREMENTS.md` (threat model, architecture document, leave-one-speaker-out analysis, VAD).
6. Optional, for the FRR target: more Persian training speakers (lower the 8-clip pool minimum), a stronger pretrained backbone, or multi-utterance test input.

## Other finished items
- S8 first pass: `vbd/system.py` (enrol / verify / embedding-only SQLite store). Latency p95 2.30 s on 2 vCPUs (target <= 2 s on 8 cores: unverified on the specified hardware; ECAPA embedding is 94% of the time). Storage inspection passed: one file `templates.sqlite`, 768-byte vectors only, no audio. `results/reports/s8_latency_cpu2core.csv`.
- Git mirror: github.com/assaabriiii/voice-biometrics-detection, in sync with Drive as of this update (code, docs, protocol files, result tables, logs). `CLAUDE.md` in the repo root is the handoff guide.

## Known open issues
- Criterion 2 (FRR <= 10%) missed by about 0.2 points.
- Text-dependent mode not measurable; accent stratification not possible; female share 27% (proposal: 40%).
- Duplicate-identity screen is a cosine > 0.5 heuristic on embeddings.
- Replay is a software model, not a physical replay; ASVspoof 2019 PA not obtained.
- `vbd/bootstrap.py` does not rebuild the `adapt` cache or its codec copies (only needed to re-run stage 3; commands in `logs/chain_s3_cm.sh`).
- A failed 690 MB GPU allocation warning appeared once in `logs/eval_spoof_replay.log`; the run finished. Not investigated.
