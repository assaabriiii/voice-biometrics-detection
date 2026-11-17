# Persian voice biometrics + spoof detection MVP — handoff guide

This file is the entry point for any Claude model continuing this project. Read it fully, then `docs/STATE.md`, before doing anything.

## What the project is

A feasibility MVP from a funded proposal (Persian title: «طراحی و پیاده‌سازی سامانه تصدیق هویت بیومتریک صوتی و تشخیص جعل صدا مبتنی بر یادگیری عمیق در بانکداری دیجیتال»): verify a Persian speaker by voice over a telephone-banking-like channel and reject spoofed audio (replay, TTS, voice conversion). Two engines fused into one decision:

1. Speaker verification (SV): ECAPA-TDNN pretrained on VoxCeleb, fine-tuned on Persian Common Voice.
2. Liveness / countermeasure (CM): LCNN and RawNet2 trained on ASVspoof 2019, to be fine-tuned on Persian spoofs.

The plan, the 10 acceptance criteria and every deviation from the proposal are in `docs/PROJECT_FLOW.md`.

## Hard rules from the user (do not break)

1. **Never run project code on the user's Mac.** All execution happens on Google Colab through the Colab MCP (`mcp__colab-mcp__*`). Local actions are limited to editing files in this repo and git.
2. **Everything lives on Google Drive** at `MyDrive/voice-biometrics-detection/` (code, data, checkpoints, results, docs). Drive is not mounted on the Mac; reach it by mounting it inside Colab.
3. **This git repo mirrors the Drive copy of code, docs and small result tables.** After any change to `vbd/` or `docs/` on Drive, sync it here, commit and push. Never commit models, audio, caches or large files (see `.gitignore`).
4. **Keep the progress log detailed**: append to `docs/PROGRESS.md` after every meaningful step (what ran, numbers, failures, next). Update `docs/STATE.md` at the end of every session.
5. **No voice recording by the user.** Public datasets and software simulation only. Each substitution for the proposal's recorded data is listed in `docs/PROJECT_FLOW.md` section 3.
6. **Checkpoint every epoch to Drive** so a Colab disconnect loses at most one epoch.
7. **No leakage**: test-fold speakers never enter training, adaptation, cohorts or threshold setting. Thresholds are fixed on validation folds only.
8. **Report honestly**: every number carries its condition label and confidence interval; a missed criterion is reported as missed; negative results are logged.

## Where things are

| Thing | Location |
|---|---|
| Current state and exact resume commands | `docs/STATE.md` (read first) |
| Full history with all result tables | `docs/PROGRESS.md` |
| Plan, criteria, substitutions, rules | `docs/PROJECT_FLOW.md` |
| Rebuilding a fresh Colab runtime | `docs/RESUME.md` |
| Code package | `vbd/` |
| Small result tables | `results/reports/` |
| Protocol (speakers, folds, exclusions) | `data/protocols/` |
| Models, embeddings, raw data, trial score files | Drive only |

`vbd/` modules: `config` (paths, constants), `data` (manifest, FA-EVAL-120 selection, folds, pool), `augment` (noise, reverb, codecs), `cache` / `codec_cache` (decoded-audio memmaps), `metrics` (EER, FAR/FRR, bootstrap), `sv` (embeddings, scoring, AS-norm, reports), `conditions` (test conditions + evaluation), `train_sv` (stage 2), `train_sv3` (stage 3 per-fold adaptation), `cm` (LCNN, RawNet2), `train_cm` (CM on ASVspoof), `system` (enrol/verify service + embedding-only store), `bootstrap` (rebuild Colab working dir).

## How to work on Colab

1. Ask the user to open Colab with the Colab MCP link, on a **T4 GPU** runtime for training (CPU is enough for scoring, latency, UI).
2. `mcp__colab-mcp__open_colab_browser_connection`, then load the cell tools with ToolSearch.
3. First cell: mount Drive (the user may have to approve a popup in the browser), then `python -m vbd.bootstrap` (about 9 minutes; rebuilds `/content/work` from archives on Drive, downloads nothing).
4. Write code to Drive with `%%writefile` cells; run long jobs with `nohup ... &` writing logs to `logs/` on Drive; monitor with short cells that only read logs.

Lessons already paid for:

- The runtime has **12 GB RAM and 2 vCPUs**. Running heavy cells in the notebook kernel while a training job runs got both OOM-killed. During training, the kernel only reads logs.
- Free-tier GPU time ran out after about 3.5 hours in one day. Plan GPU work in sessions; put CPU-only work in the gaps.
- `run_code_cell` often returns before a long cell finishes. Re-read the cell with `get_cells` or use a small status cell; do not assume it failed.
- Opening the MCP link in a new tab creates a **second runtime** (often CPU-only) and the MCP attaches to the newest tab. A job started earlier keeps running on the old runtime; monitor it through Drive logs and never start a second writer for the same run.
- The notebook is an unsaved `empty.ipynb`; reloading its tab can kill the runtime. All durable code must be in `vbd/`, not in cells.
- `pgrep -f train_sv` matches its own shell; use `pgrep -f "[v]bd.train_sv"`.
- System ffmpeg lacks AMR-NB; a static build is stored at `tools/ffmpeg` on Drive (the user approved that download) and copied to `/content/work/ffmpeg_static` by bootstrap.
- Per-clip ffmpeg calls are slow; use `augment.codec_batch`.

## Evaluation protocol in one paragraph

FA-EVAL-120: 120 Common Voice Persian speakers, 5 speaker-disjoint folds of 24. Per speaker 5 enrolment clips and 15 test clips (5 clean, 5 noisy, 5 simulated telephone). Each test clip is scored against all 24 templates of its fold: 1,800 genuine and 41,400 impostor trials (`main15`). Rotation k: test fold k, validation fold k+1, the other three for adaptation. Thresholds come from the validation fold at FAR 0.8% (and 0.1%) and are applied unchanged to the test fold. Extra controlled conditions use the clean clips: noise at 10 dB, GSM-FR, AMR-NB, G.711, Opus, and a >= 5 s set. Everything is text-independent because public Persian data has no repeated passphrases.

## Status snapshot (2026-11-17; `docs/STATE.md` is authoritative)

- SV reported model: `checkpoints/sv/stage2v2/best.pt`. On `main15`: FAR 0.82% (target <= 1, met), **FRR 10.22% (target <= 10, missed)**; all EER targets met on point estimates. Per-fold adaptation (stage 3) gave 10.17%, a near-null result.
- CM: LCNN at 11 of 30 epochs on ASVspoof 2019 LA (dev EER 0.63%); RawNet2 not started.
- Service logic and embedding-only store: `vbd/system.py` (thresholds are placeholders until fusion is done). Storage inspection passed (one SQLite file, vectors only).
- Latency: p95 2.30 s on Colab's 2-vCPU CPU runtime (target <= 2 s on an 8-core CPU, so unverified on the specified hardware). ECAPA embedding is 94% of the time; no optimisation tried yet.
- Not started: Persian spoof generation, CM fine-tuning, score fusion, demo UI, final report.

## Syncing Drive to this repo

There is no direct path from Drive to the Mac. The method that worked: a Colab cell zips `vbd/*.py`, `docs/*.md`, `results/reports/*.csv`, `data/protocols/*.csv`, launcher scripts and training logs, then calls `google.colab.files.download(...)`, which saves the zip to the Mac's `~/Downloads` through the browser. Unzip it into a scratch folder, copy over the repo, review `git status`, commit and push. Do this at the end of every session.

## Next steps, in order

1. Finish LCNN, train RawNet2 (`docs/STATE.md` has the command); report ASVspoof 2019 LA evaluation-split EER.
2. Generate 3,600 Persian spoofs for the 120 speakers (1,200 each: simulated replay, cloned TTS, voice conversion). Candidate open models are listed in `docs/PROGRESS.md`; none is tested. Test quality on a few samples before generating in bulk.
3. Fine-tune the CMs on Persian spoofs inside the 5-fold protocol; leave-one-attack-out; confirm noisy/telephone bona fide audio is not flagged.
4. Fuse SV and CM scores; measure integrated FAR, FRR and spoof acceptance (criteria 7, 8).
5. CPU latency (criterion 9) and embedding-only storage inspection (criterion 10).
6. Demo UI (Streamlit or FastAPI), final evaluation report, reproduction scripts.
7. Optional for the FRR miss: more Persian training speakers, a stronger pretrained backbone, or multi-utterance test input.

## Known limits to keep stating in reports

- Text-dependent criteria are measured in text-independent mode.
- Noise, telephone calls and replay are simulated, not recorded.
- No accent stratification; 27% female speakers (proposal asked for 40%).
- Duplicate identities in Common Voice are screened with a cosine > 0.5 heuristic (`data/protocols/pool_exclude_dup_suspects.csv`, `eval_dup_suspect_pairs.csv`).
- ASVspoof 2019 PA could not be downloaded (official host returned 403).
