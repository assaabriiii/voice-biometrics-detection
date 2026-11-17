# Persian Voice Biometrics + Spoof Detection (MVP) — Project Flow & Rules

Source: proposal "طراحی و پیاده‌سازی سامانه تصدیق هویت بیومتریک صوتی و تشخیص جعل صدا مبتنی بر یادگیری عمیق در بانکداری دیجیتال" (phase 1, feasibility + MVP).
Written 2026-11-16. This file is the plan; `PROGRESS.md` is the running log.

## 1. What the project is

A lab-grade MVP that, for one Persian utterance arriving over a telephone-banking-like channel, returns:
`accept / reject`, a **speaker-similarity score**, a **liveness (anti-spoof) score**, the user id and a timestamp.

Two engines, fused into one decision:

1. **Speaker verification (SV)** — x-vector baseline, then ECAPA-TDNN (pretrained on VoxCeleb1+2, fine-tuned for Persian with AAM-Softmax, adapted to the telephone/noise scenario).
2. **Liveness / countermeasure (CM)** — LCNN and RawNet2 (trained on ASVspoof 2019, fine-tuned on Persian spoofs: replay, TTS, voice conversion).

Storage rule from the proposal: only the one-way embedding (voiceprint) is stored, never raw audio.

## 2. Acceptance criteria (from the proposal — these are the targets we report against)

| # | Metric | Condition | Committed | Stretch |
|---|--------|-----------|-----------|---------|
| 1 | FAR | pooled 5-fold test, threshold fixed on validation fold at FAR=0.8% | <= 1% | <= 0.1% |
| 2 | FRR | same operating point | <= 10% | <= 5% (and <= 25% at FAR=0.1%) |
| 3 | EER, text-dependent | clean test vs. clean enrolment | <= 4% | <= 2.5% |
| 4 | EER, telephone/noisy | each of: GSM/AMR-NB codec, VoIP codec, noise at SNR=10 dB, real call | <= 7% | <= 4% |
| 5 | EER, text-independent | test utterance >= 5 s, clean | <= 6% | <= 3% |
| 6 | CM EER | 3,600 spoofs vs 1,800 bona fide; leave-one-attack-out reported separately | <= 10% | <= 5% |
| 7 | Spoof acceptance rate, integrated system | same spoofs, final thresholds | <= 10% | <= 5% |
| 8 | FRR, integrated system (SV + CM) | genuine test trials | <= 12% | <= 7% |
| 9 | Response time | p95 of 200 requests, <= 5 s audio, 8-core CPU, no GPU | <= 2 s | <= 1 s |
| 10 | Secure template storage | inspect storage after enrolment | embeddings only, no raw audio | — |

All rates are reported with 95% bootstrap confidence intervals (resampling speakers) and mean +/- std over the 5 folds.

## 3. Deviation from the proposal: no self-recorded data

The proposal's core evaluation set is 120 Persian speakers recorded by the team in 3 sessions, plus 3,600 team-made spoofs.
**Project rule: nothing is recorded by us.** Every recorded element is replaced by public data or simulation, and each substitution is listed here so results are never presented as something they are not.

| Proposal element | Replacement used here | What is lost / caveat |
|---|---|---|
| 120 recorded speakers, 6 accent groups x 20, >= 40% per gender | **FA-EVAL-120**: 120 speakers selected from Common Voice 22 Persian (CC0) with >= 30 clips each | Accent labels are missing for ~92% of CV-fa speakers, so accent stratification is not possible. Gender is skewed (of 268 speakers with >= 30 clips: 147 male, 37 female, 84 unlabelled); we take as many labelled females as possible and report the actual ratio. |
| 3 sessions >= 1 week apart | Session proxy: Common Voice clip-ID gaps (IDs are assigned in upload order). Enrolment from the earliest cluster, tests from later clusters where the speaker has them | Only ~129 of the 268 candidate speakers show >= 2 clusters; true time gap is unknown. Speakers with multi-cluster data are preferred for selection. |
| Fixed passphrase, 5 repetitions (text-dependent mode) | **Not reproducible**: in Common Voice no speaker repeats a sentence (verified: 0 repeated speaker-sentence pairs) | Criteria 1-3 are evaluated in **text-independent** mode on short utterances. Text-independent is the harder task, so passing it is conservative evidence, but it is not a text-dependent measurement and is labelled as such in every report. DeepMine would fix this if a research licence is obtained. |
| Session 2 natural noise (cafeteria / office / street, 50-65 dBA) | MUSAN noise + babble + RIR reverberation added in software | Simulated, not natural noise. |
| Session 3 real mobile call (AMR-NB over the operator network) | Codec simulation with FFmpeg: band-limit 300-3400 Hz, 8 kHz, GSM-FR / AMR-NB / G.711 / Opus | No real network effects (packet loss, handset variety). Criterion 4 "real call" is reported as "simulated telephone chain". |
| Replay spoofs with 3 playback devices | Simulated replay: loudspeaker/device impulse response + room RIR + re-recording noise (same idea as ASVspoof 2019 PA, which is itself simulated) | Not physical replays. |
| TTS and VC spoofs of the same 120 speakers | Generated on Colab with open-source Persian-capable TTS / zero-shot voice cloning and voice conversion, targeting FA-EVAL-120 speakers | Model choice is verified on Colab before it is fixed (see stage S5). |
| Live demo with 5 new volunteers | Demo UI is built; the scripted acceptance run uses held-out public speakers and uploaded files | A live human session is outside this rule set. |
| Team training course, operator requirement interviews, consent forms | Out of scope for the code work | — |

## 4. Public data and models actually used (availability verified from Colab on 2026-11-16)

| Asset | Source | Size | Role |
|---|---|---|---|
| ECAPA-TDNN pretrained on VoxCeleb1+2 | `speechbrain/spkrec-ecapa-voxceleb` | 0.09 GB | Stage-1 SV model (not retrained) |
| x-vector pretrained | `speechbrain/spkrec-xvect-voxceleb` | 0.03 GB | SV baseline |
| Common Voice 22 Persian | `fsicoli/common_voice_22_0` (train/dev/test/other shards) | 2.3 GB audio; 85,644 clips, 93.6 h, 4,247 speakers | Persian fine-tuning pool + FA-EVAL-120 |
| ASVspoof 2019 LA | `Bisher/ASVspoof_2019_LA` (parquet) | 7.5 GB | CM base training (LCNN, RawNet2) |
| ASVspoof 2019 PA | Edinburgh DataShare returned HTTP 403 to a HEAD request | — | **Open item**: find a working mirror, otherwise replay training relies on simulated replay only |
| MUSAN | openslr.org/resources/17 | 11.1 GB | Noise / babble / music augmentation and SNR=10 dB test condition |
| RIRS_NOISES | openslr.org/resources/28 | 1.3 GB | Reverberation augmentation, simulated replay |
| Persian TTS voices | `rhasspy/piper-voices` (6 fa_IR voices), `facebook/mms-tts-fas` | < 1 GB | TTS spoof sources (cloning/VC candidates checked in S5) |

## 5. Evaluation protocol (FA-EVAL-120)

- 120 speakers -> 5 folds x 24, speaker-disjoint, gender-balanced as far as labels allow. All utterances of a speaker stay in one fold.
- Each rotation: 3 folds (72 spk) adapt, 1 fold (24) validation/threshold, 1 fold (24) test. Every speaker is a test speaker exactly once.
- Per speaker: **5 enrolment clips** (earliest clips) and **15 genuine test clips** (5 clean, 5 noisy, 5 telephone). Remaining clips of the 72 adaptation speakers are used for stage-3 adaptation.
- Impostor trials: every test clip against the other 23 templates of its fold -> 120 x 15 = 1,800 genuine trials and 120 x 15 x 23 = 41,400 impostor trials pooled over folds (matches the proposal's counts).
- Extra controlled conditions built from the clean test clips: MUSAN noise at exactly SNR=10 dB, GSM-FR, AMR-NB, G.711, Opus.
- Thresholds are set on the validation fold only (FAR=0.8% and FAR=0.1%) and applied unchanged to the test fold.
- Spoof set: 3,600 spoofs (1,200 replay, 1,200 TTS, 1,200 VC) of the same 120 speakers vs. 1,800 bona fide test clips; leave-one-attack-out for unseen attacks.
- Speakers used for Persian fine-tuning (stage 2) are disjoint from FA-EVAL-120.
- All selection and augmentation uses fixed seeds; trial lists and score files are saved so results can be regenerated.

## 6. Work stages

| Stage | Content | Output on Drive |
|---|---|---|
| S0 | Colab + Drive setup, dataset availability check, flow and rules | `docs/` |
| S1 | Data: download Common Voice fa, build manifests, select FA-EVAL-120, folds, trial lists | `data/manifests`, `data/protocols` |
| S2 | Channel/noise simulation library (MUSAN, RIR, codecs) and the test conditions | `vbd/augment.py`, condition audio lists |
| S3 | SV baselines: pretrained x-vector and ECAPA, zero-shot on FA-EVAL-120, all conditions | `results/scores`, first report |
| S4 | SV Persian fine-tuning (AAM-Softmax + augmentation), per-fold adaptation, score normalisation, thresholds | `checkpoints/sv` |
| S5 | Spoof data: simulated replay, TTS, VC for the 120 speakers | `data/spoof` |
| S6 | CM: LCNN and RawNet2 on ASVspoof 2019 LA, fine-tune on Persian spoofs, leave-one-attack-out | `checkpoints/cm` |
| S7 | Fusion and decision logic, integrated-system metrics (criteria 7, 8) | `results/reports` |
| S8 | Latency benchmark on CPU, embedding-only template store | `results/reports` |
| S9 | Demo UI (enrol / verify / spoof result), final evaluation report and reproduction scripts | `demo/`, `docs/` |

## 7. Rules

1. **Execution**: all code runs on Google Colab through the Colab MCP connection. Nothing is executed on the local machine.
2. **Storage**: everything lives under `MyDrive/voice-biometrics-detection/` — code, manifests, protocols, checkpoints, scores, reports, docs.
3. **Working copies**: Drive is slow with many small files. Downloaded archives are kept on Drive (`data/raw`), extracted to the Colab local disk (`/content/work`) for processing, and only derived artefacts are written back. Every notebook can rebuild `/content/work` from Drive.
4. **Progress log**: `docs/PROGRESS.md` is updated after every meaningful step with what was run, the numbers obtained, problems, and what is next. Failures and negative results are logged too.
5. **No recording**: only public datasets and software simulation. Every substitution is listed in section 3.
6. **No leakage**: test-fold speakers never appear in fine-tuning, adaptation, score-normalisation cohorts or threshold setting for that rotation. FA-EVAL-120 speakers never appear in stage-2 fine-tuning.
7. **Thresholds** are fixed on validation data and never tuned on test data.
8. **Reproducibility**: fixed seeds, saved trial lists, saved score files, config files for every run.
9. **Honest reporting**: every metric is reported with its condition label (e.g. "simulated telephone", "text-independent proxy") and a 95% confidence interval. A criterion that is not met is reported as not met.
10. **Privacy**: the enrolment store keeps embeddings only. Synthetic spoofs of public speakers are used only for countermeasure training and evaluation and stay in the private Drive folder.
11. **Code layout**: reusable code is the `vbd/` Python package on Drive; notebooks only orchestrate. Long runs checkpoint to Drive so a disconnected Colab session can resume.

## 8. Folder layout

```
voice-biometrics-detection/
  docs/         PROJECT_FLOW.md, PROGRESS.md, reports
  vbd/          Python package (data, augment, metrics, sv, cm, fusion, store)
  notebooks/    orchestration notebooks
  configs/      run configs
  data/raw      downloaded archives        data/manifests  per-clip metadata
  data/protocols  folds + trial lists      data/spoof      generated spoof audio
  features/     cached embeddings          checkpoints/    sv/, cm/
  results/      scores/, reports/, figures/
  demo/         UI app                     logs/           run logs
```
