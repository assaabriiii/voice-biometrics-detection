# Proposal requirements and status tracker

This file states what the funded proposal commits to and tracks where each item stands. The final project report will be written from this file, `PROGRESS.md` and the result tables. **Update the Status and Evidence columns whenever a result changes.**

Source: the phase-1 proposal (feasibility + MVP, 6 months, 700 person-hours), Persian title «طراحی و پیاده‌سازی سامانه تصدیق هویت بیومتریک صوتی و تشخیص جعل صدا مبتنی بر یادگیری عمیق در بانکداری دیجیتال». Last updated: 2026-11-17.

Status words used below: **met**, **missed**, **partial**, **not started**, **not possible** (cannot be done under the project rules), **out of scope** (not part of the code work).

## 1. Goal and scope

- Show whether a Persian speaker-verification engine with a liveness (anti-spoofing) module can reach an acceptable first accuracy under telephone-codec degradation and ambient noise.
- Build a minimum demonstrable MVP: voice enrolment, verification, and spoof detection, with a simple demo interface.
- Scope is a limited lab scenario (telephone-banking call, GSM/VoIP codec simulation, a limited speaker set). A commercial product or deployment in a bank network is not in scope.
- Later phases (for the report's outlook section): phase 2 engineering prototype and operator pilot, target FAR 0.1%; phase 3 product, target FAR < 0.01% with FRR <= 3%.

## 2. System specification

| # | Requirement | Status | Evidence / note |
|---|---|---|---|
| R1 | Input: an audio file or stream of the user's speech | partial | Files supported (`vbd/system.py`); streaming not implemented |
| R2 | Two modes: text-independent and text-dependent | partial | Text-independent only. Text-dependent is **not possible** with public Persian data (no repeated passphrases in Common Voice) |
| R3 | Output per request: accept/reject, speaker-similarity score, liveness score, user id, timestamp | met | `VoiceAuth.verify` returns all five plus a reason and processing time |
| R4 | Functions: enrolment and voiceprint creation, signal pre-processing, feature and embedding extraction, speaker matching, liveness detection | partial | Enrolment, embedding, matching and liveness are in `vbd/system.py`; no VAD / silence removal / noise reduction step yet |
| R5 | Templates stored only as one-way, irreversible vectors; no raw audio kept | met | SQLite store with a 768-byte vector per user; inspection logged 2026-11-17 |
| R6 | Sampling rate at least 8 kHz (telephone), preferably 16 kHz | met | 16 kHz pipeline; telephone conditions pass through real 8 kHz codecs |
| R7 | Processing time per verification request under 2 s | missed on available hardware | p95 2.30 s on 2 vCPUs; the criterion specifies 8 cores (see criterion 9) |
| R8 | Architecture ready to be exposed as a service to IVR / mobile-bank / call-centre systems in later phases | partial | Service class exists; no HTTP API yet |
| R9 | Cryptography and security handling for sensitive banking data in the design | not started | Stored vectors are not encrypted at rest |

## 3. Data requirements

The proposal's data strategy is "pretrain on large public data, then fine-tune and evaluate on dedicated Persian data". The project rule is that nothing is recorded by the team, so every recorded element is replaced. The full substitution table with caveats is in `PROJECT_FLOW.md` section 3.

| # | Proposal item | Status | What was done |
|---|---|---|---|
| D1 | VoxCeleb1+2 pretrained ECAPA-TDNN weights (> 7,000 speakers), not retrained | met | `speechbrain/spkrec-ecapa-voxceleb` |
| D2 | ASVspoof 2019 LA and PA for base CM training | partial | LA downloaded and in use; PA not obtained (official host returned HTTP 403) |
| D3 | Common Voice Persian subset, >= 200 speakers, ~25 h, for language fine-tuning | met | Pool of 826 speakers, 26.9 h; 734 used for training after duplicate screening |
| D4 | MUSAN and RIR for augmentation and controlled noise conditions | met | Both in use; every 5th file reserved for evaluation only |
| D5 | 120 recorded Persian speakers, 6 accent groups x 20, >= 40% each gender, 3 sessions >= 1 week apart, ~13 h, ~8,700 utterances | not possible | Replaced by FA-EVAL-120 from Common Voice: 120 speakers, 67 M / 32 F / 21 unlabelled, no accent labels, session proxy from clip IDs |
| D6 | 3,600 Persian spoof samples of the same 120 speakers (1,200 replay with 3 playback devices, 1,200 TTS, 1,200 VC) | not started | Planned: simulated replay, open-source cloned TTS and VC |
| D7 | Per-file metadata (speaker, gender, age group, accent, session, place, equipment, noise level, SNR) | partial | Speaker, gender, age where Common Voice provides them; condition and SNR for simulated conditions |
| D8 | Written consent, labelling guideline | out of scope | No recording takes place |
| D9 | DeepMine added to fine-tuning if a research licence is obtained | not started | No licence; would also enable text-dependent evaluation |

## 4. Method requirements (the proposal's seven steps)

| Step | Requirement | Status | Note |
|---|---|---|---|
| 1 | Requirements analysis, success metrics, threat model, initial architecture | partial | Metrics fixed in this file; threat-model and architecture documents not written |
| 2 | Team training; public data prepared; Persian data and spoofs collected; 5-fold speaker-disjoint split | partial | Public data and the 5-fold protocol done; spoofs not started; training course out of scope |
| 3 | Pre-processing (resampling, VAD, silence removal, noise reduction, amplitude normalisation, noise/codec augmentation) and features (MFCC, log-mel) | partial | Resampling, normalisation, augmentation done; features are the models' own (fbank for ECAPA, LFCC for LCNN, raw waveform for RawNet2); no VAD or noise reduction |
| 4 | Baseline (i-vector or x-vector), then ECAPA-TDNN for SV and LCNN / RawNet2 for liveness; compare accuracy and compute cost; choose the final model on accuracy, latency, codec/noise robustness and simplicity | partial | x-vector baseline and ECAPA done and compared; LCNN partly trained; RawNet2 not started; CM comparison and final choice pending |
| 5 | Fuse speaker and liveness scores; thresholds set only on validation data from the DET curve, applied unchanged to test; operating points at FAR 0.8% (for the <= 1% commitment) and FAR 0.1% | partial | SV operating points done this way; fusion not started |
| 6 | Evaluation by speaker-disjoint 5-fold cross-validation; enrol on session 1, test on sessions 2 and 3; report FAR, FRR, EER, DET curves, spoof acceptance and latency per condition (quiet, natural noise, controlled noise at SNR 10 dB, GSM/AMR-NB and VoIP codecs, real call) with 95% bootstrap CIs over speakers and mean +/- std over the 5 folds | partial | Done for SV. Missing: DET curve figures, spoof metrics |
| 6b | Supplementary leave-one-speaker-out for the light final stage (score normalisation and thresholding), compared with the 5-fold result | not started | |
| 6c | Leave-one-attack-out: train the CM on two attack types, test on the third | not started | |
| 7 | Demo interface (Streamlit or FastAPI) showing enrolment, verification and the spoof result; documentation | not started | |

Four-stage training plan in the proposal and its status: (1) pretrained SV weights + CM base training on ASVspoof — SV done, CM in progress; (2) Persian language fine-tuning with AAM-Softmax — done (`stage2v2`); (3) scenario/channel adaptation on the 72 adaptation speakers per rotation with a low learning rate on the last layers — done, near-null gain; (4) threshold setting, early stopping and score normalisation on the validation fold — thresholds done; AS-norm tested and gave no gain.

## 5. Committed acceptance criteria

All SV numbers are from FA-EVAL-120 (1,800 genuine and 41,400 impostor trials pooled over 5 folds), model `stage2v2`, text-independent. Stage 3 (per-fold adaptation) gives FRR 10.17%.

| # | Metric | Condition | Committed | Stretch | Current | Status | Evidence |
|---|---|---|---|---|---|---|---|
| 1 | FAR | pooled test folds, threshold fixed on validation at FAR 0.8% | <= 1% | <= 0.1% | 0.82% (CI 0.66-0.98) | met (text-independent proxy) | `results/reports/s4_stage2v2_ecapa_operating_points.csv` |
| 2 | FRR | same operating point | <= 10% | <= 5%; <= 25% at FAR 0.1% | 10.22% (CI 7.44-13.56); 23.06% at FAR 0.13% | **missed by 0.22** | same file |
| 3 | EER, text-dependent | clean test vs clean enrolment | <= 4% | <= 2.5% | 2.50% (CI 1.66-3.35), text-independent | met as proxy; true text-dependent not possible | `s4_stage2v2_ecapa_summary.csv` |
| 4a | EER, GSM codec | GSM-FR | <= 7% | <= 4% | 5.83% (CI 4.33-7.19) | met, CI crosses 7 | same |
| 4b | EER, AMR-NB codec | | <= 7% | <= 4% | 4.83% (CI 3.68-6.31) | met | same |
| 4c | EER, VoIP codec | Opus 12 kbps (G.711: 4.48%) | <= 7% | <= 4% | 4.50% (CI 3.52-6.17) | met | same |
| 4d | EER, controlled noise | MUSAN at SNR 10 dB | <= 7% | <= 4% | 3.50% (CI 2.66-4.34) | met | same |
| 4e | EER, real phone call | simulated: noise 15-30 dB + AMR-NB | <= 7% | <= 4% | 5.67% (CI 4.33-7.67) | met on a simulated call, CI crosses 7 | same |
| 5 | EER, text-independent | test utterance >= 5 s, clean | <= 6% | <= 3% | 2.00% (CI 1.25-2.83) | met | same |
| 6 | CM EER | 3,600 Persian spoofs vs 1,800 bona fide; unseen attacks reported separately | <= 10% | <= 5% | — | not started | — |
| 7 | Spoof acceptance, integrated system | same spoofs, final thresholds | <= 10% | <= 5% | — | not started | — |
| 8 | FRR, integrated system (SV + liveness) | genuine test trials | <= 12% | <= 7% | — | not started | — |
| 9 | Response time | p95 of 200 requests, audio <= 5 s, 8-core CPU, no GPU | <= 2 s | <= 1 s | p95 2.30 s on **2 vCPUs** | missed on available hardware; unverified on 8 cores | `s8_latency_cpu2core.csv` |
| 10 | Secure template storage | inspect storage after enrolment | embeddings only, no raw audio | — | one SQLite file, vectors only | met | `PROGRESS.md`, 2026-11-17 01:15 entry |

The MVP is accepted only if all committed criteria and all live-demo conditions hold. The stretch column is reported, not committed.

## 6. Live demo acceptance scenario

Run with the demo interface in front of a representative; record the outcome in minutes.

| Step | Requirement | Pass rule | Status |
|---|---|---|---|
| 1 | Five new volunteers (absent from training and test data) each enrol with three repetitions of the passphrase (~20 s of speech) | — | not started |
| 2 | Each volunteer makes five genuine attempts (three by microphone, two by phone call or coded channel) | at least 20 of 25 accepted | not started |
| 3 | Each volunteer tries the other four identities with their own voice | at most 1 false accept in 20 | not started |
| 4 | Five replays through a phone loudspeaker, five voice-conversion samples and five TTS samples targeting the volunteers | at least 12 of 15 rejected | not started |
| 5 | Response time shown for every request; then the store is inspected | >= 95% under 2 s; only vectors stored | not started |

Under the no-recording rule the demo interface is built, and the scripted run uses held-out public speakers and uploaded files. A session with live volunteers is the user's decision.

## 7. Deliverables named in the proposal

| Deliverable | Status | Where |
|---|---|---|
| Feasibility report | not started | to be written from `PROGRESS.md` and this file |
| Requirements document | partial | this file |
| Threat model | not started | |
| System architecture document | not started | |
| Initial dataset and labelling guideline | partial | protocol files in `data/protocols/`; spoof set pending |
| Trained models | partial | SV done (Drive `checkpoints/sv/stage2v2`); CM in progress |
| Software code and MVP | partial | `vbd/` |
| Evaluation report with trial lists, score files and reproduction scripts | partial | tables in `results/reports/`; trial score files on Drive `results/scores/` |
| Demo interface | not started | |
| Acceptance-test minutes | not started | |
| Plan for the next phase | not started | |

## 8. Risks from the proposal that the report must address

| Risk | Mitigation named in the proposal | What happened so far |
|---|---|---|
| Too little or too narrow Persian data | public datasets, augmentation | Public data only; no accent labels; gender imbalance |
| Accuracy loss from noise, codecs, voice change | noise/codec augmentation, multi-session enrolment, gradual template update | Augmentation closed most of the codec gap (AMR-NB EER 10.66 -> 4.83); template update not implemented |
| Model too heavy for the 2 s limit | light architecture, quantisation and pruning | p95 2.30 s on 2 vCPUs; no optimisation tried |
| Overfitting to few speakers | speaker-disjoint 5-fold CV, pretrained models, early stopping | Applied; duplicate identities in Common Voice found and screened |
| CM fails on unseen attacks | wide range of TTS/VC systems, separate unseen-attack evaluation | Pending |
| Channel mismatch lab vs telephone | codec simulation, score normalisation, real-call test data | Real-codec augmentation helped; AS-norm gave no gain; no real calls |
| FAR 0.1% not reachable or not provable | commit to FAR <= 1% with margin, report CIs, defer 0.1% to phase 2 | FAR 0.82% met; at the 0.1% target the measured FAR is 0.13% with FRR 23% |
| Limited compute | rented cloud GPU, pretrained models, mixed precision | Free Colab T4, about 3.5 GPU-hours per account per day |
| Leak or misuse of voice data and templates | consent, anonymisation, encryption, delete raw audio | Public CC0 data; embeddings-only store; no encryption yet |

## 9. Suggested outline for the final report

1. Problem, scope and phase-1 goal.
2. Requirements and acceptance criteria (sections 2 and 5 here).
3. Data: what the proposal planned, what was used, and the consequences (section 3 here and `PROJECT_FLOW.md` section 3).
4. Evaluation protocol (FA-EVAL-120, folds, conditions, thresholds, statistics).
5. Speaker verification: baselines, fine-tuning stages, results per condition, operating points, negative results (AS-norm, multi-condition enrolment, channel thresholds, stage 3), the duplicate-identity finding.
6. Countermeasure: models, ASVspoof results, Persian spoof set, unseen-attack results.
7. Integrated system: fusion, criteria 7-8, latency, storage.
8. Criteria scoreboard with confidence intervals; what is met, missed, unverified.
9. Limits and threats to validity.
10. Plan for phase 2.
