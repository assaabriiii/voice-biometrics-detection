# Progress log — Persian voice biometrics + spoof detection MVP

Newest entries at the bottom. Plan and rules: `PROJECT_FLOW.md`.

## 2026-11-16 — S0: setup, understanding, flow

**Done**
- Read the proposal (phase 1: feasibility + MVP, 6 months / 700 person-hours, 10 acceptance criteria).
- Connected to Colab through the Colab MCP. Runtime: Tesla T4 (15 GB), 2 vCPU, 12 GB RAM, Python 3.13, torch 2.14.1+cu130, torchaudio 2.11.0, ffmpeg present, sox absent.
- Mounted Google Drive, created the project tree at `MyDrive/voice-biometrics-detection/` (about 107 GB free on Drive).
- Checked dataset/model availability from Colab without downloading:
  - OK: `speechbrain/spkrec-ecapa-voxceleb`, `speechbrain/spkrec-xvect-voxceleb`, `fsicoli/common_voice_22_0` (fa: 3.1 GB), `hezarai/common-voice-13-fa`, `Bisher/ASVspoof_2019_LA` (7.5 GB parquet), MUSAN (openslr, 11.1 GB), RIRS_NOISES (openslr, 1.3 GB), `rhasspy/piper-voices` (6 fa_IR voices), `facebook/mms-tts-fas`, `myshell-ai/OpenVoiceV2`, `Plachta/Seed-VC`, `coqui/XTTS-v2`.
  - Not OK: ASVspoof 2019 LA.zip / PA.zip on Edinburgh DataShare answered HTTP 403 to a HEAD request; `Mahdih/ShEMO` does not exist on the Hub; `mozilla-foundation/common_voice_*` repos are empty shells (2 files).
- Analysed Common Voice 22 Persian metadata (TSVs only):
  - Audio available in shards: train 29,789 clips / 194 spk / 31.6 h; dev 10,676 / 985 / 12.6 h; test 10,676 / 2,818 / 14.6 h; other 34,503 / 859 / 34.7 h. Total 85,644 clips, 93.6 h, 4,247 speakers.
  - `validated.tsv` lists 338,056 clips but only 51,141 of them have audio in this mirror (train/dev/test).
  - Speakers with >= 5 / 10 / 20 / 30 / 50 / 100 / 200 clips: 1,802 / 786 / 386 / 268 / 187 / 115 / 61. Median speaker has 4 clips.
  - Among the 268 speakers with >= 30 clips: gender 147 male, 37 female, 84 unlabelled; accent label missing for 246.
  - No speaker repeats a sentence (0 repeated speaker-sentence pairs); only 3 sentences are spoken by >= 20 speakers.
  - Session proxy (clip-ID gap > 20,000): 129 of the 268 have >= 2 clusters, 84 have >= 3.
  - Speakers overlap between `other` and train/dev/test (96 / 143 / 370), so splits must be rebuilt by speaker.

**Decisions**
- Evaluation set FA-EVAL-120 is built from Common Voice fa instead of self-recorded speakers; full substitution table in `PROJECT_FLOW.md` section 3.
- Text-dependent mode cannot be measured with this data; criteria 1-3 are reported in text-independent mode and labelled as a proxy.
- Accent stratification (6 x 20) is not possible; gender balance will be below the proposal's 40% and the real ratio is reported.

**Open items**
- ASVspoof 2019 PA source (403 on the official host).
- Which open-source voice-cloning TTS / VC handles Persian acceptably — to be tested in S5.

**Next**: S1 — download Common Voice fa audio, build manifests, select FA-EVAL-120, folds and trial lists.


## 2026-11-16 — S1: data and protocol

**Done**
- Downloaded Common Voice 22 fa shards (train/dev/test/other, 2.3 GB) to `data/raw/cv22_fa/` and extracted 85,644 mp3 clips (32 kHz) to the Colab disk. RIRS_NOISES (1.3 GB) downloaded and extracted. MUSAN (11.1 GB) download from openslr runs at about 3.7 MB/s.
- Code package on Drive: `vbd/config.py`, `metrics.py` (EER, FAR/FRR, threshold at target FAR, speaker bootstrap), `data.py` (manifest, selection, folds, pool), `augment.py` (noise at set SNR, reverb, band-limit, codecs via ffmpeg, batched codec, fast mu-law telephone approximation), `sv.py` (embeddings, trial scoring, summaries, operating point), `cache.py` (int16 memmap audio cache), `train_sv.py` (stage-2 fine-tuning).
- Built **FA-EVAL-120** (`data/protocols/fa_eval120_speakers.csv`, `fa_eval120_utts.csv`), seed 20261116:
  - Eligible clips: 3-12 s. Candidates: speakers with >= 30 eligible clips. Labelled female speakers taken first, then speakers with >= 2 session-proxy clusters.
  - 120 speakers: 67 male, 32 female, 21 unlabelled gender (female share 27%, below the proposal's 40%).
  - 5 folds x 24 speakers, each with 6-7 female / 13-14 male / 4-5 unlabelled.
  - Per speaker: 5 enrolment clips (earliest, mean 4.9 s each) and 15 test clips (mean 4.5 s), assigned 5 clean / 5 noisy / 5 telephone.
  - 86 of 120 speakers have all test clips from a later session-proxy cluster than enrolment; for the other 34 the test clips are simply later clips of the same cluster.
  - 511 of the 1,800 test clips are >= 5 s (relevant to criterion 5).
  - Remaining 29,584 clips of these speakers are marked `adapt` (min 10, median 89, max 3,446 per speaker).
- Built the fine-tuning pool (`data/manifests/fa_pool.csv`): 826 speakers disjoint from FA-EVAL-120, 24,356 clips, 26.9 h (>= 8 clips per speaker; gender 278 M / 42 F / 506 unlabelled).

**Problems**
- System ffmpeg on Colab has no AMR-NB encoder. A static ffmpeg build (johnvansickle.com, 7.0.2) with `libopencore_amrnb` was downloaded after explicit user approval and stored in `tools/ffmpeg`; `augment.codec("amr_nb")` uses it.
- Per-clip ffmpeg calls are slow on 2 vCPUs: about 6.6 min per codec for 1,800 clips. `augment.codec_batch` (one ffmpeg round-trip per 30 min of audio) was written to replace it; not yet used in a run.

## 2026-11-16 — S3a: zero-shot baselines (pretrained models, no Persian data)

Setup: enrolment = mean of 5 clip embeddings (original wideband audio); every test clip scored by cosine against all 24 templates of its fold; 1,800 genuine and 41,400 impostor trials per condition. Codec conditions: all 15 test clips per speaker passed through 300-3400 Hz band-limit + codec (enrolment stays wideband). **Text-independent**, short utterances. 95% CI = bootstrap over speakers.

| Model | Condition | EER % | 95% CI | fold mean +/- std |
|---|---|---|---|---|
| ECAPA-TDNN (VoxCeleb) | original audio | 4.22 | 2.92-5.62 | 4.18 +/- 1.25 |
| ECAPA-TDNN | G.711 A-law | 9.11 | 7.56-10.87 | 9.08 +/- 1.71 |
| ECAPA-TDNN | GSM-FR | 12.50 | 10.61-14.89 | 12.54 +/- 0.68 |
| ECAPA-TDNN | Opus 12 kbps narrowband | 9.56 | 8.08-11.47 | 9.23 +/- 1.62 |
| x-vector (VoxCeleb) | original audio | 13.83 | 11.11-16.60 | 13.69 +/- 1.48 |
| x-vector | G.711 A-law | 29.44 | 26.40-32.05 | 29.25 +/- 3.17 |
| x-vector | GSM-FR | 31.11 | 28.08-33.70 | 30.66 +/- 4.24 |
| x-vector | Opus 12 kbps narrowband | 25.95 | 23.58-28.81 | 25.95 +/- 1.11 |

Operating point (threshold from the validation fold at FAR = 0.8%, applied to the test fold, original audio, pooled over 5 rotations):
- ECAPA: FAR 0.92%, FRR 10.06%.
- x-vector: FAR 0.77%, FRR 39.33%.

**Reading**
- ECAPA without any Persian data is already near the committed text-dependent target on clean audio (EER 4.22% vs <= 4%, FRR 10.06% vs <= 10%) even though this is the harder text-independent task. It is not yet a pass.
- Telephone codecs are the gap: 9-12.5% EER against the committed <= 7%. Wideband enrolment against narrowband test is a strong mismatch; this is what stage-2/3 fine-tuning with telephone augmentation and score normalisation must fix.
- x-vector is far behind and stays as the baseline row only.
- Not yet measured: noise at SNR = 10 dB, AMR-NB, the mixed clean/noisy/telephone main condition, the >= 5 s condition.
- Not yet checked: whether some Common Voice client IDs belong to the same person (would create false "impostor" trials).

Files: `results/scores/s3_zeroshot_{ecapa,xvector}_codecs.csv.gz`, `results/reports/s3_zeroshot_*_summary.csv`, `features/s3_zeroshot_*_emb.npz`.

**Next**: stage-2 fine-tuning of ECAPA on the pool (launched in the background: audio cache, wait for MUSAN, then 12 epochs x 400 iterations, batch 64, log in `logs/train_sv_stage2.log` and `checkpoints/sv/stage2/log.jsonl`).


## 2026-11-16 — S3b / S4: conditions, duplicate-ID check, stage-2 fine-tuning started

**Done**
- MUSAN downloaded (11.1 GB, on Drive) and extracted at 13:50 UTC. Every 5th MUSAN/RIR file (sorted order) is reserved for evaluation conditions; training augmentation never uses those files.
- AMR-NB condition generated with the batched codec path: 213 s for 1,800 clips (band-limit + codec), against about 400 s per codec with per-clip calls. Batched vs per-clip GSM-FR waveforms correlate 0.95 on average (min 0.86); the difference comes from codec frame alignment.
- `vbd/conditions.py` written: builds all test conditions deterministically (main15 = 5 clean + 5 noisy at SNR 5-20 dB + 5 telephone with AMR-NB; clean; snr10; gsm_fr; amr_nb; g711a; opus_nb; ti5s) and evaluates an encoder on all of them. The condition audio was built and cached at `/content/work/eval_conditions.pkl` (Colab disk only).
- Duplicate-identity check on FA-EVAL-120 with zero-shot ECAPA templates: 7 speaker-ID pairs have template cosine > 0.6 and 48 pairs > 0.5, while the median genuine trial score is 0.61. Top pairs: S00353/S02732 (0.788), S01500/S01155 (0.763), S03824/S00070 (0.693), S01688/S00310 (0.671), S00485/S03253 (0.662). These are probably the same person under two Common Voice client IDs. All ten top pairs fall in different folds, so they do not create false impostor trials (trials are within-fold), but they can leak a test speaker into another fold's adaptation data. **Not yet handled.** The same check between FA-EVAL-120 and the fine-tuning pool has not been run.
- Stage-2 fine-tuning launched 13:51 UTC (`checkpoints/sv/stage2/`): 766 training speakers, 60 held-out pool speakers as dev set, AAM-Softmax (m=0.2, s=30), batch 64, 3 s crops, 400 iterations per epoch, 12 epochs, backbone lr 3e-5, head lr 2e-3, cosine schedule, epoch 0 trains the head only. Augmentation: reverb 25%, MUSAN noise/music/babble 60%, mu-law 8 kHz telephone approximation 35%.

**Stage-2 dev results so far** (60 held-out pool speakers, 3 enrolment clips, EER %)

| Point | original audio | telephone approximation |
|---|---|---|
| pretrained (no Persian data) | 3.08 | 7.56 |
| epoch 0 (head only, backbone unchanged) | 3.08 | 7.56 |
| epoch 1 | 2.13 | 3.88 |

Speed: about 0.5 s/iteration with the backbone frozen, about 0.75 s/iteration when training the full model (5.2 min per epoch including dev evaluation).

**Problems**
- 14:03 UTC: the Linux OOM killer killed the notebook kernel (8.8 GB resident) and the training process stopped with it. Cause: the full baseline cell was run in the notebook kernel while training was running; the kernel still held the earlier codec variants in memory and the machine has 12 GB RAM. The baseline evaluation for the new conditions did not finish, so **noise, AMR-NB, main15 and ti5s baseline numbers are still missing**.
- Training resumed at 14:10 UTC from the epoch-1 checkpoint with `--resume` (epoch 2 restarted from its beginning).
- One GPU out-of-memory warning during the first dev evaluation (batch of long clips); PyTorch retried and continued.
- The earlier monitor cell reported "RUNNING" while training was dead, because `pgrep -f train_sv` matched its own shell command. Monitoring now uses the pattern `[v]bd.train_sv`.

**Rule added**: while a training job runs, the notebook kernel only runs light monitoring. Heavy evaluation waits until training ends.

**Next**: let stage 2 finish (expected about 15:05 UTC), then run the full-condition evaluation for zero-shot ECAPA, x-vector and the fine-tuned model.


## 2026-11-16 14:52 UTC — S4 stage-2 fine-tuning: 11 of 12 epochs done, session paused (usage limit)

Dev EER % on 60 held-out pool speakers (original audio / telephone approximation):

| Epoch | orig | tel | Epoch | orig | tel |
|---|---|---|---|---|---|
| pretrained | 3.08 | 7.56 | 6 | 1.90 | 2.52 |
| 1 | 2.13 | 3.88 | 7 | 1.74 | 2.67 |
| 2 | 2.13 | 3.14 | 8 | 1.74 | 2.52 |
| 3 | 1.98 | 3.12 | 9 | 1.94 | 2.67 |
| 4 | 1.94 | 2.67 | 10 | 1.76 | 2.48 (best, saved as `best.pt`) |
| 5 | 1.94 | 2.75 | 11 | running at 14:52 (iteration 200 of 400) |

- The curve is flat over the last four completed epochs (average 2.12-2.31).
- `checkpoints/sv/stage2/last.pt` holds epoch 10; epoch 11 writes it again when it ends (about 14:54 UTC) if the runtime stays up.
- Added `vbd/bootstrap.py` (rebuilds `/content/work` from Drive archives; **not yet tested on a fresh runtime**) and `docs/RESUME.md`. Checkpoint writes made atomic for future runs.

**Not done yet**
- Evaluation of the fine-tuned model on FA-EVAL-120 has **not been run**. Notebook cell 39 does it for zero-shot ECAPA, x-vector and `stage2/best.pt` over all conditions (main15, clean, noisy, tel, snr10, gsm_fr, amr_nb, g711a, opus_nb, ti5s). It must run only after training has ended.
- Still missing baseline numbers: noise at 10 dB, AMR-NB, main15, ti5s.
- Duplicate client IDs (eval vs eval across folds, eval vs pool) not handled.
- S4 per-fold adaptation and score normalisation, S5-S9 not started. ASVspoof 2019 PA source still open.

**Next session**: if the Colab runtime was recycled, follow `docs/RESUME.md` (bootstrap, then `--resume`, which only finishes epoch 11 if needed); then run cell 39 and log the results.


## 2026-11-16 (session 2, from 18:23 UTC) — S4 stage-2 result on FA-EVAL-120, decision-rule experiments, leakage check

**Runtime**: the first Colab runtime was recycled. Stage-2 training had finished before that (12/12 epochs, `done` at 14:53 UTC; final dev EER 1.72% original / 2.47% telephone approximation; `best.pt` = epoch 11). `vbd/bootstrap.py` rebuilt the working directory on the new runtime from Drive archives in 8.5 minutes (18:23-18:32), first real test, worked.

### Full-condition results (text-independent, EER %, 95% speaker-bootstrap CI)

600 genuine / 13,800 impostor trials per condition; main15 = clean + noisy + tel pooled (1,800 / 41,400). Enrolment: 5 original wideband clips.

| Condition | Target | x-vector zero-shot | ECAPA zero-shot | ECAPA stage 2 (fine-tuned) |
|---|---|---|---|---|
| clean | <= 4 (TD) | 13.69 | 3.83 (2.67-5.58) | **2.33** (1.64-3.31) |
| ti5s (>= 5 s) | <= 6 | 13.00 | 3.04 (1.84-4.48) | **2.00** (1.17-2.67) |
| snr10 (MUSAN, 10 dB) | <= 7 | 24.83 | 9.30 (7.66-10.67) | **3.14** (2.36-4.11) |
| gsm_fr | <= 7 | 31.17 | 12.13 (10.00-14.33) | **6.33** (4.74-7.83) |
| amr_nb | <= 7 | 27.83 | 10.66 (8.41-12.66) | **5.82** (4.50-6.83) |
| g711a | <= 7 | 30.20 | 8.67 (7.00-10.83) | **3.79** (2.98-5.17) |
| opus_nb (VoIP) | <= 7 | 25.83 | 9.34 (7.81-11.51) | **4.67** (3.52-6.26) |
| tel (noise 15-30 dB + AMR-NB; stands in for "real call") | <= 7 | 33.31 | 13.00 (10.83-15.33) | **6.21** (4.67-7.93) |
| noisy (MUSAN 5-20 dB) | — | 23.83 | 8.67 | 3.17 (2.08-4.18) |
| main15 | — | 26.55 | 9.67 | 4.04 (3.14-5.17) |

Operating point on main15 (one threshold per rotation from the validation fold, applied to the test fold):

| Model | target FAR | FAR % | FRR % | committed |
|---|---|---|---|---|
| x-vector zero-shot | 0.8 | 0.86 | 65.50 | FAR <= 1, FRR <= 10 |
| ECAPA zero-shot | 0.8 | 0.84 | 29.11 | |
| ECAPA stage 2 | 0.8 | **0.82** | **11.11** | FAR met, **FRR not met** |
| ECAPA zero-shot | 0.1 | 0.13 | 44.94 | stretch: FRR <= 25 |
| ECAPA stage 2 | 0.1 | 0.11 | 25.39 | stretch not met (by 0.4) |

Stage-2 FRR by test group at the common threshold: clean 5.17%, noisy 8.33%, **tel 19.83%**. The telephone tests carry the miss.

**Status against criteria**: 3, 4, 5 met on point estimates (GSM-FR and tel CIs extend to 7.8-7.9, above the 7% line); criterion 1 (FAR) met; criterion 2 (FRR <= 10%) **not met** (11.11%). Criteria 1-3 are text-independent proxies for the proposal's text-dependent mode.

### Decision-rule experiments on the stage-2 embeddings (all negative)

| Variant | FAR % | FRR % (target FAR 0.8) | Note |
|---|---|---|---|
| cosine, one threshold (reference) | 0.82 | 11.11 | |
| AS-norm, cohort 766 pool speakers x {orig, telephone}, top-100 | 0.83 | 11.39 | no gain; EERs equal or slightly worse |
| AS-norm, top-300 | 0.82 | 11.56 | |
| multi-condition enrolment, template = mean(orig, AMR-NB copy) | 0.81 | 12.61 | tel FRR 12.8 but clean 11.0 / noisy 14.0 |
| multi-condition enrolment, mean(orig, AMR-NB, GSM-FR) | 0.82 | 14.56 | |
| A: separate thresholds for wideband and telephone tests | 0.82 (0.64-1.01) | 10.89 (8.00-14.28) | tel FRR 18.3 |
| B: A + narrowband template for telephone tests | 0.83 | 11.78 | tel EER 6.67 vs 6.21 with original template |

These variants were all compared on the test folds, so picking the best of them would be tuning on test data; none is adopted. Conclusion: the decision rule is not the lever; the telephone embedding quality is (tel EER about 6% gives about 18-20% FRR at FAR 0.8%).

### Leakage check (duplicate identities) — found a real issue

Template cosine between FA-EVAL-120 speakers and fine-tuning-pool speakers with the stage-2 model (impostor 99.9% quantile is 0.377; genuine median 0.563):
- 24 eval speakers have a pool speaker with cosine > 0.5 (32 pool speakers involved); 17 > 0.6; 11 > 0.7. Top: S02732/S01778 0.833, S01409/S01633 0.799, S03592/S03993 0.781.
- These are very likely the same people under different Common Voice client IDs, so **stage 2 was trained on voices of up to about 17-24 evaluation speakers**. This violates rule 6 (no leakage).
- Sensitivity: dropping the 17 suspect eval speakers changes main15 EER from 4.04 to 4.14 and FRR from 11.11 to 11.13 (FAR 0.82 to 0.93). The inflation is small, but the stage-2 numbers above must be read with this caveat.
- Eval-eval pairs > 0.6: 3 (different folds).
- Matrices saved: `features/s4_stage2_eval_vs_pool_cos.pkl`, `features/s4_stage2_eval_vs_eval_cos.pkl`.

**Decision**: retrain stage 2 as **stage2v2** with (a) every pool speaker whose cosine to any eval speaker exceeds 0.5 removed from training and from the cohort, and (b) real-codec augmentation (pre-computed AMR-NB and GSM-FR copies of the pool) in addition to the mu-law approximation, to attack the telephone gap. Stage2v2 replaces stage 2 as the reported model. Per-fold stage-3 adaptation follows, excluding adaptation speakers that match a test/validation speaker.

Files: `results/reports/s3_zeroshot_*`, `s4_stage2_ecapa_*`, `s4_stage2_channel_aware_operating_points.csv`; trials in `results/scores/`; embeddings in `features/`.


## 2026-11-16 19:30 UTC — stage2v2 training started; S5/S6 preparation

- Exclusion lists saved: `data/protocols/pool_exclude_dup_suspects.csv` (32 pool speakers with cosine > 0.5 to an eval speaker) and `data/protocols/eval_dup_suspect_pairs.csv` (5 eval-eval pairs > 0.5: S00070/S03824 0.64, S00353/S02732 0.73, S00485/S03253 0.55, S01155/S01500 0.69, S01671/S03341 0.52).
- Real-codec copies of the pool audio computed with `vbd/codec_cache.py` (AMR-NB and GSM-FR, 24,356 clips each, about 38 min with both jobs sharing 2 vCPUs) and copied to Drive `features/cache/`.
- `vbd/train_sv.py` extended: `--exclude`, `--codecs`, `--p_codec`, `--p_mulaw`; checkpoint writes are atomic.
- **stage2v2** started 19:29 UTC: same recipe as stage 2 (12 epochs x 400 iterations, batch 64, same 60 dev speakers) but 32 suspect pool speakers removed (734 training speakers) and 30% of items drawn from the AMR-NB / GSM-FR copies, mu-law share reduced to 20% of the rest. Log: `checkpoints/sv/stage2v2/log.jsonl`. Expected end about 20:25 UTC.
- ASVspoof 2019 LA downloaded to Drive (`data/raw/asvspoof2019_la/data/`, parquet: train 1.6 GB, validation 1.6 GB, test 4.4 GB / 71,237 rows; columns speaker_id, audio_file_name, audio bytes, system_id, key; system_id "-" = bona fide).
- `vbd/cm.py` written: LFCC-LCNN (0.34 M parameters) and RawNet2 (17.6 M parameters); forward pass shape-tested on CPU only. `vbd/train_cm.py` written (ASVspoof cache builder, balanced sampling, noise + telephone augmentation on both classes, per-epoch checkpoints, resume); **not run yet**.
- Persian spoof-generation candidates found on the Hub (not tested): `mazrba/Chatterbox-TTS-Persian-gguf`, `KiaBush/Persian-IPA-to-Speech-F5`, `alikhabazian/XTTS_Persian` (cloning TTS); `Kamtera/persian-tts-*-vits`, piper fa_IR voices, `facebook/mms-tts-fas` (plain TTS); kNN-VC, OpenVoice v2, Seed-VC (voice conversion).

**Next**: evaluate stage2v2 on FA-EVAL-120; decide on per-fold stage-3 adaptation from the FRR result; then build ASVspoof caches and train LCNN / RawNet2.


## 2026-11-16 20:25 UTC — stage2v2 result on FA-EVAL-120 (leak-suspect speakers removed, real-codec augmentation)

Training: 12/12 epochs, finished 20:19 UTC without interruption. Dev (60 held-out pool speakers) best = epoch 9: 1.55% original / 2.14% telephone approximation (stage 2: 1.72 / 2.47). `checkpoints/sv/stage2v2/best.pt`.

Note on runtimes: at about 19:50 a second, CPU-only Colab runtime was opened by a new notebook tab; the training kept running on the original GPU runtime and was monitored through Drive. The MCP connection later re-attached to the original tab.

| Condition | Target | stage 2 | **stage2v2** (95% CI) | Met |
|---|---|---|---|---|
| clean | <= 4 (TD) | 2.33 | 2.50 (1.66-3.35) | yes (text-independent proxy) |
| ti5s | <= 6 | 2.00 | 2.00 (1.25-2.83) | yes |
| snr10 | <= 7 | 3.14 | 3.50 (2.66-4.34) | yes |
| gsm_fr | <= 7 | 6.33 | 5.83 (4.33-7.19) | yes, CI crosses 7 |
| amr_nb | <= 7 | 5.82 | 4.83 (3.68-6.31) | yes |
| g711a | <= 7 | 3.79 | 4.48 (3.33-5.50) | yes |
| opus_nb | <= 7 | 4.67 | 4.50 (3.52-6.17) | yes |
| tel (simulated call) | <= 7 | 6.21 | 5.67 (4.33-7.67) | yes, CI crosses 7 |
| noisy | — | 3.17 | 3.33 | |
| main15 | — | 4.04 | 4.11 (3.06-5.05) | |

Operating point on main15 (threshold from validation fold, one threshold):

| target FAR | FAR % (95% CI) | FRR % (95% CI) | committed / stretch |
|---|---|---|---|
| 0.8 | 0.82 (0.66-0.98) | **10.22** (7.44-13.56) | FAR <= 1 met; FRR <= 10 **not met by 0.22 points** |
| 0.1 | 0.13 | 23.06 | stretch FAR 0.1 not met (0.13); stretch FRR <= 25 met |

FRR by group at the common threshold: clean 5.33, noisy 8.50, tel 16.83 (stage 2: 5.17 / 8.33 / 19.83).

**Reading**
- Real-codec augmentation improved the coded conditions (AMR-NB 5.82 -> 4.83, GSM-FR 6.33 -> 5.83, tel 6.21 -> 5.67, tel FRR 19.8 -> 16.8) at a small cost on clean / G.711 / snr10. Removing the 32 suspect pool speakers did not hurt.
- Criterion 2 is on the line: 10.22% with a CI of 7.4-13.6. With 1,800 genuine trials the test cannot distinguish 10.2 from 10.0; it is reported as **not met**.
- stage2v2 is the reported stage-2 model from here on; stage 2 is kept only for comparison.
- Remaining leakage caveat: the duplicate screen uses a cosine > 0.5 rule on stage-2 embeddings; duplicates below that would not be caught.

**Next**: stage 3 (per-rotation adaptation on the 72 adaptation speakers with the telephone chain, last layers only, early stopping and thresholds on the validation fold), then the countermeasure models.


## 2026-11-17 01:05 UTC — stage 3 result (per-rotation adaptation) and first LCNN epochs; GPU allowance used up

### Stage 3 (ran 20:40-21:21 UTC on 2026-11-16, all 5 rotations completed)

Recipe: start from stage2v2; train only MFA + attentive pooling + embedding layer, lr 2e-5, 3 epochs x 300 iterations, batch 64; 50% of items from the rotation's adaptation speakers (69-70 after dropping 2-3 duplicate suspects per rotation), 50% from the 734 pool speakers; real-codec copies in both. Model selection on the validation fold's main15 EER, with "no adaptation" as a candidate.

| Rotation (test fold) | val EER, no adaptation | epoch 0 | epoch 1 | epoch 2 | selected |
|---|---|---|---|---|---|
| 0 | 5.00 | 5.28 | 5.27 | 5.28 | no adaptation |
| 1 | 2.50 | 2.50 | 2.51 | 2.71 | no adaptation |
| 2 | 4.17 | 4.45 | 4.72 | 4.67 | no adaptation |
| 3 | 3.67 | 3.55 | 3.61 | 3.61 | epoch 0 |
| 4 | 4.21 | n/a in summary | 4.17 | 4.20 | epoch 1 |

Test-fold results pooled over rotations (each rotation's selected model; thresholds from its validation fold):

| Condition | stage2v2 | stage 3 (95% CI) |
|---|---|---|
| clean | 2.50 | 2.50 (1.67-3.33) |
| ti5s | 2.00 | 2.00 (1.32-3.00) |
| snr10 | 3.50 | 3.53 (2.50-4.43) |
| gsm_fr | 5.83 | 5.50 (4.17-7.24) |
| amr_nb | 4.83 | 4.83 (3.69-6.33) |
| g711a | 4.48 | 4.33 (3.20-5.50) |
| opus_nb | 4.50 | 4.70 (3.66-6.17) |
| tel | 5.67 | 5.34 (4.19-7.50) |
| noisy | 3.33 | 3.17 |
| main15 | 4.11 | 3.99 (3.00-5.03) |

| target FAR | FAR % (95% CI) | FRR % (95% CI) | stage2v2 |
|---|---|---|---|
| 0.8 | 0.82 (0.67-0.98) | **10.17** (7.44-13.50) | 0.82 / 10.22 |
| 0.1 | 0.14 (0.08-0.20) | 22.94 (18.55-27.78) | 0.13 / 23.06 |

FRR by group at FAR 0.8%: clean 5.17, noisy 8.50, tel 16.83.

**Reading**: stage 3 is a near-null result. Adaptation was selected in only 2 of 5 rotations and moved FRR from 10.22 to 10.17. **Criterion 2 (FRR <= 10%) is still not met**; the miss (0.17 points) is far inside the confidence interval. Adding 69-70 speakers to a model already trained on 734 Persian speakers with the same augmentation changes little. Ideas not yet tried for the telephone FRR: longer or multi-utterance test input, larger backbone / different pretrained model, more Persian speakers (the rest of Common Voice has thousands of low-count speakers; pool minimum was 8 clips), DeepMine if licensed.

### Countermeasure, stage 1 (LCNN on ASVspoof 2019 LA) — partial

Started automatically after stage 3 (21:21 UTC). ASVspoof train/validation caches were built on the Colab disk. Dev EER (ASVspoof 2019 LA validation split, full) per epoch: 10.01, 2.63, 2.71, 1.30, 1.10, 1.41, 0.94, 1.18, 1.21, **0.63** (epoch 9, saved as `checkpoints/cm/lcnn_asv19la/best.pt`), 1.57 (epoch 10). About 2.6 min per epoch. The run stopped in epoch 11 of 30 at 21:54 UTC when the GPU runtime ended (Colab free allowance); the last traceback is a Drive `FileNotFoundError` from the unmount. `last.pt` holds epoch 10; `--resume` continues from epoch 11.
- This is in-domain dev performance on known attack types only. It says nothing yet about Persian spoofs, telephone/noisy bona fide audio, or the ASVspoof evaluation split.
- RawNet2 has not started.

### State
- New runtime at 00:55 UTC is CPU-only. Stage 3 was scored on CPU from the saved embeddings.
- `docs/STATE.md` updated with what to run next.


## 2026-11-17 01:15 UTC — S8 first pass on a CPU-only runtime: service logic, template store, latency

- `vbd/system.py` written: `TemplateStore` (SQLite, one row per user: user id, vector dimension, 192-d float32 vector as a 768-byte blob, number of enrolment utterances, timestamps; no audio column) and `VoiceAuth` (enrol, verify, liveness, decision; returns decision, reason, speaker score, liveness score, user id, timestamp, processing time). Thresholds in the class are placeholders (speaker 0.35, liveness 0.0); real ones come from S7.
- Latency benchmark (criterion 9), Colab CPU runtime, **2 vCPU** Intel Xeon 2.20 GHz, no GPU, torch threads = 2. Pipeline timed from reading the mp3 file to the decision: decode + resample, ECAPA (stage2v2) embedding, LCNN liveness (epoch-9 checkpoint), cosine against the stored template. 10 enrolled users, 200 requests on test clips <= 5 s (mean 3.91 s, max 4.99 s), 5 warm-up requests not timed.

| | value |
|---|---|
| median | 1,199 ms |
| mean | 1,355 ms |
| **p95** | **2,299 ms** (target <= 2,000 ms on an 8-core CPU) |
| max | 3,172 ms |
| share under 2 s | 88.0% |
| share under 1 s | 18.5% |

Mean time per stage (50 requests): decode + resample 6 ms, ECAPA embedding 1,272 ms, LCNN liveness 58 ms.

- **Criterion 9 is not met on this 2-vCPU machine** (p95 2.30 s). The proposal specifies an 8-core CPU, which was not available, so the criterion is unverified on the specified hardware rather than failed. The ECAPA embedding is 94% of the time; that is the place to optimise (shorter analysis window, ONNX / quantised export, feature extraction cost). Not tried yet.
- Criterion 10 (storage inspection after enrolling 10 users): the store directory contains one file, `templates.sqlite` (24,576 bytes); table columns are user_id, dim, vec, n_utts, created, updated; each row holds a 768-byte vector; no audio file exists in the store. Enrolment audio is only held in memory. Not yet done: encryption of the stored vectors at rest (the proposal mentions cryptographic handling).
- Files: `results/reports/s8_latency_cpu2core.csv`, `s8_latency_cpu2core_ms.npy`.

## 2026-11-17 01:20 UTC — git mirror

Until now nothing had been committed to the project's git repository (everything was kept on Drive only). From this point the repo mirrors `vbd/`, `docs/`, `results/reports/*.csv`, `data/protocols/*.csv`, training logs (`results/train_logs/`) and launcher scripts. Models, embeddings, audio, caches and score files stay on Drive only. A handoff guide for other Claude models is in the repo root as `CLAUDE.md`.

## 2026-11-17 03:40 UTC — S6a finished: LCNN and RawNet2 trained on ASVspoof 2019 LA, scored on the held-out evaluation split and on genuine Persian speech

Ran on the second Google account's T4 runtime (project folder reached through the shared-folder shortcut). Launcher: `logs/chain_cm.sh` (bootstrap, LCNN resume, RawNet2), log `logs/chain_cm.log`. Evaluation: `python -m vbd.eval_cm`, log `logs/eval_cm.log`.

### Training

| Model | Params | Epochs | Time per epoch | Best dev EER (epoch) | Last-epoch dev EER | Checkpoint |
|---|---|---|---|---|---|---|
| LFCC-LCNN | 0.34 M | 30 | ~2.0 min | 0.58% (21) | 0.75% | `checkpoints/cm/lcnn_asv19la/best.pt` |
| RawNet2 | 17.6 M | 20 | ~2.6 min | 1.38% (14) | 1.53% | `checkpoints/cm/rawnet2_asv19la/best.pt` |

Both: 4 s inputs, class-balanced sampling, noise and telephone augmentation on both classes, checkpoint to Drive every epoch. RawNet2 took 52 minutes in total (the earlier guess of "a few hours" was wrong).

### ASVspoof 2019 LA evaluation split (7,355 bona fide, 63,882 spoof; attacks A07-A19, none seen in training)

Table: `results/reports/s6a_cm_asvspoof2019la.csv`. "Spoof accepted" and "bona fide rejected" use the threshold fixed at the development-set EER point (LCNN -1.93, RawNet2 -3.08) and applied unchanged.

| | LCNN | RawNet2 |
|---|---|---|
| Dev EER | 0.58 | 1.38 |
| **Eval EER, pooled** | **6.47** | **7.54** |
| Eval spoof accepted at dev threshold | 12.25 | 16.50 |
| Eval bona fide rejected at dev threshold | 0.84 | 1.51 |

Per-attack EER % (each attack's 4,914 spoofs against all 7,355 bona fide):

| Attack | LCNN | RawNet2 | | Attack | LCNN | RawNet2 |
|---|---|---|---|---|---|---|
| A07 | 1.22 | 3.02 | | A14 | 0.73 | 2.23 |
| A08 | 0.81 | 5.21 | | A15 | 0.67 | 3.28 |
| A09 | 0.20 | 0.69 | | A16 | 0.95 | 2.30 |
| A10 | 1.43 | 3.48 | | A17 | 10.28 | 12.86 |
| A11 | 0.71 | 3.01 | | A18 | 26.33 | 30.79 |
| A12 | 1.12 | 4.07 | | A19 | 4.70 | 4.01 |
| A13 | 0.47 | 1.02 | | | | |

Reading:
- Both models generalise to 10 of the 13 unseen attacks (LCNN below 1.5% EER on each). The pooled figure is dominated by the three voice-conversion attacks A17, A18, A19. At the dev threshold LCNN accepts 37% of A17 and 94% of A18; RawNet2 accepts 42% and 91%. A17 is the known hard case of this benchmark (waveform-filtering VC); A18 failing this badly is worse than published LCNN/RawNet2 numbers and is probably an effect of the telephone augmentation and of training on the train split only. Not investigated.
- LCNN is better than RawNet2 on the pooled figure and on 12 of 13 attacks, with 50 times fewer parameters and 58 ms CPU time. RawNet2 wins only on A19.
- These are English, studio-quality results. They do not count towards criterion 6, which is defined on Persian spoofs (S5/S6b).

### False alarms on genuine Persian speech (FA-EVAL-120 test clips, 600 per condition, no spoofs involved)

Table: `results/reports/s6a_cm_persian_bonafide_rejection.csv`. Share of genuine Persian clips rejected as spoof at the same dev-EER threshold:

| Condition | LCNN | RawNet2 |
|---|---|---|
| clean | 6.00 | 13.00 |
| ti5s (>= 5 s) | 5.33 | 12.17 |
| noisy | 6.50 | 9.67 |
| snr10 | 6.83 | 9.67 |
| tel (simulated call) | 6.17 | 13.00 |
| g711a | 7.33 | 10.67 |
| amr_nb | 5.83 | 14.50 |
| opus_nb | 3.50 | 15.00 |
| **gsm_fr** | **22.83** | 17.00 |

Reading:
- On ASVspoof's own bona fide speech the same thresholds reject 0.84% (LCNN) and 1.51% (RawNet2). On genuine Persian Common Voice speech they reject 6% and 13% even with no codec. This is a domain shift (language, crowd-sourced microphones, MP3 source), not a channel effect: clean and telephone conditions are about the same, except GSM-FR for LCNN (22.8%).
- Consequence for the system: used as they are, the detectors alone would add about 6 points (LCNN) to the false-reject rate, on top of the speaker engine's 10.2%. Criterion 8 (integrated FRR <= 12%) cannot be met without Persian fine-tuning of the detector and a threshold set on Persian validation data. That is what S6b is for; this table is the "before" measurement.
- GSM-FR must be among the codecs applied to bona fide audio in S6b training.

### Decision
LCNN is the primary countermeasure candidate (better accuracy, far lower cost); RawNet2 is kept for the comparison table and as a possible second score in fusion. Final choice after S6b.

### Next
S5: Persian spoof generation (simulated replay, cloned TTS, voice conversion), starting with a quality test of candidate models on a few samples.

## 2026-11-17 03:50 UTC — S5a: simulated replay set generated and scored before any Persian detector training

### Generation (`python -m vbd.spoof_replay`, CPU, 32 s; log `logs/spoof_replay.log`)
- 1,200 clips: 120 speakers x 10, 1.5 hours of audio. Index: `data/protocols/fa_spoof_replay.csv`. Audio: Drive `data/spoof/replay.tar` (110 MB, FLAC 16 kHz).
- Source of each clip: a genuine recording of the target speaker that is neither an enrolment nor a test clip (3-10 s).
- Chain: loudspeaker model (device band-pass, one resonance, tanh soft clipping) -> room impulse response (RIRS_NOISES simulated rooms) -> ambient noise (MUSAN noise, SNR 18-35 dB) -> 70 Hz microphone high-pass.
- Three device profiles, 400 clips each: phone speaker (450-7200 Hz, strong clipping), laptop speaker (220-7600 Hz), external speaker (90-7900 Hz, light clipping).
- Substitution: the proposal asked for physical replays through 3 real playback devices. This is a software model; it has no real loudspeaker or microphone response and no recorded room. Results on it must be labelled "simulated replay".

### Scoring (`python -m vbd.eval_spoof replay`; table `results/reports/s5_replay_zeroshot.csv`)
Speaker engine: stage2v2, replay clip scored against its target's enrolment template. Threshold 0.292 = pooled over all five folds at FAR 0.8% on the clean impostor trials (a quick pooled threshold for this check only, not the per-rotation validation thresholds of the reported results); genuine clean test clips pass it 94.0% of the time.
Detectors: the ASVspoof-trained LCNN and RawNet2, no Persian training, compared against the 600 genuine clean Persian test clips. All figures in %.

| Device | Passes speaker engine | LCNN EER | LCNN accepts (ASVspoof threshold) | RawNet2 EER | RawNet2 accepts | Passes both, LCNN | Passes both, RawNet2 |
|---|---|---|---|---|---|---|---|
| external speaker | 90.5 | 39.0 | 93.3 | 41.7 | 70.5 | 84.5 | 64.0 |
| laptop speaker | 81.8 | 38.7 | 93.5 | 48.8 | 81.0 | 76.5 | 65.3 |
| phone speaker | 81.5 | 39.5 | 96.0 | 47.3 | 84.8 | 78.0 | 69.5 |
| **all 1,200** | **84.6** | **39.2** | **94.3** | **46.2** | **78.8** | **79.7** | **66.3** |

Reading:
- Replay keeps the speaker's identity: 85% of replays pass the speaker engine (genuine clips: 94%). The speaker engine gives no protection against replay, as expected.
- Detectors trained only on ASVspoof LA (synthetic speech and voice conversion) do not detect replay: EER 39-46%, near chance. Without Persian replay training about 80% of replay attacks would pass the whole system with LCNN (criterion 7 target: <= 10%).
- This is the "before" number for S6b. It also shows the cost of not having ASVspoof 2019 PA (the replay part of the benchmark; download blocked with HTTP 403): replay detection has to be learnt entirely from the simulated Persian set.
- A warning about a failed 690 MB GPU allocation appeared in `logs/eval_spoof_replay.log`; the run finished and wrote its table. Not investigated.

### Cloned TTS and voice conversion
Not generated yet at this point; see the probe entry below.

### Git mirror
The replay code, index, result table and logs reached the repo later the same day (the browser had blocked the download until the user re-allowed automatic downloads for the Colab tab).

## 2026-11-17 06:10 UTC — S5 probe: voice conversion and cloned TTS on 6 speakers (first Persian synthetic spoofs)

The user approved the run. `python -m vbd.spoof_probe 6` on a CPU-only runtime (the script now picks GPU or CPU by itself); 15 minutes in total, about 1.7 minutes per speaker on CPU. Log `logs/spoof_probe.log`, scores `results/reports/s5_probe_scores.csv`, the 60 audio files in Drive `data/spoof/probe/` for listening.

Systems tried:
- **knnvc**: kNN-VC (`torch.hub` `bshall/knn-vc`, WavLM-Large features + HiFi-GAN, language-independent). Source: a clean test clip of another evaluation speaker from a different fold. Target: 12 clips of the victim that are neither enrolment nor test clips (46-70 s in total).
- **mms**: `facebook/mms-tts-fas` (Persian VITS, one fixed voice, 16 kHz, takes Persian script directly) reading Common Voice sentences. No cloning; included as a reference.
- **mms_knnvc**: the MMS output converted to the victim's voice with kNN-VC, i.e. a cloned-TTS attack.

6 speakers (4 M, 2 F), 2 clips per speaker per system, 12 clips per row. Speaker threshold 0.292 (pooled, FAR 0.8%, same as the replay check). Detector thresholds are the ASVspoof dev-EER points (LCNN -1.93, RawNet2 -3.08).

| Clips | Mean cosine to victim's template | Pass speaker engine | Pass LCNN | Pass RawNet2 |
|---|---|---|---|---|
| genuine (victim's own clean test clips) | 0.588 | 12 / 12 | 12 / 12 | 12 / 12 |
| impostor (the VC source speakers, unconverted) | 0.024 | 0 / 12 | 8 / 12 | 7 / 12 |
| knnvc | 0.467 | 12 / 12 | 9 / 12 | 10 / 12 |
| mms (fixed TTS voice) | 0.002 | 0 / 12 | 2 / 12 | 6 / 12 |
| mms_knnvc (cloned TTS) | 0.449 | 12 / 12 | 5 / 12 | 8 / 12 |

Reading (12 clips per row, so these are indications, not rates):
- kNN-VC is an effective attack on the speaker engine: every converted clip passed, both from human speech and from TTS, with about one minute of the victim's audio. It is suitable for bulk generation of both the voice-conversion and the cloned-TTS sets.
- The ASVspoof-trained detectors miss most kNN-VC audio (LCNN passes 9 of 12 VC clips and 5 of 12 cloned-TTS clips). Raw MMS output is mostly caught by LCNN (10 of 12 rejected).
- The detectors also rejected 4-5 of 12 genuine clips of the source speakers, consistent with the false-alarm problem measured in S6a.
- Audio quality was not judged by ear by anyone yet; the files are on Drive for that.
- Only one VC system and one TTS system were tried. The proposal's unseen-attack test needs at least one more system of each kind (candidates listed in the 2026-11-16 19:30 entry; none of those tested).
