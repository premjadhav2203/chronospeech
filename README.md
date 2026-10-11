# Contrastive Speech Analytics & Temporal Flaw Grounding Dataset
### Multimodal AI Hackathon 2026 — Track C

## 1. Project Overview

In spoken communication, competitive forensics, debate, and executive coaching, delivery flaws—such as rushed pacing, flat pitch intonation, awkward hesitations, or inadequate vocal energy—degrade audience engagement and clarity.

Traditional speech evaluation systems output a single global score (e.g., "72/100") without explaining where in time a mistake occurred or why it degraded the speech. 

This project provides a contrastive speech dataset and evaluation pipeline designed for **Temporal Flaw Grounding**:
* Pinpointing start and end timestamps of delivery flaws.
* Measuring acoustic deviations against a good baseline.
* Generating causal, metric-backed explanations for personalized feedback.

---

## 2. Track C Context

**Track C: Contrastive Speech Analytics & Temporal Flaw Grounding** requires multimodal systems that:
1. Receive an identical transcript spoken across varying delivery qualities.
2. Contrast flawed participant audio against an ideal reference delivery (*Good Baseline*).
3. Localize flaws to exact millisecond intervals (*Temporal Grounding*).
4. Provide verifiable acoustic metric deltas ($\Delta$) coupled with interpretable explanations.

This repository provides the audio recordings, matching transcripts, extracted acoustic features, word-level alignments, and contrastive flaw annotations for ten speeches.

---

## 3. Dataset Structure

Existing speech datasets (such as LibriSpeech or Common Voice) focus on acoustic diversity across thousands of speakers, but lack controlled, paired contrastive variations of the same passage.

This dataset introduces a controlled contrastive structure:
* **10 Original Speeches** spanning diverse oratorical genres (persuasive oratory, declamation, interpretive reading, and extemporaneous speaking).
* **7 Parallel Audio Versions per Speech** reading the **exact same transcript**:
  * 1 Ideal Baseline (`good`)
  * 6 Targeted Flawed Versions (`near_perfect`, `mild_pacing`, `severe_pacing`, `flat_tone`, `bad_pause`, `low_volume`)
* **Strict Semantic Equivalence**: Because the spoken words are identical across all 7 versions of each speech, all performance differences reside in the acoustic, prosodic, and temporal domains.

---

## 4. Contrastive Methodology

Each speech passage acts as an independent experimental unit:

```
[Canonical Transcript]  ─── Identical text across all versions ───┐
                                                                 │
┌───────────────────────┬────────────────────────────────────────▼─────────────────────────────┐
│ Ideal Baseline        │ 1. good              (Severity 0: Reference standard)                 │
├───────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Pacing Flaws          │ 2. near_perfect      (Severity 1: Minor tempo irregularity)           │
│                       │ 3. mild_pacing       (Severity 2: Noticeably rushed tempo)            │
│                       │ 4. severe_pacing     (Severity 3: Severe rush / hurry)                │
├───────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Pitch Flaws           │ 5. flat_tone         (Severity 2: Monotone / compressed F0 contour)   │
├───────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Pause Flaws           │ 6. bad_pause         (Severity 3: Unnatural silence / jarring gap)    │
├───────────────────────┼──────────────────────────────────────────────────────────────────────┤
│ Volume Flaws          │ 7. low_volume        (Severity 2: Suppressed vocal intensity / RMS)   │
└───────────────────────┴──────────────────────────────────────────────────────────────────────┘
```

By keeping the semantic content invariant, model architectures can perform direct temporal and spectral subtraction against the baseline to ground anomalies without confounding linguistic shifts.

---

## 5. The Good Baseline

The `good.wav` recording serves as the **anchor truth** for each speech:
* **Cadence & Pacing**: Natural conversational or oratorical speed (approx. 130–150 words per minute / 2.2–2.5 words per second).
* **Dynamic Pitch ($F_0$) Range**: Expressive rising and falling inflections signaling syntactic clauses, emphasis, and rhetorical questions.
* **Pause Placement**: Pauses occur naturally at commas, periods, and rhetorical transitions, ranging from 0.3s to 0.8s.
* **Acoustic Energy**: Strong, balanced vocal projection without distortion or clipping.

---

## 6. The Flawed Spectrum

Delivery degradation is not binary; speakers exhibit a continuous spectrum of performance degradation. This dataset structures flaws along a graded continuum:

* **Severity 0 (Ideal Baseline)**: Flawless baseline execution (`good.wav`).
* **Severity 1 (Near-Perfect / Minor Deviation)**: High-quality delivery containing subtle deviations (`near_perfect.wav`) that require sensitive alignment algorithms to detect.
* **Severity 2 (Moderate Flaw)**: Noticeable errors that distract an audience (`mild_pacing.wav`, `flat_tone.wav`, `low_volume.wav`).
* **Severity 3 (Severe Flaw)**: Critical performance breakdowns (`severe_pacing.wav`, `bad_pause.wav`) causing serious comprehension deficits.

---

## 7. Flaw Categories

The dataset categorizes speech delivery challenges into standard taxonomy codes:

| Flaw Type | Target Feature | Description | Example Indicator |
|:---|:---|:---|:---|
| `none` | `all` | Baseline execution; balanced prosody. | Reference profile |
| `pacing` | `speech_rate` | Uncontrolled delivery rate (excessively fast or slow). | Words/sec $> 3.5$ or $< 1.5$ |
| `pitch` | `pitch` ($F_0$) | Monotone delivery, lack of emotional inflection, or inappropriate pitch glides. | Narrow $F_0$ standard deviation |
| `pause` | `pause_duration` | Unnatural silence mid-phrase, excessive duration ($>1.5\text{s}$), or missing punctuation pauses. | Disrupted syntactic grouping |
| `volume` | `energy` (RMS) | Vocal fade, lack of projection, or abrupt drop in loudness. | Low RMS energy below noise floor |
| `clarity` | `mfcc` / `fft` | Mumbled articulation, slurred consonants, or muffled acoustic spectrum. | Spectral smearing / low high-freq power |
| `mixed` | Multiple | Compound flaws combining multiple simultaneous degradations. | Rushed tempo + monotone intonation |

---

## 8. Severity Levels

| Severity Score | Label | Operational Definition |
|:---:|:---|:---|
| **0** | Ideal / No Flaw | Baseline standard; no discernible deviation from oratorical best practices. |
| **1** | Near-Perfect | Minor deviation (e.g., $10\%\text{--}20\%$ rate increase on one clause); listener comprehension unaffected. |
| **2** | Moderate Flaw | Evident flaw (e.g., sustained $25\%\text{--}40\%$ deviation, monotone delivery); diminishes audience engagement. |
| **3** | Severe Flaw | Crippling delivery error (e.g., $>50\%$ deviation, $>1.5\text{s}$ awkward dead-air pause, severe rush); breaks speech intelligibility. |

---

## 9. Temporal Annotation Methodology

Temporal flaw grounding requires localizing errors in the time domain rather than assigning a global clip-level label.

In `annotations/temporal_annotations.csv`, each flaw interval is defined by:
* `start_time` & `end_time`: Start and end in seconds (float, e.g., `12.45` to `15.80`).
* `duration`: Length of the flawed interval (`end_time - start_time`).
* `affected_text`: Transcript phrase spoken during the interval (e.g., `"chiseled in the quiet, unremarkable choices"`).
* `baseline_value`: Quantitative measurement of the good baseline over the identical words.
* `participant_value`: Quantitative measurement of the participant recording during the interval.
* `delta` & `delta_percentage`: $\text{Delta} = \text{participant} - \text{baseline}$; $\%\Delta = \frac{\Delta}{\text{baseline}} \times 100$.
* `causal_explanation`: Explainable AI diagnosis linking acoustic delta to oratorical impact.

---
## 10. Forced Alignment

To associate acoustic frames with exact words, the pipeline uses acoustic forced alignment with the Vosk/Kaldi HMM-DNN speech recognition engine (`vosk-model-small-en-us-0.15`):
* Canonical transcripts from `transcripts/speech_XXX.txt` are normalized and tokenized into words.
* The acoustic model decodes audio waveforms into time-aligned word hypotheses with acoustic confidence scores.
* Dynamic sequence alignment maps recognized acoustic tokens to canonical transcript words.
* Word records in `annotations/word_alignment.csv` include:
  * `speech_id`: Identifier for the speech (`speech_001` to `speech_010`).
  * `version`: Recording variant (`good`, `near_perfect`, etc.).
  * `word_index`: 1-indexed sequential word position.
  * `word`: Orthographic word string.
  * `start_time`: Word onset timestamp in seconds.
  * `end_time`: Word offset timestamp in seconds.
  * `duration`: Word duration in seconds (`end_time - start_time`).
  * `confidence`: Acoustic alignment confidence score in $[0.0, 1.0]$.

---

## 11. Acoustic Feature Extraction

The dataset extracts prosodic and acoustic representations across six feature domains:

1. **MFCC (`features/mfcc.csv`)**:
   * 13 Mel-Frequency Cepstral Coefficients capturing vocal tract spectral envelope.
   * Frame size: 1024 samples, Hop length: 512 samples (~32ms frames at 16kHz).
2. **Pitch / $F_0$ (`features/pitch.csv`)**:
   * Fundamental frequency in Hertz ($F_0$) extracted via normalized autocorrelation.
   * Frame size: 1024 samples, Hop length: 256 samples, frequency search range 65 Hz to 400 Hz.
3. **RMS Energy (`features/energy.csv`)**:
   * Root Mean Square signal amplitude measuring vocal intensity.
   * Frame size: 2048 samples, Hop length: 512 samples.
4. **Speech Rate (`features/speech_rate.csv`)**:
   * Global speaking rate and vocalized speaking rate (excluding pause intervals) in words per second (WPS).
5. **Pauses (`features/pauses.csv`)**:
   * Silence detection measuring intervals $\ge 300\text{ms}$ based on energy drops below background thresholds.
6. **FFT Spectrum (`features/fft.csv`)**:
   * Magnitude spectra across representative octave bands (125 Hz to 8000 Hz).

---

## 12. Baseline Comparison

The comparison module (`scripts/compare_baseline.py` and `scripts/core_analytics.py`) evaluates participant recordings by computing differential metrics against each speech's own `good` recording:

$$\Delta_{\text{feature}} = F_{\text{participant}} - F_{\text{baseline}}$$

1. **Word Synchronization**: Word boundaries from `word_alignment.csv` establish correspondence between the baseline and flawed variants.
2. **Region Localization**: Sliding window search identifies the temporal window $[t_{\text{start}}, t_{\text{end}}]$ with maximal acoustic divergence (e.g. rate rush, monotone pitch standard deviation drop, or energy loss).
3. **Flaw Grounding**: The localized interval, measured metric delta ($\Delta$ and $\%\Delta$), and causal feedback explanation are saved to `annotations/temporal_annotations.csv`.

---

## 13. Causal Explanations

Acoustic deltas are converted into structured explanations linking measurements to delivery impact:

* **Pacing Flaw Example**:
  > *"Speaking rate accelerated to 3.85 wps (+48.1% vs baseline 2.60 wps) over interval [12.40s - 15.10s] (2.70s) across phrase: 'in the quiet, unremarkable choices'."*
* **Pause Flaw Example**:
  > *"Longest silence interval reached 2.45s (+390.0% longer than baseline average pause of 0.50s) between 16.20s and 18.65s (Unnatural silence gap between 'honesty' and 'when'), breaking communicative continuity."*
* **Pitch Flaw Example**:
  > *"Pitch intonation standard deviation collapsed by 62.4% (12.30 Hz vs 32.70 Hz baseline) over interval [8.10s - 12.30s] across phrase: 'the true test of character', producing localized monotone delivery."*

---

## 14. Dataset Structure

```
speech-contrastive-dataset/
├── README.md
├── audio/
│   ├── speech_001/ ... speech_010/        # 7 audio variants per speech (.m4a / .wav)
├── transcripts/
│   ├── speech_001.txt ... speech_010.txt  # Canonical text passages
├── annotations/
│   ├── word_alignment.csv                 # Master word alignments (6,335 words)
│   ├── temporal_annotations.csv           # Master grounded flaws (60 flaws)
│   ├── labels.csv                         # Master recording labels (70 records)
│   └── per_speech/                        # Per-speech annotation CSVs
├── features/
│   ├── energy.csv                         # Frame-level RMS energy
│   ├── pitch.csv                          # Frame-level F0 pitch
│   ├── pauses.csv                         # Detected pause intervals
│   ├── fft.csv                            # FFT spectral magnitudes
│   ├── mfcc.csv                           # 13 MFCC coefficients
│   ├── speech_rate.csv                    # Global and active speech rates
│   └── per_speech/                        # Per-speech feature CSVs
├── metadata/
│   ├── speeches.csv                       # Speech titles, genres, durations
│   ├── processing_status.json             # Batch status summary
│   └── pipeline_execution_log.json        # Execution logs and error counts
├── scripts/
│   ├── batch_pipeline.py                  # Batch orchestrator
│   ├── core_analytics.py                  # DSP and feature extraction library
│   ├── forced_alignment.py                # Vosk word alignment
│   ├── compare_baseline.py                # Contrastive analysis
│   └── extract_*.py                       # Individual feature extractors
└── config/
    └── dataset_config.json                # Processing configuration
```

---

## 15. Dataset Summary

All recordings, features, and annotations have been processed and validated:

| Component | Count | Description |
|:---|:---:|:---|
| Speeches | 10 | Diverse oratorical and declamation passages |
| Audio Recordings | 70 | 7 parallel variants per speech (1 baseline + 6 flawed) |
| Canonical Transcripts | 10 | Exact passage texts |
| Word Alignment Records | 6,335 | Acoustic start/end timestamps and confidence scores |
| Grounded Temporal Flaws | 60 | Measured flaw intervals, deltas, and causal explanations |
| Labeled Recordings | 70 | Severity (0–3), flaw type, and target acoustic feature |

All acoustic measurements and alignment timestamps are computed directly from the recorded audio files.

---

## 16. Batch Processing Pipeline

The pipeline supports end-to-end processing across all speeches:

```bash
# Process all speeches in the dataset:
python scripts/batch_pipeline.py

# Process a single target speech:
python scripts/batch_pipeline.py --speech_id speech_001
```

### Pipeline Steps:
1. Discovers audio variants in `audio/speech_XXX/`.
2. Converts `.m4a` files to standardized 16kHz mono `.wav` using `/usr/bin/afconvert` or `ffmpeg`.
3. Runs acoustic forced alignment via Vosk/Kaldi to generate word timestamps.
4. Computes frame-level RMS energy, F0 pitch, silence pauses, FFT spectra, and MFCCs.
5. Computes contrastive deltas against the speech's `good` baseline.
6. Writes per-speech CSVs and updates master dataset files.

---

## 17. Scoring Engine & Universal Evaluation

The project includes an objective scoring layer (`scripts/scoring_engine.py`) that evaluates any participant recording against its speech's `good` baseline:

$$\text{Overall Score } S = S_{\text{pacing}} + S_{\text{pitch}} + S_{\text{volume}} + S_{\text{fluency}} \quad \in [0, 100]$$

Each dimension contributes $0\text{--}25$ points computed deterministically from acoustic differentials:

1. **Pacing & Cadence ($S_{\text{pacing}} \in [0, 25]$)**:
   - Uses aligned word timestamps to compute global speaking rate and localized window rates ($K=10$ words).
   - $D_{\text{pacing}} = 0.40 \cdot \delta_{\text{global}} + 0.60 \cdot \min(2.5, \, \delta_{\text{local}})$.
   - $S_{\text{pacing}} = 25.0 \cdot \max\left(0.0, \, 1.0 - \frac{D_{\text{pacing}}}{2.0}\right)$.
   - Preserves delta sign for directionality ($+\Delta$ = rushed, $-\Delta$ = dragged).

2. **Pitch Dynamics ($S_{\text{pitch}} \in [0, 25]$)**:
   - Penalizes collapse in global F0 standard deviation, dynamic range ($IQR$), and localized monotone inflection.
   - $D_{\text{pitch}} = 0.35 \cdot C_{\sigma} + 0.25 \cdot C_{\text{range}} + 0.40 \cdot C_{\text{local}}$.
   - $S_{\text{pitch}} = 25.0 \cdot \max\left(0.0, \, 1.0 - \min(1.0, \, 1.25 \cdot D_{\text{pitch}})\right)$.

3. **Volume & Audibility ($S_{\text{volume}} \in [0, 25]$)**:
   - Uses symmetric global energy deviation around the baseline projection plus localized vocal dropouts.
   - $D_{\text{volume}} = 0.40 \cdot \min(1.5, \, \delta_{E, \text{global}}) + 0.60 \cdot \min(1.5, \, \text{Dip}_{\text{local}})$.
   - $S_{\text{volume}} = 25.0 \cdot \max\left(0.0, \, 1.0 - \frac{D_{\text{volume}}}{1.25}\right)$.

4. **Fluency & Pauses ($S_{\text{fluency}} \in [0, 25]$)**:
   - Evaluates dead-air silence duration with a stabilized baseline floor ($P_{\text{denom}} = \max(0.40, \overline{P}_{\text{base}})$) along with pause frequency and duty cycle.
   - $D_{\text{fluency}} = 0.60 \cdot \min(2.5, \, \delta_{P, \max}) + 0.20 \cdot \min(1.5, \, \delta_{F}) + 0.20 \cdot \min(1.5, \, \delta_{\rho})$.
   - $S_{\text{fluency}} = 25.0 \cdot \max\left(0.0, \, 1.0 - \frac{D_{\text{fluency}}}{1.8}\right)$.

### Severity Rubric (Heuristic Calibration)
* **Severity 0 (Normal / Clean)**: Distortion $D < 0.25$ (Score $21.8\text{--}25.0$).
* **Severity 1 (Mild Deviation)**: Distortion $0.25 \le D < 0.60$ (Score $17.0\text{--}21.8$).
* **Severity 2 (Moderate Flaw)**: Distortion $0.60 \le D < 1.10$ (Score $10.0\text{--}17.0$).
* **Severity 3 (Severe Flaw)**: Distortion $D \ge 1.10$ (Score $< 10.0$).

### Temporal IoU Evaluation
Detected intervals are generated strictly from acoustic comparison (without reference annotation leakage) and evaluated against reference annotations via Intersection over Union:
$$\text{IoU} = \frac{\text{Intersection}}{\text{Union}}$$
Mapping: `pacing` $\leftrightarrow$ `pacing`, `flat_tone` $\leftrightarrow$ `pitch`, `bad_pause` $\leftrightarrow$ `fluency`, `low_volume` $\leftrightarrow$ `volume`.

---

## 18. Interactive Evaluation Dashboard

A local interactive dashboard is provided in `dashboard/`:

```bash
# Start dashboard server:
python dashboard/app.py

# Open in browser:
http://localhost:8080
```

Features:
- Speech selector (`speech_001` through `speech_010`) and recording variant selector.
- 0–100 delivery score display with 4-dimension breakdown.
- Multi-flaw detection cards with causal coaching explanations, metric deltas, and severity badges.
- Synchronized visual canvas timeline showing color-coded flaw intervals.
- Integrated audio playback with "Play Segment" seek buttons.
- Canonical transcript view with flawed phrase highlighting.

---

## 19. Dependencies & Tools

* **Python 3.9+**
* **Vosk / Kaldi**: Acoustic speech recognition and forced alignment
* **SoundFile**: Audio I/O (`libsndfile`)
* **NumPy**: Signal processing and array mathematics
* **macOS afconvert / ffmpeg**: Audio format conversion

---

## 20. Citation & License

This dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. All speech texts in `transcripts/` are original passages created for the Multimodal AI Hackathon 2026.
>>>>>>> f986095 (Initial commit: chronoSpeech)
# chronospeech
