# Speech Contrastive Analysis Dashboard & Dataset

This archive contains the complete Speech Contrastive Dataset project (Speech 001–Speech 010 with all 7 acoustic variants), deterministic scoring engine, evaluation suite, and interactive web dashboard.

---

## 1. Prerequisites
- Python 3.9 or higher (tested on Python 3.10+)
- macOS / Linux / Windows

---

## 2. Quick Setup

1. **Extract the archive:**
   ```bash
   unzip speech_contrastive_dataset_project.zip
   cd speech_contrastive_dataset_project
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 3. Running the Interactive Dashboard

Start the local dashboard server:
```bash
python3 dashboard/app.py --port 8080
```

Open your browser at:
```
http://localhost:8080
```

### Dashboard Features:
- **Speech Selector:** Select any speech from `speech_001` to `speech_010`.
- **Variant Selector:** Choose from all 7 variants:
  - `good` (Baseline reference)
  - `near_perfect` (Minor human nuances)
  - `mild_pacing` (Slight cadence deviation)
  - `severe_pacing` (Significant rush/drag)
  - `flat_tone` (Monotone collapse)
  - `bad_pause` (Unnatural silence breaks)
  - `low_volume` (Audibility degradation)
- **Scores:** Overall score out of 100, plus four 25-point dimension scores (Pacing, Pitch, Volume, Fluency).
- **Flaw Analysis:** Acoustic evidence, baseline comparison, and human-readable causal explanation.
- **Visual Timeline:** Canvas visualization of detected flaw regions and pause anchors.
- **Audio Playback:** Built-in audio player with "Play Segment" timestamp seeking.

---

## 4. Running the Evaluation Benchmark

To re-run the deterministic evaluation benchmark across all 70 dataset recordings:
```bash
python3 evaluation/evaluate_all.py
```
Outputs will be saved in `evaluation/`:
- `evaluation/evaluation_results.json`
- `evaluation/evaluation_results.csv`
- `evaluation/benchmark_summary.json`

---

## 5. Directory Structure
```
speech_contrastive_dataset_project/
├── audio/                   # Complete audio for Speech 001–010 (7 variants each)
├── transcripts/             # Ground truth transcript text files
├── annotations/             # Word alignments & temporal annotations
├── features/                # Frame-level acoustic feature CSVs (pitch, energy, pauses, mfcc, fft)
├── metadata/                # Dataset manifest & speech metadata
├── models/                  # Vosk acoustic model for forced alignment
├── scripts/                 # Core scoring engine, alignment & feature extraction
│   ├── scoring_engine.py    # 4-dimension deterministic scoring engine
│   ├── core_analytics.py    # Audio analysis utilities
│   └── forced_alignment.py  # Acoustic alignment
├── dashboard/               # Interactive web dashboard
│   ├── app.py               # Lightweight HTTP server & evaluation REST API
│   └── index.html           # UI dashboard with visual timeline canvas
├── evaluation/              # Batch evaluation scripts & benchmark outputs
├── requirements.txt         # Python package dependencies
├── RUNNING_INSTRUCTIONS.md  # Step-by-step setup guide
└── README.md                # Full academic project documentation
```
