# Interactive Speech Evaluation Dashboard

Lightweight, browser-based dashboard for exploring contrastive speech analytics, 0–100 delivery scores, and grounded flaw intervals.

---

## Quick Start

Launch the local dashboard server from the dataset root:

```bash
python dashboard/app.py
```

Then open your browser to:
[http://localhost:8080](http://localhost:8080)

To specify a custom port:
```bash
python dashboard/app.py --port 8888
```

---

## Features

1. **Speech Selector**: Switch between `speech_001` through `speech_010`.
2. **Variant Selector**: Select any of the 7 versions (`good`, `near_perfect`, `mild_pacing`, `severe_pacing`, `flat_tone`, `bad_pause`, `low_volume`).
3. **Delivery Score (0–100)**: Overall score computed deterministically from the 4 acoustic dimensions:
   - **Pacing & Cadence** (0–25)
   - **Pitch Dynamics** (0–25)
   - **Volume & Audibility** (0–25)
   - **Fluency & Pauses** (0–25)
4. **Universal Multi-Flaw Cards**: Displays each detected flaw independently, including:
   - Category (`PACING`, `PITCH`, `VOLUME`, `FLUENCY`)
   - Severity level (`0` to `3`)
   - Timestamp interval $[t_{\text{start}}, t_{\text{end}}]$
   - Affected transcript phrase
   - Metric delta ($\Delta$ and $\%\Delta$)
   - Causal coaching diagnosis
   - Target Temporal IoU vs. ground-truth reference
5. **Interactive Audio & Timeline**:
   - Native HTML5 audio playback.
   - Synchronized visual canvas timeline showing the full recording track with color-coded flaw intervals.
   - Click "Play Segment" on any flaw card to seek directly to the flawed passage.
6. **Transcript View**:
   - Canonical passage text with detected flaw intervals highlighted.
