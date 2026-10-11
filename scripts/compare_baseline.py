
import argparse
import os
import sys
import csv
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import core_analytics as ca

def main():
    parser = argparse.ArgumentParser(description="Contrastive speech analytics comparison.")
    parser.add_argument("--speech_id", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--audio_dir", type=str, default="speech-contrastive-dataset/audio")
    parser.add_argument("--transcripts_dir", type=str, default="speech-contrastive-dataset/transcripts")
    parser.add_argument("--annotations_csv", type=str, default="speech-contrastive-dataset/annotations/temporal_annotations.csv")
    args = parser.parse_args()

    speech_dirs = sorted([d for d in os.listdir(args.audio_dir) if d.startswith("speech_") and os.path.isdir(os.path.join(args.audio_dir, d))])
    targets = [args.speech_id] if args.speech_id else speech_dirs

    existing_rows = []
    if os.path.exists(args.annotations_csv):
        with open(args.annotations_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                existing_rows.append(r)

    flaw_updates = {}
    flawed_versions = ["near_perfect", "mild_pacing", "severe_pacing", "flat_tone", "bad_pause", "low_volume"]

    for sid in targets:
        sdir = os.path.join(args.audio_dir, sid)
        tpath = os.path.join(args.transcripts_dir, f"{sid}.txt")
        audio_files = [f for f in os.listdir(sdir) if f.endswith(".wav") or f.endswith(".m4a")]
        if not audio_files or not os.path.exists(tpath):
            continue

        base_wav = os.path.join(sdir, "good.wav")
        base_m4a = os.path.join(sdir, "good.m4a")
        base_target = base_wav if os.path.exists(base_wav) else (base_m4a if os.path.exists(base_m4a) else None)
        if not base_target:
            continue

        words, _ = ca.parse_transcript_words(tpath)
        base_data, sr = ca.load_audio_mono(base_target)
        _, base_f0 = ca.extract_pitch_features(base_data, sr, sid, "good")
        _, base_pauses = ca.detect_pause_features(base_data, sr, sid, "good")

        base_meta = {
            "duration": len(base_data) / sr,
            "mean_rms": float(np.sqrt(np.mean(base_data**2))),
            "f0_std": float(np.std(base_f0)) if base_f0 else 0.0,
            "pauses": base_pauses,
            "wps_global": len(words) / (len(base_data)/sr)
        }

        flaw_updates[sid] = {}
        for ver in flawed_versions:
            p_wav = os.path.join(sdir, f"{ver}.wav")
            p_m4a = os.path.join(sdir, f"{ver}.m4a")
            p_target = p_wav if os.path.exists(p_wav) else (p_m4a if os.path.exists(p_m4a) else None)
            if not p_target:
                continue

            p_data, _ = ca.load_audio_mono(p_target)
            _, p_f0 = ca.extract_pitch_features(p_data, sr, sid, ver)
            _, p_pauses = ca.detect_pause_features(p_data, sr, sid, ver)

            p_meta = {
                "duration": len(p_data) / sr,
                "mean_rms": float(np.sqrt(np.mean(p_data**2))),
                "f0_std": float(np.std(p_f0)) if p_f0 else 0.0,
                "pauses": p_pauses,
                "wps_global": len(words) / (len(p_data)/sr)
            }

            res = ca.contrastive_flaw_analysis(sid, ver, base_meta, p_meta)
            if res:
                flaw_updates[sid][ver] = res

    for r in existing_rows:
        sid = r["speech_id"]
        ver = r["version"]
        if sid in flaw_updates and ver in flaw_updates[sid]:
            up = flaw_updates[sid][ver]
            for k, v in up.items():
                r[k] = v

    fieldnames = [
        "speech_id", "version", "flaw_id", "flaw_type", "severity",
        "start_time", "end_time", "duration", "affected_text",
        "baseline_value", "participant_value", "delta", "delta_percentage",
        "causal_explanation"
    ]
    with open(args.annotations_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)

    print("Updated temporal_annotations.csv with grounded flaw intervals.")

if __name__ == "__main__":
    main()
