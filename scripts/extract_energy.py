
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
    parser = argparse.ArgumentParser(description="Extract RMS energy from audio files.")
    parser.add_argument("--speech_id", type=str, default=None, help="Target speech ID or None for all available")
    parser.add_argument("--all", action="store_true", help="Process all available speeches in batch")
    parser.add_argument("--audio_dir", type=str, default="speech-contrastive-dataset/audio")
    parser.add_argument("--output_csv", type=str, default="speech-contrastive-dataset/features/energy.csv")
    args = parser.parse_args()

    speech_dirs = [d for d in os.listdir(args.audio_dir) if d.startswith("speech_") and os.path.isdir(os.path.join(args.audio_dir, d))]
    speech_dirs = sorted(speech_dirs)

    if args.speech_id:
        targets = [args.speech_id]
    else:
        targets = speech_dirs

    all_rows = []
    for sid in targets:
        sdir = os.path.join(args.audio_dir, sid)
        audio_files = [f for f in os.listdir(sdir) if f.endswith(".wav") or f.endswith(".m4a")]
        if not audio_files:
            continue
        print(f"Extracting energy for {sid} ({len(audio_files)} files)...")
        for ver in ["good", "near_perfect", "mild_pacing", "severe_pacing", "flat_tone", "bad_pause", "low_volume"]:
            wav_path = os.path.join(sdir, f"{ver}.wav")
            m4a_path = os.path.join(sdir, f"{ver}.m4a")
            target_path = wav_path if os.path.exists(wav_path) else (m4a_path if os.path.exists(m4a_path) else None)
            if not target_path:
                continue
            data, sr = ca.load_audio_mono(target_path)
            rows, _ = ca.extract_energy_features(data, sr, sid, ver)
            all_rows.extend(rows)

    if all_rows:
        fieldnames = ["speech_id", "version", "timestamp", "rms_energy", "energy_normalized"]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {len(all_rows)} energy frames to {args.output_csv}")

if __name__ == "__main__":
    main()
