
import argparse
import os
import sys
import csv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import core_analytics as ca

def main():
    parser = argparse.ArgumentParser(description="Extract speech rate metrics.")
    parser.add_argument("--speech_id", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--audio_dir", type=str, default="speech-contrastive-dataset/audio")
    parser.add_argument("--alignment_csv", type=str, default="speech-contrastive-dataset/annotations/word_alignment.csv")
    parser.add_argument("--output_csv", type=str, default="speech-contrastive-dataset/features/speech_rate.csv")
    args = parser.parse_args()

    speech_dirs = sorted([d for d in os.listdir(args.audio_dir) if d.startswith("speech_") and os.path.isdir(os.path.join(args.audio_dir, d))])
    targets = [args.speech_id] if args.speech_id else speech_dirs

    alignments_map = {}
    if os.path.exists(args.alignment_csv):
        with open(args.alignment_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                key = (r["speech_id"], r["version"])
                if key not in alignments_map:
                    alignments_map[key] = []
                alignments_map[key].append(r)

    all_rows = []
    for sid in targets:
        sdir = os.path.join(args.audio_dir, sid)
        audio_files = [f for f in os.listdir(sdir) if f.endswith(".wav") or f.endswith(".m4a")]
        if not audio_files:
            continue
        print(f"Computing speech rate for {sid}...")
        for ver in ["good", "near_perfect", "mild_pacing", "severe_pacing", "flat_tone", "bad_pause", "low_volume"]:
            wav_path = os.path.join(sdir, f"{ver}.wav")
            m4a_path = os.path.join(sdir, f"{ver}.m4a")
            target_path = wav_path if os.path.exists(wav_path) else (m4a_path if os.path.exists(m4a_path) else None)
            if not target_path:
                continue
            data, sr = ca.load_audio_mono(target_path)
            aligns = alignments_map.get((sid, ver), [])
            rows = ca.extract_speech_rate_features(data, sr, aligns, sid, ver)
            all_rows.extend(rows)

    if all_rows:
        fieldnames = ["speech_id", "version", "start_time", "end_time", "word_count", "duration_seconds", "words_per_second"]
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved {len(all_rows)} speech rate records to {args.output_csv}")

if __name__ == "__main__":
    main()
