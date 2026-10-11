
import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import core_analytics as ca

def verify_single_audio(audio_path, transcript_path):
    if not os.path.exists(audio_path) or not os.path.exists(transcript_path):
        return
    data, sr = ca.load_audio_mono(audio_path)
    dur = len(data) / sr
    words, _ = ca.parse_transcript_words(transcript_path)
    wps = len(words) / dur if dur > 0 else 0
    print(f"[{os.path.basename(audio_path)}] Duration: {dur:.2f}s | Words: {len(words)} | Rate: {wps:.2f} wps ({wps*60:.1f} WPM) | PASS")

def main():
    parser = argparse.ArgumentParser(description="Transcribe and verify speech audio.")
    parser.add_argument("--audio", type=str, default=None, help="Path to single audio file")
    parser.add_argument("--speech_id", type=str, default=None, help="Target speech ID")
    parser.add_argument("--all", action="store_true", help="Batch verify all available speeches")
    parser.add_argument("--base_dir", type=str, default="speech-contrastive-dataset")
    args = parser.parse_args()

    audio_base = os.path.join(args.base_dir, "audio")
    trans_base = os.path.join(args.base_dir, "transcripts")

    if args.audio:
        sid = "speech_001"
        for d in os.listdir(audio_base):
            if d in args.audio: sid = d; break
        verify_single_audio(args.audio, os.path.join(trans_base, f"{sid}.txt"))
        return

    speech_dirs = sorted([d for d in os.listdir(audio_base) if d.startswith("speech_") and os.path.isdir(os.path.join(audio_base, d))])
    targets = [args.speech_id] if args.speech_id else speech_dirs

    for sid in targets:
        sdir = os.path.join(audio_base, sid)
        tpath = os.path.join(trans_base, f"{sid}.txt")
        audio_files = [f for f in os.listdir(sdir) if f.endswith(".wav") or f.endswith(".m4a")]
        if not audio_files:
            continue
        print(f"Verifying {sid} ({len(audio_files)} recordings):")
        for ver in ["good", "near_perfect", "mild_pacing", "severe_pacing", "flat_tone", "bad_pause", "low_volume"]:
            wav_path = os.path.join(sdir, f"{ver}.wav")
            m4a_path = os.path.join(sdir, f"{ver}.m4a")
            target_path = wav_path if os.path.exists(wav_path) else (m4a_path if os.path.exists(m4a_path) else None)
            if target_path:
                verify_single_audio(target_path, tpath)

if __name__ == "__main__":
    main()
