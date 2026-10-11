
import os
import sys
import re
import csv
import wave
import json
import difflib
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import core_analytics as ca
from vosk import Model, KaldiRecognizer

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "models", "vosk-model-small-en-us-0.15")

def get_vosk_model(model_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Vosk acoustic model not found at {model_path}. "
            "Please ensure the model is extracted under speech-contrastive-dataset/models/."
        )
    return Model(model_path)

def clean_word(w):
    return re.sub(r'[^\w]', '', w.lower())

def run_acoustic_alignment(audio_path, transcript_path, speech_id="speech_001", version="good", model=None):
    wav_path = ca.convert_audio_if_needed(audio_path, target_sr=16000)

    words, _ = ca.parse_transcript_words(transcript_path)
    clean_transcript_words = [clean_word(w) for w in words]

    if model is None:
        model = get_vosk_model()

    wf = wave.open(wav_path, "rb")
    sr = wf.getframerate()
    rec = KaldiRecognizer(model, sr)
    rec.SetWords(True)

    recognized_words = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            if "result" in res:
                recognized_words.extend(res["result"])

    final_res = json.loads(rec.FinalResult())
    if "result" in final_res:
        recognized_words.extend(final_res["result"])

    clean_rec_words = [clean_word(r["word"]) for r in recognized_words]
    matcher = difflib.SequenceMatcher(None, clean_transcript_words, clean_rec_words)
    aligned_output = []
    last_end = 0.0

    for idx, (orig_w, clean_w) in enumerate(zip(words, clean_transcript_words), start=1):
        start_t = None
        end_t = None
        conf = None

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal' and i1 <= (idx - 1) < i2:
                rec_idx = j1 + ((idx - 1) - i1)
                rec_item = recognized_words[rec_idx]
                start_t = rec_item["start"]
                end_t = rec_item["end"]
                conf = rec_item["conf"]
                last_end = end_t
                break

        # Fallback to neighboring boundary if unaligned in direct recognition
        if start_t is None:
            prev_matched = [item for item in aligned_output if item["end_time"] != ""]
            start_t = float(prev_matched[-1]["end_time"]) if prev_matched else last_end
            end_t = start_t + 0.25
            conf = 0.50

        duration_val = max(0.0, end_t - start_t)
        aligned_output.append({
            "speech_id": speech_id,
            "version": version,
            "word_index": idx,
            "word": orig_w,
            "start_time": f"{start_t:.3f}",
            "end_time": f"{end_t:.3f}",
            "duration": f"{duration_val:.3f}",
            "confidence": f"{conf:.2f}"
        })

    return aligned_output

def main():
    parser = argparse.ArgumentParser(description="Acoustic word-level forced alignment.")
    parser.add_argument("--audio", type=str, default="speech-contrastive-dataset/audio/speech_001/good.wav")
    parser.add_argument("--transcript", type=str, default="speech-contrastive-dataset/transcripts/speech_001.txt")
    parser.add_argument("--speech_id", type=str, default="speech_001")
    parser.add_argument("--version", type=str, default="good")
    parser.add_argument("--output_csv", type=str, default="speech-contrastive-dataset/annotations/per_speech/speech_001/word_alignment.csv")
    args = parser.parse_args()

    results = run_acoustic_alignment(
        audio_path=args.audio,
        transcript_path=args.transcript,
        speech_id=args.speech_id,
        version=args.version
    )

    fieldnames = ["speech_id", "version", "word_index", "word", "start_time", "end_time", "duration", "confidence"]
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Aligned {len(results)} words -> {args.output_csv}")

if __name__ == "__main__":
    main()
