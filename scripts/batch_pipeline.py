
import os
import sys
import csv
import json
import argparse
import traceback
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import core_analytics as ca
import forced_alignment as fa

TARGET_VARIANTS = [
    "good", "near_perfect", "mild_pacing", "severe_pacing",
    "flat_tone", "bad_pause", "low_volume"
]

def find_audio_files(speech_dir):
    found_map = {}
    if not os.path.isdir(speech_dir):
        return found_map

    files = sorted(os.listdir(speech_dir))
    for ver in TARGET_VARIANTS:
        matches = [
            f for f in files
            if (f.lower().startswith(ver + ".") or f.lower() == ver or f.lower().startswith(ver + "_"))
            and (".m4a" in f.lower() or ".wav" in f.lower())
        ]
        if matches:
            wav_match = next((m for m in matches if m.lower().endswith(".wav")), None)
            selected = wav_match if wav_match else matches[0]
            found_map[ver] = os.path.join(speech_dir, selected)

    return found_map

def write_csv(filepath, fieldnames, rows):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def append_or_replace_master_csv(master_path, fieldnames, new_rows, speech_id):
    existing_rows = []
    if os.path.exists(master_path):
        with open(master_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("speech_id") != speech_id:
                    existing_rows.append(r)

    write_csv(master_path, fieldnames, existing_rows + new_rows)

def process_speech(speech_id, base_dir, errors_log, model=None):
    audio_dir = os.path.join(base_dir, "audio", speech_id)
    transcript_path = os.path.join(base_dir, "transcripts", f"{speech_id}.txt")

    if not os.path.exists(audio_dir):
        errors_log.append({"speech_id": speech_id, "error": f"Audio directory not found: {audio_dir}"})
        return None

    if not os.path.exists(transcript_path):
        errors_log.append({"speech_id": speech_id, "error": f"Transcript not found: {transcript_path}"})
        return None

    try:
        words, raw_text = ca.parse_transcript_words(transcript_path)
    except Exception as e:
        errors_log.append({"speech_id": speech_id, "error": f"Error parsing transcript {transcript_path}: {str(e)}"})
        return None

    audio_map = find_audio_files(audio_dir)
    if not audio_map:
        errors_log.append({"speech_id": speech_id, "error": f"No audio files discovered in {audio_dir}"})
        return None

    if "good" not in audio_map:
        errors_log.append({"speech_id": speech_id, "error": f"Missing baseline 'good' audio file in {audio_dir}"})
        return None

    print(f"[{speech_id}] {len(audio_map)}/7 variants, {len(words)} transcript words.")

    speech_energy = []
    speech_pitch = []
    speech_pauses = []
    speech_fft = []
    speech_mfcc = []
    speech_rate = []
    speech_word_alignment = []
    variant_metrics = {}
    variant_alignments = {}
    variant_audio_data = {}

    for ver in TARGET_VARIANTS:
        if ver not in audio_map:
            errors_log.append({"speech_id": speech_id, "version": ver, "error": f"Variant '{ver}' missing from {speech_id}"})
            continue

        audio_file = audio_map[ver]
        try:
            data, sr = ca.load_audio_mono(audio_file)
            variant_audio_data[ver] = data
            dur = len(data) / sr

            align_rows = fa.run_acoustic_alignment(audio_file, transcript_path, speech_id, ver, model=model)
            speech_word_alignment.extend(align_rows)
            variant_alignments[ver] = align_rows

            e_rows, rms_arr = ca.extract_energy_features(data, sr, speech_id, ver)
            speech_energy.extend(e_rows)
            mean_rms = float(np.mean(rms_arr)) if len(rms_arr) > 0 else 0.0

            p_rows, voiced_f0 = ca.extract_pitch_features(data, sr, speech_id, ver)
            speech_pitch.extend(p_rows)
            f0_std = float(np.std(voiced_f0)) if voiced_f0 else 0.0

            pa_rows, pauses = ca.detect_pause_features(data, sr, speech_id, ver)
            speech_pauses.extend(pa_rows)

            f_rows = ca.extract_fft_features(data, sr, speech_id, ver)
            speech_fft.extend(f_rows)

            m_rows = ca.extract_mfcc_features(data, sr, speech_id, ver)
            speech_mfcc.extend(m_rows)

            sr_rows = ca.extract_speech_rate_features(data, sr, len(words), pauses, speech_id, ver)
            speech_rate.extend(sr_rows)

            wps_global = len(words) / dur if dur > 0 else 0.0

            variant_metrics[ver] = {
                "duration": dur,
                "mean_rms": mean_rms,
                "f0_std": f0_std,
                "pauses": pauses,
                "wps_global": wps_global,
                "audio_file": f"audio/{speech_id}/{os.path.basename(audio_file)}"
            }

        except Exception as e:
            tb = traceback.format_exc()
            errors_log.append({
                "speech_id": speech_id,
                "version": ver,
                "file": audio_file,
                "error": str(e),
                "traceback": tb
            })
            print(f"  [ERROR] {speech_id}/{ver} ({audio_file}): {str(e)}")

    base_meta = variant_metrics.get("good")
    speech_temporal = []
    speech_labels = []

    if base_meta:
        speech_labels.append({
            "speech_id": speech_id,
            "version": "good",
            "audio_file": base_meta["audio_file"],
            "flaw_type": "none",
            "severity": 0,
            "description": "Baseline reference delivery",
            "target_feature": "all"
        })

        flaw_specs = [
            ("near_perfect", "pacing", 1, "speech_rate"),
            ("mild_pacing", "pacing", 2, "speech_rate"),
            ("severe_pacing", "pacing", 3, "speech_rate"),
            ("flat_tone", "pitch", 2, "pitch"),
            ("bad_pause", "pause", 3, "pause_duration"),
            ("low_volume", "volume", 2, "energy")
        ]

        for ver, default_flaw, default_sev, default_feat in flaw_specs:
            if ver in variant_metrics:
                part_meta = variant_metrics[ver]
                flaw = ca.contrastive_flaw_analysis(
                    speech_id, ver, base_meta, part_meta,
                    base_align=variant_alignments.get("good"),
                    part_align=variant_alignments.get(ver),
                    base_data=variant_audio_data.get("good"),
                    part_data=variant_audio_data.get(ver),
                    sr=16000
                )
                if flaw:
                    speech_temporal.append(flaw)
                    speech_labels.append({
                        "speech_id": speech_id,
                        "version": ver,
                        "audio_file": part_meta["audio_file"],
                        "flaw_type": flaw["flaw_type"],
                        "severity": flaw["severity"],
                        "description": flaw["causal_explanation"],
                        "target_feature": default_feat
                    })

    per_speech_feat = os.path.join(base_dir, "features", "per_speech", speech_id)
    per_speech_ann = os.path.join(base_dir, "annotations", "per_speech", speech_id)

    write_csv(os.path.join(per_speech_ann, "word_alignment.csv"),
              ["speech_id", "version", "word_index", "word", "start_time", "end_time", "duration", "confidence"], speech_word_alignment)
    write_csv(os.path.join(per_speech_feat, "energy.csv"),
              ["speech_id", "version", "timestamp", "rms_energy", "energy_normalized"], speech_energy)
    write_csv(os.path.join(per_speech_feat, "pitch.csv"),
              ["speech_id", "version", "timestamp", "f0_hz", "pitch_normalized"], speech_pitch)
    write_csv(os.path.join(per_speech_feat, "pauses.csv"),
              ["speech_id", "version", "start_time", "end_time", "duration_seconds", "pause_type"], speech_pauses)
    write_csv(os.path.join(per_speech_feat, "fft.csv"),
              ["speech_id", "version", "timestamp", "frequency_hz", "magnitude"], speech_fft)
    write_csv(os.path.join(per_speech_feat, "mfcc.csv"),
              ["speech_id", "version", "timestamp", "frame", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13"], speech_mfcc)
    write_csv(os.path.join(per_speech_feat, "speech_rate.csv"),
              ["speech_id", "version", "start_time", "end_time", "word_count", "duration_seconds", "words_per_second"], speech_rate)
    write_csv(os.path.join(per_speech_ann, "temporal_annotations.csv"),
              ["speech_id", "version", "flaw_id", "flaw_type", "severity", "start_time", "end_time", "duration", "affected_text", "baseline_value", "participant_value", "delta", "delta_percentage", "causal_explanation"], speech_temporal)
    write_csv(os.path.join(per_speech_ann, "labels.csv"),
              ["speech_id", "version", "audio_file", "flaw_type", "severity", "description", "target_feature"], speech_labels)

    return {
        "word_alignment": speech_word_alignment,
        "energy": speech_energy,
        "pitch": speech_pitch,
        "pauses": speech_pauses,
        "fft": speech_fft,
        "mfcc": speech_mfcc,
        "speech_rate": speech_rate,
        "temporal": speech_temporal,
        "labels": speech_labels,
        "variants_processed": len(variant_metrics)
    }

def run_pipeline(base_dir="speech-contrastive-dataset", target_speech=None):
    audio_base = os.path.join(base_dir, "audio")
    speech_dirs = sorted([d for d in os.listdir(audio_base) if d.startswith("speech_") and os.path.isdir(os.path.join(audio_base, d))])

    if target_speech:
        speech_dirs = [d for d in speech_dirs if d == target_speech]
        if not speech_dirs:
            print(f"Speech ID '{target_speech}' not found in {audio_base}")
            sys.exit(1)

    print(f"Running pipeline on {len(speech_dirs)} speeches: {', '.join(speech_dirs)}")

    model = fa.get_vosk_model()

    errors_log = []
    processed_speeches = []

    master_word_alignment = []
    master_energy = []
    master_pitch = []
    master_pauses = []
    master_fft = []
    master_mfcc = []
    master_speech_rate = []
    master_temporal = []
    master_labels = []

    for sid in speech_dirs:
        result = process_speech(sid, base_dir, errors_log, model=model)
        if result:
            master_word_alignment.extend(result["word_alignment"])
            master_energy.extend(result["energy"])
            master_pitch.extend(result["pitch"])
            master_pauses.extend(result["pauses"])
            master_fft.extend(result["fft"])
            master_mfcc.extend(result["mfcc"])
            master_speech_rate.extend(result["speech_rate"])
            master_temporal.extend(result["temporal"])
            master_labels.extend(result["labels"])
            processed_speeches.append(sid)

    feat_dir = os.path.join(base_dir, "features")
    ann_dir = os.path.join(base_dir, "annotations")

    write_csv(os.path.join(ann_dir, "word_alignment.csv"),
              ["speech_id", "version", "word_index", "word", "start_time", "end_time", "duration", "confidence"], master_word_alignment)
    write_csv(os.path.join(feat_dir, "energy.csv"),
              ["speech_id", "version", "timestamp", "rms_energy", "energy_normalized"], master_energy)
    write_csv(os.path.join(feat_dir, "pitch.csv"),
              ["speech_id", "version", "timestamp", "f0_hz", "pitch_normalized"], master_pitch)
    write_csv(os.path.join(feat_dir, "pauses.csv"),
              ["speech_id", "version", "start_time", "end_time", "duration_seconds", "pause_type"], master_pauses)
    write_csv(os.path.join(feat_dir, "fft.csv"),
              ["speech_id", "version", "timestamp", "frequency_hz", "magnitude"], master_fft)
    write_csv(os.path.join(feat_dir, "mfcc.csv"),
              ["speech_id", "version", "timestamp", "frame", "mfcc_1", "mfcc_2", "mfcc_3", "mfcc_4", "mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8", "mfcc_9", "mfcc_10", "mfcc_11", "mfcc_12", "mfcc_13"], master_mfcc)
    write_csv(os.path.join(feat_dir, "speech_rate.csv"),
              ["speech_id", "version", "start_time", "end_time", "word_count", "duration_seconds", "words_per_second"], master_speech_rate)
    write_csv(os.path.join(ann_dir, "temporal_annotations.csv"),
              ["speech_id", "version", "flaw_id", "flaw_type", "severity", "start_time", "end_time", "duration", "affected_text", "baseline_value", "participant_value", "delta", "delta_percentage", "causal_explanation"], master_temporal)
    write_csv(os.path.join(ann_dir, "labels.csv"),
              ["speech_id", "version", "audio_file", "flaw_type", "severity", "description", "target_feature"], master_labels)

    log_path = os.path.join(base_dir, "metadata", "pipeline_execution_log.json")
    status_summary = {
        "status": "completed",
        "total_speeches": len(speech_dirs),
        "speeches_successfully_processed": processed_speeches,
        "total_audio_recordings_analyzed": len(master_labels),
        "total_grounded_flaws": len(master_temporal),
        "errors_count": len(errors_log),
        "errors_details": errors_log
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(status_summary, f, indent=2)

    print(f"Completed {len(processed_speeches)}/{len(speech_dirs)} speeches ({len(master_labels)} audio files, {len(master_temporal)} flaws).")

def main():
    parser = argparse.ArgumentParser(description="Batch pipeline for contrastive speech dataset.")
    parser.add_argument("--base_dir", type=str, default="speech-contrastive-dataset")
    parser.add_argument("--speech_id", type=str, default=None, help="Target specific speech ID (e.g. speech_002)")
    args = parser.parse_args()

    run_pipeline(base_dir=args.base_dir, target_speech=args.speech_id)

if __name__ == "__main__":
    main()
