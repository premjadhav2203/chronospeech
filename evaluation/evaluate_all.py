
import os
import sys
import csv
import json
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from scoring_engine import SpeechEvaluator

VARIANTS = [
    "good", "near_perfect", "mild_pacing", "severe_pacing",
    "flat_tone", "bad_pause", "low_volume"
]

def run_evaluation_benchmark(base_dir=None):
    if base_dir is None:
        if os.path.exists("annotations"):
            base_dir = "."
        elif os.path.exists("speech-contrastive-dataset/annotations"):
            base_dir = "speech-contrastive-dataset"
        else:
            base_dir = "."

    evaluator = SpeechEvaluator(base_dir=base_dir)
    audio_base = os.path.join(base_dir, "audio")
    speech_ids = sorted([d for d in os.listdir(audio_base) if d.startswith("speech_") and os.path.isdir(os.path.join(audio_base, d))])

    eval_results = []
    summary_rows = []

    print(f"Evaluating {len(speech_ids)} speeches across 7 variants ({len(speech_ids) * len(VARIANTS)} recordings)...")

    for sid in speech_ids:
        for ver in VARIANTS:
            try:
                res = evaluator.evaluate_with_iou(sid, ver)
                eval_results.append(res)

                # Format tabular row
                flaws_str = "; ".join([f"{f['flaw_type']}(Sev{f['severity']})[{f['start_time']:.2f}-{f['end_time']:.2f}s]" for f in res["detected_flaws"]])
                summary_rows.append({
                    "speech_id": sid,
                    "version": ver,
                    "overall_score": round(res["overall_score"], 2),
                    "pacing_score": round(res["dimension_scores"]["pacing"], 2),
                    "pitch_score": round(res["dimension_scores"]["pitch"], 2),
                    "volume_score": round(res["dimension_scores"]["volume"], 2),
                    "fluency_score": round(res["dimension_scores"]["fluency"], 2),
                    "pacing_severity": res["severities"]["pacing"],
                    "pitch_severity": res["severities"]["pitch"],
                    "volume_severity": res["severities"]["volume"],
                    "fluency_severity": res["severities"]["fluency"],
                    "flaws_count": len(res["detected_flaws"]),
                    "temporal_iou": round(res["temporal_iou"]["target_flaw_iou"], 3) if res["temporal_iou"]["target_flaw_iou"] is not None else "",
                    "detected_flaws_summary": flaws_str
                })
            except Exception as e:
                print(f"  [ERROR] {sid} {ver}: {str(e)}")

    # Save exports
    out_dir = os.path.join(base_dir, "evaluation")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "evaluation_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    csv_path = os.path.join(out_dir, "evaluation_results.csv")
    fieldnames = [
        "speech_id", "version", "overall_score",
        "pacing_score", "pitch_score", "volume_score", "fluency_score",
        "pacing_severity", "pitch_severity", "volume_severity", "fluency_severity",
        "flaws_count", "temporal_iou", "detected_flaws_summary"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Compute statistics
    scores = [r["overall_score"] for r in eval_results]
    variant_scores = {v: [] for v in VARIANTS}
    for r in eval_results:
        variant_scores[r["version"]].append(r["overall_score"])

    # Severity distribution
    sev_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in eval_results:
        for dim, s in r["severities"].items():
            sev_counts[s] += 1

    # IoU statistics (only for flawed variants where a reference ground-truth flaw exists)
    flawed_ious = [r["temporal_iou"]["target_flaw_iou"] for r in eval_results if r["version"] != "good"]
    mean_flawed_iou = float(np.mean(flawed_ious)) if flawed_ious else 0.0

    iou_by_type = {
        "pacing": [],
        "pitch": [],
        "fluency": [],
        "volume": []
    }
    for r in eval_results:
        if r["version"] in ("near_perfect", "mild_pacing", "severe_pacing"):
            iou_by_type["pacing"].append(r["temporal_iou"]["target_flaw_iou"])
        elif r["version"] == "flat_tone":
            iou_by_type["pitch"].append(r["temporal_iou"]["target_flaw_iou"])
        elif r["version"] == "bad_pause":
            iou_by_type["fluency"].append(r["temporal_iou"]["target_flaw_iou"])
        elif r["version"] == "low_volume":
            iou_by_type["volume"].append(r["temporal_iou"]["target_flaw_iou"])

    stats_summary = {
        "total_recordings_evaluated": len(eval_results),
        "overall_score_mean": round(float(np.mean(scores)), 2),
        "overall_score_median": round(float(np.median(scores)), 2),
        "overall_score_std": round(float(np.std(scores)), 2),
        "overall_score_min": round(float(np.min(scores)), 2),
        "overall_score_max": round(float(np.max(scores)), 2),
        "variant_mean_scores": {v: round(float(np.mean(variant_scores[v])), 2) for v in VARIANTS},
        "severity_distribution_total": sev_counts,
        "mean_flawed_temporal_iou": round(mean_flawed_iou, 3),
        "iou_by_type": {k: round(float(np.mean(v)), 3) for k, v in iou_by_type.items() if v}
    }

    stats_path = os.path.join(out_dir, "benchmark_summary.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_summary, f, indent=2)

    print("\n=======================================================")
    print("DATASET EVALUATION BENCHMARK SUMMARY")
    print("=======================================================")
    print(f"Total Audio Recordings Evaluated: {len(eval_results)} / 70")
    print(f"Overall Score Mean:                {stats_summary['overall_score_mean']} / 100")
    print(f"Overall Score Range:               {stats_summary['overall_score_min']} - {stats_summary['overall_score_max']}")
    print(f"\nMean Score by Variant:")
    for v in VARIANTS:
        print(f"  - {v:<16}: {stats_summary['variant_mean_scores'][v]:.2f} / 100")
    print(f"\nSeverity Distribution across All Dimensions:")
    print(f"  - Severity 0 (Normal):   {sev_counts[0]}")
    print(f"  - Severity 1 (Mild):     {sev_counts[1]}")
    print(f"  - Severity 2 (Moderate): {sev_counts[2]}")
    print(f"  - Severity 3 (Severe):   {sev_counts[3]}")
    print(f"\nTemporal IoU by Flaw Type:")
    for k, v in stats_summary["iou_by_type"].items():
        print(f"  - {k:<10}: {v:.3f}")
    print(f"Mean Flawed Temporal IoU:          {stats_summary['mean_flawed_temporal_iou']:.3f}")
    print(f"\nSaved structured results to:")
    print(f"  - {json_path}")
    print(f"  - {csv_path}")
    print(f"  - {stats_path}")
    print("=======================================================\n")
    return stats_summary

if __name__ == "__main__":
    run_evaluation_benchmark()
