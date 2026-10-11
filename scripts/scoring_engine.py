
import os
import sys
import csv
import json
import numpy as np

EPSILON = 1e-4

CATEGORY_MAPPING = {
    "pacing": "pacing",
    "pitch": "flat_tone",
    "fluency": "bad_pause",
    "volume": "low_volume"
}

REVERSE_CATEGORY_MAPPING = {
    "pacing": "pacing",
    "flat_tone": "pitch",
    "bad_pause": "fluency",
    "low_volume": "volume"
}

def load_csv_rows(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def calculate_iou(pred_interval, ref_interval):
    """
    Computes Intersection over Union between two 1D time intervals [start, end].
    """
    if not pred_interval and not ref_interval:
        return 1.0
    if not pred_interval or not ref_interval:
        return 0.0

    p_st, p_et = pred_interval
    r_st, r_et = ref_interval

    inter_st = max(p_st, r_st)
    inter_et = min(p_et, r_et)
    inter_dur = max(0.0, inter_et - inter_st)

    union_st = min(p_st, r_st)
    union_et = max(p_et, r_et)
    union_dur = max(0.0, union_et - union_st)

    if union_dur <= 0.0:
        return 1.0 if inter_dur == 0.0 else 0.0

    return inter_dur / union_dur

def get_severity_from_distortion(d):
    """
    Deterministic heuristic calibration for severity:
    0 = Normal / Clean (< 0.25)
    1 = Mild Deviation [0.25, 0.60)
    2 = Moderate Flaw  [0.60, 1.10)
    3 = Severe Flaw    (>= 1.10)
    """
    if d < 0.25:
        return 0
    elif d < 0.60:
        return 1
    elif d < 1.10:
        return 2
    else:
        return 3

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import core_analytics as ca

class SpeechEvaluator:
    def __init__(self, base_dir=None):
        if base_dir is None:
            if os.path.exists("annotations"):
                base_dir = "."
            elif os.path.exists("speech-contrastive-dataset/annotations"):
                base_dir = "speech-contrastive-dataset"
            else:
                base_dir = "."
        self.base_dir = base_dir
        self.audio_dir = os.path.join(base_dir, "audio")
        self.annotations_dir = os.path.join(base_dir, "annotations")
        self.features_dir = os.path.join(base_dir, "features")
        self.transcripts_dir = os.path.join(base_dir, "transcripts")

    def _get_audio_path(self, speech_id, version):
        s_dir = os.path.join(self.audio_dir, speech_id)
        if not os.path.isdir(s_dir):
            return None
        for f in os.listdir(s_dir):
            if (f.lower().startswith(version + ".") or f.lower() == version or f.lower().startswith(version + "_")) and (".m4a" in f.lower() or ".wav" in f.lower()):
                return os.path.join(s_dir, f)
        return None

    def _load_speech_data(self, speech_id):
        per_speech_feat = os.path.join(self.features_dir, "per_speech", speech_id)
        per_speech_ann = os.path.join(self.annotations_dir, "per_speech", speech_id)

        data = {
            "energy": load_csv_rows(os.path.join(per_speech_feat, "energy.csv")),
            "pitch": load_csv_rows(os.path.join(per_speech_feat, "pitch.csv")),
            "pauses": load_csv_rows(os.path.join(per_speech_feat, "pauses.csv")),
            "speech_rate": load_csv_rows(os.path.join(per_speech_feat, "speech_rate.csv")),
            "word_alignment": load_csv_rows(os.path.join(per_speech_ann, "word_alignment.csv")),
        }
        return data

    def evaluate(self, speech_id, version):
        """
        Universal evaluator that independently tests all 4 dimensions for any recording.
        Compares against good baseline without assuming which flaw is present.
        """
        speech_data = self._load_speech_data(speech_id)
        
        # 1. Filter version rows
        base_rate_rows = [r for r in speech_data["speech_rate"] if r["version"] == "good"]
        part_rate_rows = [r for r in speech_data["speech_rate"] if r["version"] == version]

        if not base_rate_rows or not part_rate_rows:
            raise ValueError(f"Missing speech rate data for {speech_id} (good or {version})")

        base_rate_row = base_rate_rows[0]
        part_rate_row = part_rate_rows[0]

        base_align = [r for r in speech_data["word_alignment"] if r["version"] == "good"]
        part_align = [r for r in speech_data["word_alignment"] if r["version"] == version]

        base_energy = [float(r["rms_energy"]) for r in speech_data["energy"] if r["version"] == "good"]
        part_energy = [float(r["rms_energy"]) for r in speech_data["energy"] if r["version"] == version]

        base_pitch_f0 = [float(r["f0_hz"]) for r in speech_data["pitch"] if r["version"] == "good" and r["f0_hz"] != ""]
        part_pitch_f0 = [float(r["f0_hz"]) for r in speech_data["pitch"] if r["version"] == version and r["f0_hz"] != ""]

        base_pauses = [
            (float(r["start_time"]), float(r["end_time"]), float(r["duration_seconds"]))
            for r in speech_data["pauses"] if r["version"] == "good"
        ]
        part_pauses = [
            (float(r["start_time"]), float(r["end_time"]), float(r["duration_seconds"]))
            for r in speech_data["pauses"] if r["version"] == version
        ]

        # ----------------- DIMENSION 1: PACING & CADENCE -----------------
        wps_base_global = float(base_rate_row["words_per_second"])
        wps_part_global = float(part_rate_row["words_per_second"])
        dur_base = float(base_rate_row["duration_seconds"])
        dur_part = float(part_rate_row["duration_seconds"])

        delta_wps_global = wps_part_global - wps_base_global
        delta_pct_global = (delta_wps_global / (wps_base_global + EPSILON)) * 100.0
        delta_rel_global = abs(delta_wps_global) / (wps_base_global + EPSILON)

        # Localized window rate analysis
        pacing_best_reg = None
        max_rate_divergence = 0.0

        if base_align and part_align and len(base_align) == len(part_align) and len(base_align) >= 5:
            win_len = min(10, len(base_align))
            for i in range(0, len(base_align) - win_len + 1, 2):
                b_w = base_align[i : i + win_len]
                p_w = part_align[i : i + win_len]
                b_st, b_et = float(b_w[0]["start_time"]), float(b_w[-1]["end_time"])
                p_st, p_et = float(p_w[0]["start_time"]), float(p_w[-1]["end_time"])
                b_dur = b_et - b_st
                p_dur = p_et - p_st

                if b_dur <= 0.15 or p_dur <= 0.15:
                    continue

                b_wps_loc = win_len / b_dur
                p_wps_loc = win_len / p_dur
                loc_delta = p_wps_loc - b_wps_loc
                loc_div = abs(loc_delta) / (b_wps_loc + EPSILON)

                if loc_div > max_rate_divergence:
                    max_rate_divergence = loc_div
                    phrase = " ".join(w["word"] for w in p_w)
                    pacing_best_reg = (p_st, p_et, p_dur, b_wps_loc, p_wps_loc, loc_delta, phrase)

        d_pacing = 0.40 * delta_rel_global + 0.60 * min(2.5, max_rate_divergence)
        score_pacing = max(0.0, 25.0 * max(0.0, 1.0 - (d_pacing / 2.0)))
        sev_pacing = get_severity_from_distortion(d_pacing)

        pacing_flaw = None
        if sev_pacing >= 1:
            if pacing_best_reg and max_rate_divergence >= 0.25:
                p_st, p_et, p_dur, b_rate, p_rate, r_diff, phrase = pacing_best_reg
                dir_word = "accelerated / rushed" if r_diff > 0 else "dragged / prolonged"
                diff_pct = (r_diff / (b_rate + EPSILON)) * 100.0
                explanation = (
                    f"Speaking rate {dir_word} to {p_rate:.2f} wps ({diff_pct:+.1f}% vs baseline {b_rate:.2f} wps) "
                    f"over interval [{p_st:.2f}s - {p_et:.2f}s] ({p_dur:.2f}s) across phrase: \"{phrase}\"."
                )
                pacing_flaw = {
                    "flaw_type": "pacing",
                    "severity": sev_pacing,
                    "start_time": p_st,
                    "end_time": p_et,
                    "duration": p_dur,
                    "affected_text": phrase,
                    "baseline_value": b_rate,
                    "participant_value": p_rate,
                    "delta": r_diff,
                    "delta_percentage": diff_pct,
                    "causal_explanation": explanation
                }
            else:
                dir_word = "accelerated / rushed" if delta_wps_global > 0 else "dragged / prolonged"
                explanation = (
                    f"Global speaking rate {dir_word} by {delta_pct_global:+.1f}% "
                    f"({wps_part_global:.2f} wps vs baseline {wps_base_global:.2f} wps) across entire passage."
                )
                pacing_flaw = {
                    "flaw_type": "pacing",
                    "severity": sev_pacing,
                    "start_time": 0.0,
                    "end_time": dur_part,
                    "duration": dur_part,
                    "affected_text": "Entire passage delivery",
                    "baseline_value": wps_base_global,
                    "participant_value": wps_part_global,
                    "delta": delta_wps_global,
                    "delta_percentage": delta_pct_global,
                    "causal_explanation": explanation
                }

        # ----------------- DIMENSION 2: PITCH DYNAMICS -----------------
        f0_base_std = float(np.std(base_pitch_f0)) if len(base_pitch_f0) > 1 else 1.0
        f0_part_std = float(np.std(part_pitch_f0)) if len(part_pitch_f0) > 1 else 1.0

        iqr_base = float(np.percentile(base_pitch_f0, 75) - np.percentile(base_pitch_f0, 25)) if len(base_pitch_f0) > 4 else f0_base_std
        iqr_part = float(np.percentile(part_pitch_f0, 75) - np.percentile(part_pitch_f0, 25)) if len(part_pitch_f0) > 4 else f0_part_std

        c_std = max(0.0, (f0_base_std - f0_part_std) / (f0_base_std + EPSILON))
        c_range = max(0.0, (iqr_base - iqr_part) / (iqr_base + EPSILON))

        # Localized window pitch collapse
        pitch_best_reg = None
        max_pitch_collapse = 0.0

        part_audio_path = self._get_audio_path(speech_id, version)
        base_audio_path = self._get_audio_path(speech_id, "good")
        part_audio_data = None
        base_audio_data = None
        sr = 16000

        if part_audio_path and base_audio_path:
            try:
                part_audio_data, sr = ca.load_audio_mono(part_audio_path, target_sr=16000)
                base_audio_data, _ = ca.load_audio_mono(base_audio_path, target_sr=16000)
            except Exception:
                part_audio_data = None
                base_audio_data = None

        if base_align and part_align and len(base_align) == len(part_align) and len(base_align) >= 5:
            win_len = min(10, len(part_align))
            for i in range(0, len(part_align) - win_len + 1, 2):
                b_w = base_align[i : i + win_len]
                p_w = part_align[i : i + win_len]
                b_st, b_et = float(b_w[0]["start_time"]), float(b_w[-1]["end_time"])
                p_st, p_et = float(p_w[0]["start_time"]), float(p_w[-1]["end_time"])

                if base_audio_data is not None and part_audio_data is not None:
                    sub_b = base_audio_data[int(b_st * sr) : int(b_et * sr)]
                    sub_p = part_audio_data[int(p_st * sr) : int(p_et * sr)]
                    if len(sub_b) >= 1024 and len(sub_p) >= 1024:
                        _, b_voiced = ca.extract_pitch_features(sub_b, sr, speech_id, "base")
                        _, p_voiced = ca.extract_pitch_features(sub_p, sr, speech_id, "part")
                        if len(b_voiced) >= 4 and len(p_voiced) >= 4:
                            b_sub_std = float(np.std(b_voiced))
                            p_sub_std = float(np.std(p_voiced))
                            loc_collapse = max(0.0, (b_sub_std - p_sub_std) / (b_sub_std + EPSILON))
                            if loc_collapse > max_pitch_collapse:
                                max_pitch_collapse = loc_collapse
                                phrase = " ".join(w["word"] for w in p_w)
                                pitch_best_reg = (p_st, p_et, p_et - p_st, b_sub_std, p_sub_std, p_sub_std - b_sub_std, phrase)
                else:
                    b_sub = [float(r["f0_hz"]) for r in speech_data["pitch"]
                             if r["version"] == "good" and r["f0_hz"] != "" and b_st <= float(r["timestamp"]) <= b_et]
                    p_sub = [float(r["f0_hz"]) for r in speech_data["pitch"]
                             if r["version"] == version and r["f0_hz"] != "" and p_st <= float(r["timestamp"]) <= p_et]
                    if len(b_sub) >= 4 and len(p_sub) >= 4:
                        b_sub_std = float(np.std(b_sub))
                        p_sub_std = float(np.std(p_sub))
                        loc_collapse = max(0.0, (b_sub_std - p_sub_std) / (b_sub_std + EPSILON))
                        if loc_collapse > max_pitch_collapse:
                            max_pitch_collapse = loc_collapse
                            phrase = " ".join(w["word"] for w in p_w)
                            pitch_best_reg = (p_st, p_et, p_et - p_st, b_sub_std, p_sub_std, p_sub_std - b_sub_std, phrase)

        d_pitch = 0.35 * c_std + 0.25 * c_range + 0.40 * max_pitch_collapse
        score_pitch = max(0.0, 25.0 * max(0.0, 1.0 - min(1.0, 1.25 * d_pitch)))
        sev_pitch = get_severity_from_distortion(d_pitch)

        pitch_flaw = None
        if sev_pitch >= 1:
            if pitch_best_reg and max_pitch_collapse >= 0.25:
                p_st, p_et, p_dur, b_s, p_s, delta_s, phrase = pitch_best_reg
                diff_pct = (delta_s / (b_s + EPSILON)) * 100.0
                explanation = (
                    f"Pitch intonation standard deviation collapsed by {abs(diff_pct):.1f}% "
                    f"({p_s:.2f} Hz vs {b_s:.2f} Hz baseline) over interval [{p_st:.2f}s - {p_et:.2f}s] ({p_dur:.2f}s) "
                    f"across phrase: \"{phrase}\", producing localized monotone delivery."
                )
                pitch_flaw = {
                    "flaw_type": "pitch",
                    "severity": sev_pitch,
                    "start_time": p_st,
                    "end_time": p_et,
                    "duration": p_dur,
                    "affected_text": phrase,
                    "baseline_value": b_s,
                    "participant_value": p_s,
                    "delta": delta_s,
                    "delta_percentage": diff_pct,
                    "causal_explanation": explanation
                }
            else:
                delta_global_s = f0_part_std - f0_base_std
                diff_pct = (delta_global_s / (f0_base_std + EPSILON)) * 100.0
                explanation = (
                    f"Pitch intonation standard deviation dropped by {abs(diff_pct):.1f}% "
                    f"({f0_part_std:.2f} Hz vs {f0_base_std:.2f} Hz baseline) across entire passage."
                )
                pitch_flaw = {
                    "flaw_type": "pitch",
                    "severity": sev_pitch,
                    "start_time": 0.0,
                    "end_time": dur_part,
                    "duration": dur_part,
                    "affected_text": "Entire passage delivery (monotone cadence)",
                    "baseline_value": f0_base_std,
                    "participant_value": f0_part_std,
                    "delta": delta_global_s,
                    "delta_percentage": diff_pct,
                    "causal_explanation": explanation
                }

        # ----------------- DIMENSION 3: VOLUME & AUDIBILITY -----------------
        mean_rms_base = float(np.mean(base_energy)) if base_energy else 0.03
        mean_rms_part = float(np.mean(part_energy)) if part_energy else 0.03

        delta_rms_rel = abs(mean_rms_part - mean_rms_base) / (mean_rms_base + EPSILON)

        # Localized window energy dips
        volume_best_reg = None
        min_loc_rms = 1e9
        max_energy_dip = 0.0

        if part_align and len(part_align) >= 5:
            win_len = min(8, len(part_align))
            for i in range(0, len(part_align) - win_len + 1):
                p_w = part_align[i : i + win_len]
                p_st = float(p_w[0]["start_time"])
                p_et = float(p_w[-1]["end_time"])

                if part_audio_data is not None:
                    sub_samples = part_audio_data[int(p_st * sr) : int(p_et * sr)]
                    if len(sub_samples) > 256:
                        p_sub_rms = float(np.sqrt(np.mean(sub_samples**2)))
                        loc_dip = max(0.0, (mean_rms_base - p_sub_rms) / (mean_rms_base + EPSILON))
                        if loc_dip > max_energy_dip:
                            max_energy_dip = loc_dip
                        if p_sub_rms < min_loc_rms:
                            min_loc_rms = p_sub_rms
                            phrase = " ".join(w["word"] for w in p_w)
                            volume_best_reg = (p_st, p_et, p_et - p_st, mean_rms_base, p_sub_rms, p_sub_rms - mean_rms_base, phrase)
                else:
                    p_sub = [float(r["rms_energy"]) for r in speech_data["energy"]
                             if r["version"] == version and p_st <= float(r["timestamp"]) <= p_et]
                    if p_sub:
                        p_sub_rms = float(np.mean(p_sub))
                        loc_dip = max(0.0, (mean_rms_base - p_sub_rms) / (mean_rms_base + EPSILON))
                        if loc_dip > max_energy_dip:
                            max_energy_dip = loc_dip
                        if p_sub_rms < min_loc_rms:
                            min_loc_rms = p_sub_rms
                            phrase = " ".join(w["word"] for w in p_w)
                            volume_best_reg = (p_st, p_et, p_et - p_st, mean_rms_base, p_sub_rms, p_sub_rms - mean_rms_base, phrase)

        d_volume = 0.40 * min(1.5, delta_rms_rel) + 0.60 * min(1.5, max_energy_dip)
        score_volume = max(0.0, 25.0 * max(0.0, 1.0 - (d_volume / 1.25)))
        sev_volume = get_severity_from_distortion(d_volume)

        volume_flaw = None
        if sev_volume >= 1:
            if volume_best_reg and max_energy_dip >= 0.25:
                p_st, p_et, p_dur, b_rms, p_rms, delta_rms, phrase = volume_best_reg
                diff_pct = (delta_rms / (b_rms + EPSILON)) * 100.0
                explanation = (
                    f"Localized vocal energy RMS dropped by {abs(diff_pct):.1f}% "
                    f"({p_rms:.4f} RMS vs {b_rms:.4f} RMS baseline) over interval [{p_st:.2f}s - {p_et:.2f}s] ({p_dur:.2f}s) "
                    f"across phrase: \"{phrase}\", causing diminished vocal audibility."
                )
                volume_flaw = {
                    "flaw_type": "volume",
                    "severity": sev_volume,
                    "start_time": p_st,
                    "end_time": p_et,
                    "duration": p_dur,
                    "affected_text": phrase,
                    "baseline_value": b_rms,
                    "participant_value": p_rms,
                    "delta": delta_rms,
                    "delta_percentage": diff_pct,
                    "causal_explanation": explanation
                }
            else:
                delta_global_rms = mean_rms_part - mean_rms_base
                diff_pct = (delta_global_rms / (mean_rms_base + EPSILON)) * 100.0
                dir_word = "dropped" if delta_global_rms < 0 else "elevated"
                explanation = (
                    f"Mean vocal energy RMS {dir_word} by {abs(diff_pct):.1f}% "
                    f"({mean_rms_part:.4f} RMS vs {mean_rms_base:.4f} RMS baseline) across entire passage."
                )
                volume_flaw = {
                    "flaw_type": "volume",
                    "severity": sev_volume,
                    "start_time": 0.0,
                    "end_time": dur_part,
                    "duration": dur_part,
                    "affected_text": "Entire passage delivery (volume shift)",
                    "baseline_value": mean_rms_base,
                    "participant_value": mean_rms_part,
                    "delta": delta_global_rms,
                    "delta_percentage": diff_pct,
                    "causal_explanation": explanation
                }

        # ----------------- DIMENSION 4: FLUENCY & PAUSES -----------------
        base_pause_durs = [p[2] for p in base_pauses]
        part_pause_durs = [p[2] for p in part_pauses]

        base_avg_pause = float(np.mean(base_pause_durs)) if base_pause_durs else 0.45
        base_max_pause = max(base_pause_durs) if base_pause_durs else 0.60
        part_max_pause = max(part_pause_durs) if part_pause_durs else 0.0

        p_denom = max(0.40, base_avg_pause)
        allowed_max = max(1.0, 1.40 * base_max_pause)
        d_p_max = max(0.0, (part_max_pause - allowed_max) / p_denom)

        rate_pause_base = (len(base_pauses) / (dur_base + EPSILON)) * 60.0
        rate_pause_part = (len(part_pauses) / (dur_part + EPSILON)) * 60.0
        d_p_freq = abs(rate_pause_part - rate_pause_base) / (rate_pause_base + EPSILON)

        duty_base = sum(base_pause_durs) / (dur_base + EPSILON)
        duty_part = sum(part_pause_durs) / (dur_part + EPSILON)
        d_p_duty = abs(duty_part - duty_base) / (duty_base + EPSILON)

        d_fluency = 0.60 * min(2.5, d_p_max) + 0.20 * min(1.5, d_p_freq) + 0.20 * min(1.5, d_p_duty)
        score_fluency = max(0.0, 25.0 * max(0.0, 1.0 - (d_fluency / 1.8)))
        sev_fluency = get_severity_from_distortion(d_fluency)

        fluency_flaw = None
        if sev_fluency >= 1 and part_pauses:
            longest_pause = max(part_pauses, key=lambda p: p[2])
            p_st, p_et, p_dur = longest_pause
            delta_dur = p_dur - base_avg_pause
            diff_pct = (delta_dur / (base_avg_pause + EPSILON)) * 100.0

            affected_text = f"Acoustic silence gap at [{p_st:.2f}s - {p_et:.2f}s]"
            if part_align:
                prev_words = [w for w in part_align if float(w["end_time"]) <= p_st + 0.20]
                next_words = [w for w in part_align if float(w["start_time"]) >= p_et - 0.20]
                prev_w = prev_words[-1]["word"] if prev_words else ""
                next_w = next_words[0]["word"] if next_words else ""
                if prev_w and next_w:
                    affected_text = f"Unnatural silence gap between \"{prev_w}\" and \"{next_w}\""

            explanation = (
                f"Longest silence interval reached {p_dur:.2f}s (+{diff_pct:.1f}% longer than baseline average pause of {base_avg_pause:.2f}s) "
                f"between {p_st:.2f}s and {p_et:.2f}s ({affected_text}), breaking communicative continuity."
            )
            fluency_flaw = {
                "flaw_type": "fluency",
                "severity": sev_fluency,
                "start_time": p_st,
                "end_time": p_et,
                "duration": p_dur,
                "affected_text": affected_text,
                "baseline_value": base_avg_pause,
                "participant_value": p_dur,
                "delta": delta_dur,
                "delta_percentage": diff_pct,
                "causal_explanation": explanation
            }

        # ----------------- OVERALL SCORE & MULTI-FLAW AGGREGATION -----------------
        overall_score = score_pacing + score_pitch + score_volume + score_fluency

        detected_flaws = []
        if pacing_flaw: detected_flaws.append(pacing_flaw)
        if pitch_flaw: detected_flaws.append(pitch_flaw)
        if volume_flaw: detected_flaws.append(volume_flaw)
        if fluency_flaw: detected_flaws.append(fluency_flaw)

        return {
            "speech_id": speech_id,
            "version": version,
            "overall_score": overall_score,
            "dimension_scores": {
                "pacing": score_pacing,
                "pitch": score_pitch,
                "volume": score_volume,
                "fluency": score_fluency
            },
            "severities": {
                "pacing": sev_pacing,
                "pitch": sev_pitch,
                "volume": sev_volume,
                "fluency": sev_fluency
            },
            "detected_flaws": detected_flaws
        }

    def evaluate_with_iou(self, speech_id, version):
        """
        Evaluates recording and compares independently predicted flaw intervals
        against the reference annotations in annotations/temporal_annotations.csv.
        Reference annotations are ONLY read here for IoU comparison.
        """
        eval_result = self.evaluate(speech_id, version)

        # Load reference annotations for comparison
        ref_path = os.path.join(self.annotations_dir, "temporal_annotations.csv")
        ref_rows = load_csv_rows(ref_path)
        matching_refs = [r for r in ref_rows if r["speech_id"] == speech_id and r["version"] == version]

        # In temporal_annotations.csv, flaw_type is one of: pacing, pitch, pause, volume
        TYPE_SYNONYMS = {
            "pacing": ["pacing"],
            "pitch": ["pitch", "flat_tone"],
            "fluency": ["pause", "bad_pause"],
            "volume": ["volume", "low_volume"]
        }

        # Reference ground-truth category for this version (if any)
        ref_flaw = matching_refs[0] if matching_refs else None
        target_iou = 1.0 if not ref_flaw else 0.0

        flaw_ious = []
        for pred_flaw in eval_result["detected_flaws"]:
            p_type = pred_flaw["flaw_type"]
            p_interval = (pred_flaw["start_time"], pred_flaw["end_time"])

            # Check if this predicted flaw matches the reference flaw
            matched_ref = None
            if ref_flaw:
                allowed_types = TYPE_SYNONYMS.get(p_type, [p_type])
                if ref_flaw["flaw_type"] in allowed_types or ref_flaw["version"] in allowed_types:
                    matched_ref = ref_flaw

            if matched_ref:
                r_interval = (float(matched_ref["start_time"]), float(matched_ref["end_time"]))
                iou = calculate_iou(p_interval, r_interval)
                target_iou = iou
                flaw_ious.append({
                    "flaw_type": p_type,
                    "ref_flaw_type": matched_ref["flaw_type"],
                    "predicted_interval": p_interval,
                    "reference_interval": r_interval,
                    "iou": iou
                })
            else:
                flaw_ious.append({
                    "flaw_type": p_type,
                    "ref_flaw_type": None,
                    "predicted_interval": p_interval,
                    "reference_interval": None,
                    "iou": None
                })

        eval_result["temporal_iou"] = {
            "target_flaw_iou": target_iou,
            "mean_iou": target_iou,
            "flaw_ious": flaw_ious
        }
        return eval_result

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deterministic speech scoring and evaluation engine.")
    parser.add_argument("--speech_id", type=str, default="speech_001")
    parser.add_argument("--version", type=str, default="mild_pacing")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    evaluator = SpeechEvaluator()

    if not args.all:
        res = evaluator.evaluate_with_iou(args.speech_id, args.version)
        print(f"\n=======================================================")
        print(f"EVALUATION RESULT: {res['speech_id']} ({res['version']})")
        print(f"=======================================================")
        print(f"Overall Score:        {res['overall_score']:.2f} / 100")
        print(f"  Pacing & Cadence:   {res['dimension_scores']['pacing']:.2f} / 25 (Sev {res['severities']['pacing']})")
        print(f"  Pitch Dynamics:     {res['dimension_scores']['pitch']:.2f} / 25 (Sev {res['severities']['pitch']})")
        print(f"  Volume & Audib:     {res['dimension_scores']['volume']:.2f} / 25 (Sev {res['severities']['volume']})")
        print(f"  Fluency & Pauses:   {res['dimension_scores']['fluency']:.2f} / 25 (Sev {res['severities']['fluency']})")
        print(f"\nTemporal IoU:         {res['temporal_iou']['mean_iou']:.3f}")
        print(f"\nDetected Flaws ({len(res['detected_flaws'])}):")
        for f in res['detected_flaws']:
            print(f"  - [{f['flaw_type'].upper()} Sev {f['severity']}] [{f['start_time']:.2f}s - {f['end_time']:.2f}s]: {f['causal_explanation']}")
        print(f"=======================================================\n")

if __name__ == "__main__":
    main()
