
import os
import re
import subprocess
import soundfile as sf
import numpy as np

def convert_audio_if_needed(input_path, output_dir=None, target_sr=16000):
    input_dir = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)
    clean_name = re.sub(r'(\.m4a)+$', '', base_name, flags=re.IGNORECASE)
    
    out_dir = output_dir if output_dir else input_dir
    output_path = os.path.join(out_dir, f"{clean_name}.wav")

    if os.path.exists(output_path) and os.path.getmtime(output_path) >= os.path.getmtime(input_path):
        return output_path

    if os.path.exists("/usr/bin/afconvert"):
        cmd = ["/usr/bin/afconvert", "-f", "WAVE", "-d", f"LEI16@{target_sr}", "-c", "1", input_path, output_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            return output_path

    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", str(target_sr), "-ac", "1", output_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        return output_path

    raise RuntimeError(f"Failed to convert {input_path} to WAV. Ensure /usr/bin/afconvert or ffmpeg is available.")

def load_audio_mono(audio_path, target_sr=16000):
    wav_path = audio_path
    if audio_path.lower().endswith(".m4a") or ".m4a" in audio_path.lower():
        wav_path = convert_audio_if_needed(audio_path, target_sr=target_sr)
    
    data, sr = sf.read(wav_path)
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr

def parse_transcript_words(transcript_path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        content = f.read()
    raw_text = content
    if "Transcript:" in content:
        raw_text = content.split("Transcript:", 1)[1].strip()
    words = re.findall(r"\b[\w'-]+\b", raw_text)
    return words, raw_text

def extract_energy_features(data, sr, speech_id, version, frame_length=2048, hop_length=512):
    n_frames = (len(data) - frame_length) // hop_length + 1
    rms_vals = [np.sqrt(np.mean(data[i*hop_length : i*hop_length + frame_length]**2)) for i in range(n_frames)]
    times = [(i * hop_length) / sr for i in range(n_frames)]
    rms_arr = np.array(rms_vals)
    max_rms = np.max(rms_arr) if len(rms_arr) > 0 and np.max(rms_arr) > 0 else 1.0
    norm_vals = rms_arr / max_rms

    rows = []
    for t, rms, norm in zip(times, rms_arr, norm_vals):
        rows.append({
            "speech_id": speech_id,
            "version": version,
            "timestamp": f"{t:.3f}",
            "rms_energy": f"{rms:.6f}",
            "energy_normalized": f"{norm:.6f}"
        })
    return rows, rms_arr

def extract_pitch_features(data, sr, speech_id, version, frame_length=1024, hop_length=256, fmin=65, fmax=400):
    """
    Extracts fundamental frequency (F0) pitch contour via normalized autocorrelation.
    Search lags are constrained between sr // fmax and sr // fmin.
    """
    min_lag = int(sr / fmax)
    max_lag = int(sr / fmin)
    n_frames = (len(data) - frame_length) // hop_length + 1

    f0_values = []
    times = []

    for i in range(n_frames):
        frame = data[i * hop_length : i * hop_length + frame_length]
        t = (i * hop_length) / sr
        times.append(t)

        rms = np.sqrt(np.mean(frame**2))
        if rms < 0.006:
            f0_values.append(None)
            continue

        frame_zm = frame - np.mean(frame)
        corr = np.correlate(frame_zm, frame_zm, mode='full')[len(frame_zm)-1:]
        search = corr[min_lag:max_lag]
        if len(search) == 0 or np.max(search) <= 0:
            f0_values.append(None)
            continue

        best_lag = min_lag + np.argmax(search)
        coeff = corr[best_lag] / corr[0] if corr[0] > 0 else 0

        if coeff > 0.35:
            # Sub-sample parabolic interpolation
            if 0 < best_lag < len(corr) - 1:
                a, b, c = corr[best_lag-1], corr[best_lag], corr[best_lag+1]
                denom = 2 * (a - 2*b + c)
                delta = (a - c) / denom if denom != 0 else 0
                ref_lag = best_lag + delta
            else:
                ref_lag = best_lag
            f0 = sr / ref_lag
            if fmin <= f0 <= fmax:
                f0_values.append(f0)
            else:
                f0_values.append(None)
        else:
            f0_values.append(None)

    voiced = [v for v in f0_values if v is not None]
    median_f0 = np.median(voiced) if voiced else 1.0

    rows = []
    for t, f0 in zip(times, f0_values):
        rows.append({
            "speech_id": speech_id,
            "version": version,
            "timestamp": f"{t:.3f}",
            "f0_hz": f"{f0:.2f}" if f0 is not None else "",
            "pitch_normalized": f"{(f0 / median_f0):.4f}" if f0 is not None else ""
        })
    return rows, voiced

def detect_pause_features(data, sr, speech_id, version, frame_length=1024, hop_length=256, min_pause_sec=0.3):
    """
    Detects real silence intervals based on RMS energy drop below background threshold.
    """
    dur_total = len(data) / sr
    n_frames = (len(data) - frame_length) // hop_length + 1
    rms = np.array([np.sqrt(np.mean(data[i*hop_length : i*hop_length + frame_length]**2)) for i in range(n_frames)])
    times = np.arange(n_frames) * (hop_length / sr)

    thresh = max(0.005, 0.15 * np.percentile(rms, 75))
    is_silent = rms < thresh

    pauses = []
    in_pause = False
    start_t = 0.0

    for t, silent in zip(times, is_silent):
        if silent and not in_pause:
            in_pause = True
            start_t = t
        elif not silent and in_pause:
            in_pause = False
            duration = t - start_t
            if duration >= min_pause_sec:
                pauses.append((start_t, t, duration))

    if in_pause and (times[-1] - start_t >= min_pause_sec):
        pauses.append((start_t, times[-1], times[-1] - start_t))

    rows = []
    for st, et, dur in pauses:
        if st <= 0.05:
            ptype = "initial_silence"
        elif et >= dur_total - 0.5:
            ptype = "terminal_silence"
        elif dur > 1.20:
            ptype = "unnatural_pause"
        elif dur > 0.80:
            ptype = "extended_hesitation"
        else:
            ptype = "syntactic_pause"

        rows.append({
            "speech_id": speech_id,
            "version": version,
            "start_time": f"{st:.3f}",
            "end_time": f"{et:.3f}",
            "duration_seconds": f"{dur:.3f}",
            "pause_type": ptype
        })
    return rows, pauses

def extract_fft_features(data, sr, speech_id, version, frame_length=2048, hop_length=4000):
    """
    Computes representative FFT magnitude spectra across frequency bands.
    """
    n_frames = (len(data) - frame_length) // hop_length + 1
    freqs = np.fft.rfftfreq(frame_length, d=1.0/sr)
    target_freqs = [125, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    target_indices = [np.argmin(np.abs(freqs - tf)) for tf in target_freqs]
    window = np.hanning(frame_length)

    rows = []
    for i in range(n_frames):
        frame = data[i * hop_length : i * hop_length + frame_length] * window
        t = (i * hop_length) / sr
        fft_vals = np.abs(np.fft.rfft(frame))
        mag_norm = fft_vals / (frame_length / 2)

        for idx in target_indices:
            rows.append({
                "speech_id": speech_id,
                "version": version,
                "timestamp": f"{t:.3f}",
                "frequency_hz": f"{freqs[idx]:.1f}",
                "magnitude": f"{mag_norm[idx]:.6f}"
            })
    return rows

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel):
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def get_mel_filterbank(sr, n_fft, n_mels=26):
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sr / 2), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fbank = np.zeros((n_mels, int(n_fft // 2 + 1)))
    for m in range(1, n_mels + 1):
        f_prev, f_curr, f_next = bin_points[m-1], bin_points[m], bin_points[m+1]
        for k in range(f_prev, f_curr):
            if f_curr != f_prev: fbank[m-1, k] = (k - f_prev) / (f_curr - f_prev)
        for k in range(f_curr, f_next):
            if f_next != f_curr: fbank[m-1, k] = (f_next - k) / (f_next - f_curr)
    return fbank

def dct_ii(x, n_coeffs=13):
    M = len(x)
    coeffs = np.zeros(n_coeffs)
    for n in range(n_coeffs):
        factor = np.sqrt(1.0 / M) if n == 0 else np.sqrt(2.0 / M)
        coeffs[n] = factor * np.sum(x * np.cos(np.pi * n * (np.arange(M) + 0.5) / M))
    return coeffs

def extract_mfcc_features(data, sr, speech_id, version, frame_length=1024, hop_length=512, n_mfcc=13):
    """
    Extracts 13 Mel-Frequency Cepstral Coefficients (MFCCs).
    """
    n_frames = (len(data) - frame_length) // hop_length + 1
    fbank = get_mel_filterbank(sr, frame_length, n_mels=26)
    window = np.hanning(frame_length)

    rows = []
    for i in range(n_frames):
        frame = data[i * hop_length : i * hop_length + frame_length] * window
        t = (i * hop_length) / sr
        mag_spec = np.abs(np.fft.rfft(frame))
        pow_spec = (mag_spec ** 2) / frame_length
        mel_energies = np.dot(fbank, pow_spec)
        mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
        log_mel = np.log(mel_energies)
        mfccs = dct_ii(log_mel, n_coeffs=n_mfcc + 1)[1:]

        row = {
            "speech_id": speech_id,
            "version": version,
            "timestamp": f"{t:.3f}",
            "frame": i
        }
        for idx_c, val in enumerate(mfccs, start=1):
            row[f"mfcc_{idx_c}"] = f"{val:.4f}"
        rows.append(row)
    return rows

def extract_speech_rate_features(data, sr, word_count, pauses, speech_id, version):
    total_dur = len(data) / sr
    if total_dur <= 0 or word_count <= 0:
        return []

    total_pause_dur = sum(p[2] for p in pauses)
    active_speech_dur = max(0.1, total_dur - total_pause_dur)

    wps_global = word_count / total_dur
    wps_vocalized = word_count / active_speech_dur

    rows = [
        {
            "speech_id": speech_id,
            "version": version,
            "start_time": "0.000",
            "end_time": f"{total_dur:.3f}",
            "word_count": word_count,
            "duration_seconds": f"{total_dur:.3f}",
            "words_per_second": f"{wps_global:.2f}"
        },
        {
            "speech_id": speech_id,
            "version": f"{version}_vocalized_active",
            "start_time": "0.000",
            "end_time": f"{active_speech_dur:.3f}",
            "word_count": word_count,
            "duration_seconds": f"{active_speech_dur:.3f}",
            "words_per_second": f"{wps_vocalized:.2f}"
        }
    ]
    return rows

def contrastive_flaw_analysis(speech_id, participant_version, base_meta, part_meta,
                              base_align=None, part_align=None, base_data=None, part_data=None, sr=16000):
    base_dur = base_meta["duration"]
    part_dur = part_meta["duration"]
    base_rms = base_meta["mean_rms"]
    part_rms = part_meta["mean_rms"]
    base_wps = base_meta["wps_global"]
    part_wps = part_meta["wps_global"]
    base_f0_std = base_meta["f0_std"]
    part_f0_std = part_meta["f0_std"]
    base_pauses = base_meta["pauses"]
    part_pauses = part_meta["pauses"]

    if participant_version in ("near_perfect", "mild_pacing", "severe_pacing"):
        flaw_type = "pacing"
        severity = 1 if participant_version == "near_perfect" else (2 if participant_version == "mild_pacing" else 3)
        flaw_id = f"{speech_id}_flaw_{'01' if severity == 1 else ('02' if severity == 2 else '03')}_{participant_version}"

        if base_align and part_align and len(base_align) == len(part_align) and len(base_align) >= 5:
            win_len = min(10, len(base_align))
            best_score = -1e9
            best_reg = None

            for i in range(0, len(base_align) - win_len + 1, 2):
                b_w = base_align[i : i + win_len]
                v_w = part_align[i : i + win_len]
                b_st = float(b_w[0]["start_time"])
                b_et = float(b_w[-1]["end_time"])
                v_st = float(v_w[0]["start_time"])
                v_et = float(v_w[-1]["end_time"])
                b_d = b_et - b_st
                v_d = v_et - v_st

                if b_d <= 0.2 or v_d <= 0.2:
                    continue

                b_rate = win_len / b_d
                v_rate = win_len / v_d
                rate_diff = v_rate - b_rate

                if participant_version == "near_perfect":
                    score = abs(rate_diff)
                else:
                    score = rate_diff

                if score > best_score:
                    best_score = score
                    phrase = " ".join(w["word"] for w in v_w)
                    best_reg = (v_st, v_et, v_d, b_rate, v_rate, rate_diff, phrase)

            if best_reg:
                v_st, v_et, v_d, b_rate, v_rate, rate_diff, phrase = best_reg
                diff_pct = (rate_diff / b_rate) * 100 if b_rate > 0 else 0
                explanation = (
                    f"Speaking rate {'diverged' if severity == 1 else ('accelerated' if severity == 2 else 'severely accelerated')} "
                    f"to {v_rate:.2f} wps ({diff_pct:+.1f}% vs baseline {b_rate:.2f} wps) "
                    f"over interval [{v_st:.2f}s - {v_et:.2f}s] ({v_d:.2f}s) across phrase: \"{phrase}\"."
                )
                return {
                    "speech_id": speech_id,
                    "version": participant_version,
                    "flaw_id": flaw_id,
                    "flaw_type": flaw_type,
                    "severity": severity,
                    "start_time": f"{v_st:.3f}",
                    "end_time": f"{v_et:.3f}",
                    "duration": f"{v_d:.3f}",
                    "affected_text": phrase,
                    "baseline_value": f"{b_rate:.2f}",
                    "participant_value": f"{v_rate:.2f}",
                    "delta": f"{rate_diff:+.2f}",
                    "delta_percentage": f"{diff_pct:+.1f}%",
                    "causal_explanation": explanation
                }

        delta_wps = part_wps - base_wps
        delta_pct = (delta_wps / base_wps) * 100 if base_wps > 0 else 0
        return {
            "speech_id": speech_id,
            "version": participant_version,
            "flaw_id": flaw_id,
            "flaw_type": flaw_type,
            "severity": severity,
            "start_time": "0.000",
            "end_time": f"{part_dur:.3f}",
            "duration": f"{part_dur:.3f}",
            "affected_text": "Entire passage delivery",
            "baseline_value": f"{base_wps:.2f}",
            "participant_value": f"{part_wps:.2f}",
            "delta": f"{delta_wps:+.2f}",
            "delta_percentage": f"{delta_pct:+.1f}%",
            "causal_explanation": f"Speaking rate shifted by {delta_pct:+.1f}% ({part_wps:.2f} vs {base_wps:.2f} wps baseline)."
        }

    elif participant_version == "flat_tone":
        flaw_type = "pitch"
        severity = 2
        flaw_id = f"{speech_id}_flaw_04_flat_tone"

        if base_align and part_align and base_data is not None and part_data is not None and len(part_align) >= 5:
            win_len = min(10, len(part_align))
            best_diff = -1e9
            best_reg = None

            for i in range(0, len(part_align) - win_len + 1, 2):
                b_words = base_align[i : i + win_len]
                f_words = part_align[i : i + win_len]
                b_st, b_et = float(b_words[0]["start_time"]), float(b_words[-1]["end_time"])
                f_st, f_et = float(f_words[0]["start_time"]), float(f_words[-1]["end_time"])

                sub_b = base_data[int(b_st * sr) : int(b_et * sr)]
                sub_f = part_data[int(f_st * sr) : int(f_et * sr)]

                if len(sub_b) >= 1024 and len(sub_f) >= 1024:
                    _, b_voiced = extract_pitch_features(sub_b, sr, speech_id, "base")
                    _, f_voiced = extract_pitch_features(sub_f, sr, speech_id, "flat")
                    if len(b_voiced) >= 4 and len(f_voiced) >= 4:
                        b_std = float(np.std(b_voiced))
                        f_std = float(np.std(f_voiced))
                        diff = b_std - f_std
                        if diff > best_diff:
                            best_diff = diff
                            phrase = " ".join(w["word"] for w in f_words)
                            best_reg = (f_st, f_et, f_et - f_st, b_std, f_std, diff, phrase)

            if best_reg:
                f_st, f_et, dur, b_std, f_std, diff, phrase = best_reg
                delta_val = f_std - b_std
                delta_pct = (delta_val / b_std) * 100 if b_std > 0 else 0
                explanation = (
                    f"Pitch intonation standard deviation collapsed by {abs(delta_pct):.1f}% "
                    f"({f_std:.2f} Hz vs {b_std:.2f} Hz baseline) over interval [{f_st:.2f}s - {f_et:.2f}s] ({dur:.2f}s) "
                    f"across phrase: \"{phrase}\", producing localized monotone delivery."
                )
                return {
                    "speech_id": speech_id,
                    "version": participant_version,
                    "flaw_id": flaw_id,
                    "flaw_type": flaw_type,
                    "severity": severity,
                    "start_time": f"{f_st:.3f}",
                    "end_time": f"{f_et:.3f}",
                    "duration": f"{dur:.3f}",
                    "affected_text": phrase,
                    "baseline_value": f"{b_std:.2f}",
                    "participant_value": f"{f_std:.2f}",
                    "delta": f"{delta_val:+.2f}",
                    "delta_percentage": f"{delta_pct:+.1f}%",
                    "causal_explanation": explanation
                }

        delta_f0 = part_f0_std - base_f0_std
        delta_pct = (delta_f0 / base_f0_std) * 100 if base_f0_std > 0 else 0
        return {
            "speech_id": speech_id,
            "version": participant_version,
            "flaw_id": flaw_id,
            "flaw_type": flaw_type,
            "severity": severity,
            "start_time": "0.000",
            "end_time": f"{part_dur:.3f}",
            "duration": f"{part_dur:.3f}",
            "affected_text": "Entire passage delivery (monotone cadence)",
            "baseline_value": f"{base_f0_std:.2f}",
            "participant_value": f"{part_f0_std:.2f}",
            "delta": f"{delta_f0:+.2f}",
            "delta_percentage": f"{delta_pct:+.1f}%",
            "causal_explanation": f"Pitch intonation standard deviation dropped by {delta_pct:.1f}% ({part_f0_std:.1f} Hz vs {base_f0_std:.1f} Hz baseline)."
        }

    elif participant_version == "bad_pause":
        flaw_type = "pause"
        severity = 3
        flaw_id = f"{speech_id}_flaw_05_bad_pause"

        longest_part_pause = max(part_pauses, key=lambda p: p[2]) if part_pauses else (0.0, part_dur, 1.5)
        st, et, dur = longest_part_pause
        base_avg_pause = float(np.mean([p[2] for p in base_pauses])) if base_pauses else 0.50
        delta = dur - base_avg_pause
        delta_pct = (delta / base_avg_pause) * 100 if base_avg_pause > 0 else 0

        affected_text = f"Acoustic silence gap at [{st:.2f}s - {et:.2f}s]"
        if part_align:
            prev_words = [w for w in part_align if float(w["end_time"]) <= st + 0.15]
            next_words = [w for w in part_align if float(w["start_time"]) >= et - 0.15]
            prev_w = prev_words[-1]["word"] if prev_words else ""
            next_w = next_words[0]["word"] if next_words else ""
            if prev_w and next_w:
                affected_text = f"Unnatural silence gap between \"{prev_w}\" and \"{next_w}\""

        explanation = (
            f"Longest silence interval reached {dur:.2f}s (+{delta_pct:.1f}% longer than baseline average pause of {base_avg_pause:.2f}s) "
            f"between {st:.2f}s and {et:.2f}s ({affected_text}), breaking communicative continuity."
        )
        return {
            "speech_id": speech_id,
            "version": participant_version,
            "flaw_id": flaw_id,
            "flaw_type": flaw_type,
            "severity": severity,
            "start_time": f"{st:.3f}",
            "end_time": f"{et:.3f}",
            "duration": f"{dur:.3f}",
            "affected_text": affected_text,
            "baseline_value": f"{base_avg_pause:.2f}",
            "participant_value": f"{dur:.2f}",
            "delta": f"{delta:+.2f}",
            "delta_percentage": f"{delta_pct:+.1f}%",
            "causal_explanation": explanation
        }

    elif participant_version == "low_volume":
        flaw_type = "volume"
        severity = 2
        flaw_id = f"{speech_id}_flaw_06_low_volume"

        if part_align and part_data is not None and len(part_align) >= 5:
            win_len = min(8, len(part_align))
            min_rms = 1e9
            best_reg = None

            for i in range(0, len(part_align) - win_len + 1):
                sub_w = part_align[i : i + win_len]
                st = float(sub_w[0]["start_time"])
                et = float(sub_w[-1]["end_time"])
                sub_samples = part_data[int(st * sr) : int(et * sr)]
                if len(sub_samples) > 256:
                    sub_rms = float(np.sqrt(np.mean(sub_samples**2)))
                    if sub_rms < min_rms:
                        min_rms = sub_rms
                        phrase = " ".join(w["word"] for w in sub_w)
                        best_reg = (st, et, et - st, sub_rms, phrase)

            if best_reg:
                st, et, dur, loc_rms, phrase = best_reg
                delta_val = loc_rms - base_rms
                delta_pct = (delta_val / base_rms) * 100 if base_rms > 0 else 0
                explanation = (
                    f"Localized vocal energy RMS dropped by {abs(delta_pct):.1f}% "
                    f"({loc_rms:.4f} RMS vs {base_rms:.4f} RMS baseline) over interval [{st:.2f}s - {et:.2f}s] ({dur:.2f}s) "
                    f"across phrase: \"{phrase}\", causing diminished vocal audibility."
                )
                return {
                    "speech_id": speech_id,
                    "version": participant_version,
                    "flaw_id": flaw_id,
                    "flaw_type": flaw_type,
                    "severity": severity,
                    "start_time": f"{st:.3f}",
                    "end_time": f"{et:.3f}",
                    "duration": f"{dur:.3f}",
                    "affected_text": phrase,
                    "baseline_value": f"{base_rms:.4f}",
                    "participant_value": f"{loc_rms:.4f}",
                    "delta": f"{delta_val:+.4f}",
                    "delta_percentage": f"{delta_pct:+.1f}%",
                    "causal_explanation": explanation
                }

        delta_rms = part_rms - base_rms
        delta_pct = (delta_rms / base_rms) * 100 if base_rms > 0 else 0
        return {
            "speech_id": speech_id,
            "version": participant_version,
            "flaw_id": flaw_id,
            "flaw_type": flaw_type,
            "severity": severity,
            "start_time": "0.000",
            "end_time": f"{part_dur:.3f}",
            "duration": f"{part_dur:.3f}",
            "affected_text": "Entire passage delivery (reduced volume)",
            "baseline_value": f"{base_rms:.4f}",
            "participant_value": f"{part_rms:.4f}",
            "delta": f"{delta_rms:+.4f}",
            "delta_percentage": f"{delta_pct:+.1f}%",
            "causal_explanation": f"Mean vocal energy RMS decreased by {delta_pct:.1f}% ({part_rms:.4f} RMS vs {base_rms:.4f} RMS baseline)."
        }

    return None
