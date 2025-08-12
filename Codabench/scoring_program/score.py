# pylint: disable=logging-fstring-interpolation
"""Scoring for Speech Restoration with stems:
- SI-SNR (dB): higher is better
- FAD-CLAP: lower is better

Folder structure (both reference and prediction):
<root>/<stem>/*.wav
where stem in ["bass","drums","guitars","keyboards","orchestral","percussion","synthesizers","vocals"]
"""

import argparse
import datetime
import glob
import os
from os.path import join, isfile
import json
import logging
import sys
import time
import yaml
from filelock import FileLock
from typing import Dict, Tuple, List, Optional

import psutil
import numpy as np
import soundfile as sf
import scipy.signal
import torch

from metrics import SI_SNR, FAD_CLAP


VERBOSITY_LEVEL = 'INFO'
WAIT_TIME = 50
MAX_TIME_DIFF = datetime.timedelta(seconds=30)

STEMS = [
    "bass", "drums", "guitars", "keyboards",
    "orchestral", "percussion", "synthesizers", "vocals"
]

# -------- Validation knobs --------
_MIN_SR = 8000           # minimum allowable sample rate (Hz)
_MAX_SR = 384000         # maximum allowable sample rate (Hz)
_MIN_SEC_WARN = 0.5      # warn if audio shorter than this (seconds)
_MAX_SEC_WARN = 30.0     # warn if audio longer than this (seconds)

# -------- Structure knobs --------
EXPECTED_FILES_PER_STEM = 125  # hard check: exact file count per stem
EXPECTED_NAME_TEMPLATE = "clip_{:04d}.wav"  # expected filename pattern

# -------------- Logging --------------
def get_logger(verbosity_level, use_error_log=False):
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(filename)s: %(message)s')
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    if use_error_log:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(formatter)
        logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger

LOGGER = get_logger(VERBOSITY_LEVEL)

# -------------- FS helpers --------------
def _ls(pattern): return sorted(glob.glob(pattern))

def _here(*args):
    here_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.abspath(join(here_dir, *args))

def _listdir_case_insensitive(path: str) -> List[str]:
    try:
        return sorted(os.listdir(path))
    except FileNotFoundError:
        return []

def _list_wavs(directory):
    return sorted(glob.glob(join(directory, "*.wav")))

# -------------- Audio helpers --------------
def _safe_read_wav(path: str) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    """Read wav with guardrails. Returns (data, sr, err_msg)."""
    try:
        data, sr = sf.read(path, always_2d=False)
    except Exception as e:
        return None, None, f"unreadable file: {e}"
    if not isinstance(data, np.ndarray):
        return None, None, "not a numpy array payload"
    if data.size == 0:
        return None, None, "zero-length audio"
    if not np.isfinite(data).all():
        return None, None, "contains NaN/Inf"
    if not (_MIN_SR <= int(sr) <= _MAX_SR):
        return None, None, f"sample rate out of bounds: {sr}"
    dur = data.shape[0] / float(sr) if sr else 0.0
    if dur < _MIN_SEC_WARN:
        LOGGER.warning(f"[VALIDATION] Very short audio ({dur:.3f}s): {path}")
    if dur > _MAX_SEC_WARN:
        LOGGER.warning(f"[VALIDATION] Very long audio ({dur:.3f}s): {path}")
    return data, int(sr), None

def _read_wav_mono(path):
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    return data.astype(np.float64), int(sr)

def _resample_to(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data.astype(np.float64)
    g = np.gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return scipy.signal.resample_poly(data, up, down).astype(np.float64)

def _align_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(a), len(b))
    return a[:n], b[:n]

# -------------- Structure validation --------------
def _expected_clip_names(n: int) -> List[str]:
    return [EXPECTED_NAME_TEMPLATE.format(i) for i in range(n)]

def _validate_stem_folder(root: str, expected_stems: List[str], role: str) -> Dict[str, List[str]]:
    """Return map stem->wav_paths; log warnings for structure issues."""
    found = {stem: _list_wavs(join(root, stem)) for stem in expected_stems}
    for stem, files in found.items():
        if not os.path.isdir(join(root, stem)):
            LOGGER.warning(f"[VALIDATION] {role}: missing stem folder: {join(root, stem)}")
        elif len(files) == 0:
            LOGGER.warning(f"[VALIDATION] {role}: no .wav files in stem: {join(root, stem)}")

    entries = _listdir_case_insensitive(root)
    extra = [d for d in entries if os.path.isdir(join(root, d)) and d not in expected_stems]
    if extra:
        LOGGER.warning(f"[VALIDATION] {role}: unexpected stem folder(s) present: {extra}")

    for stem in expected_stems:
        stem_dir = join(root, stem)
        if not os.path.isdir(stem_dir):
            continue
        other = [f for f in _listdir_case_insensitive(stem_dir) if f.lower().endswith(".wav") is False]
        other = [f for f in other if isfile(join(stem_dir, f))]
        if other:
            LOGGER.warning(f"[VALIDATION] {role}:{stem}: non-wav files ignored: {other}")
    return found

def _validate_per_stem_count(root: str, expected_count: Optional[int], role: str):
    """Hard check: each stem folder must exist and contain exactly expected_count files."""
    if expected_count is None:
        return
    for stem in STEMS:
        stem_dir = join(root, stem)
        files = _list_wavs(stem_dir)
        if not os.path.isdir(stem_dir):
            msg = f"[STRUCTURE] {role}: missing stem folder: {stem_dir}"
            print(msg); sys.exit(1)
        if len(files) != expected_count:
            msg = f"[STRUCTURE] {role}:{stem} expected {expected_count} .wav files, found {len(files)}"
            print(msg); sys.exit(1)

def _validate_expected_filenames(root: str, expected_count: int, role: str):
    """Hard check: filenames must match clip_0000.wav..clip_xxxx.wav exactly."""
    expected = set(_expected_clip_names(expected_count))
    for stem in STEMS:
        stem_dir = join(root, stem)
        basenames = sorted(os.path.basename(p) for p in _list_wavs(stem_dir))
        got = set(basenames)
        missing = sorted(expected - got)
        extra   = sorted(got - expected)
        if missing or extra:
            print(f"[STRUCTURE] {role}:{stem} filename mismatch:")
            if missing:
                print("  Missing:")
                for name in missing:
                    print(f"    {name}")
            if extra:
                print("  Extra:")
                for name in extra:
                    print(f"    {name}")
            sys.exit(1)

def _validate_overlap(ref_map: Dict[str, List[str]], pred_map: Dict[str, List[str]], strict: bool):
    """Compare basenames in ref vs pred; warn or exit on mismatches."""
    for stem in STEMS:
        ref_basenames = {os.path.basename(p) for p in ref_map.get(stem, [])}
        pred_basenames = {os.path.basename(p) for p in pred_map.get(stem, [])}
        missing = sorted(list(ref_basenames - pred_basenames))
        orphan  = sorted(list(pred_basenames - ref_basenames))
        if missing:
            msg = f"[STRUCTURE] prediction missing {len(missing)} file(s) for stem '{stem}': {missing}"
            if strict:
                print(msg); sys.exit(1)
            else:
                LOGGER.warning(msg)
        if orphan:
            msg = f"[STRUCTURE] prediction has {len(orphan)} extra file(s) for stem '{stem}': {orphan}"
            if strict:
                print(msg); sys.exit(1)
            else:
                LOGGER.warning(msg)

def preflight_validate_inputs(solution_dir: str,
                              prediction_dir: str,
                              expected_files_per_stem: Optional[int],
                              strict_structure: bool) -> None:
    """Strict structure checks + light sanity validation. Exits on fatal issues."""
    # Root dirs must exist
    if not os.path.isdir(solution_dir):
        print(f"[STRUCTURE] Reference directory does not exist: {solution_dir}"); sys.exit(1)
    if not os.path.isdir(prediction_dir):
        print(f"[STRUCTURE] Prediction directory does not exist: {prediction_dir}"); sys.exit(1)

    # Required stem folders presence + per-stem count (hard stop)
    _validate_per_stem_count(solution_dir, expected_files_per_stem, "reference")
    _validate_per_stem_count(prediction_dir, expected_files_per_stem, "prediction")

    # Expected filenames (hard stop)
    if expected_files_per_stem is not None:
        _validate_expected_filenames(solution_dir, expected_files_per_stem, "reference")
        _validate_expected_filenames(prediction_dir, expected_files_per_stem, "prediction")

    # Mappings and filename overlap (optionally strict)
    ref_map  = _validate_stem_folder(solution_dir, STEMS, role="reference")
    pred_map = _validate_stem_folder(prediction_dir, STEMS, role="prediction")
    _validate_overlap(ref_map, pred_map, strict=strict_structure)

    # Probe a few files for readability (warn only)
    probes = 0; readable = 0
    for stem in STEMS:
        for p in (ref_map.get(stem, [])[:2] + pred_map.get(stem, [])[:2]):
            probes += 1
            _, _, err = _safe_read_wav(p)
            if err is None:
                readable += 1
            else:
                LOGGER.warning(f"[VALIDATION] Unusable wav ({err}): {p}")

    total_ref = sum(len(v) for v in ref_map.values())
    total_pred = sum(len(v) for v in pred_map.values())
    if total_ref == 0:
        print("[STRUCTURE] No reference .wav files found under expected stems."); sys.exit(1)
    if total_pred == 0:
        print("[STRUCTURE] No prediction .wav files found under expected stems."); sys.exit(1)
    if readable == 0 and probes > 0:
        LOGGER.warning("[VALIDATION] None of the sampled files were readable; scoring may fail later.")

# -------------- Metrics --------------
def _compute_stem_metrics(stem_ref_dir: str, stem_pred_dir: str, device: str):
    """Compute SI-SNR (mean) and FAD-CLAP for a single stem directory pair."""
    ref_files = _list_wavs(stem_ref_dir)
    if not ref_files:
        LOGGER.warning(f"No reference wavs in {stem_ref_dir}; skipping stem.")
        return None

    si_snr_metric = SI_SNR().to(device)
    fad_metric = FAD_CLAP().to(device)

    missing = 0
    used = 0

    for ref_path in ref_files:
        fname = os.path.basename(ref_path)
        pred_path = join(stem_pred_dir, fname)
        if not os.path.exists(pred_path):
            missing += 1
            continue

        try:
            clean, sr_c = _read_wav_mono(ref_path)
            pred, sr_p = _read_wav_mono(pred_path)
        except Exception as e:
            LOGGER.warning(f"[VALIDATION] Failed to read pair ({fname}): {e}; skipping.")
            continue

        # SI-SNR @ 16 kHz
        clean_16 = _resample_to(clean, sr_c, 16000)
        pred_16 = _resample_to(pred, sr_p, 16000)
        clean_16, pred_16 = _align_pair(clean_16, pred_16)
        if len(clean_16) == 0:
            LOGGER.warning(f"[VALIDATION] Zero-length after align @16k: {fname}; skipping.")
            continue
        t_clean_16 = torch.from_numpy(clean_16).to(device).unsqueeze(0)
        t_pred_16 = torch.from_numpy(pred_16).to(device).unsqueeze(0)
        si_snr_metric.update(t_pred_16, t_clean_16)

        # FAD-CLAP @ 48 kHz
        clean_48 = _resample_to(clean, sr_c, 48000)
        pred_48 = _resample_to(pred, sr_p, 48000)
        clean_48, pred_48 = _align_pair(clean_48, pred_48)
        if len(clean_48) == 0:
            LOGGER.warning(f"[VALIDATION] Zero-length after align @48k: {fname}; skipping for FAD.")
        else:
            t_clean_48 = torch.from_numpy(clean_48).to(device).unsqueeze(0).float()
            t_pred_48 = torch.from_numpy(pred_48).to(device).unsqueeze(0).float()
            fad_metric.update(t_pred_48, t_clean_48)

        used += 1

    if used == 0:
        LOGGER.warning(f"No overlapping files for stem dir {stem_ref_dir}; skipping.")
        return None

    si = si_snr_metric.compute()   # {'mean','std','count'}
    fad = fad_metric.compute()     # {'fad','count'}
    return {"si_snr": float(si["mean"]), "fad_clap": float(fad["fad"]),
            "count": int(si.get("count", 0)), "missing": missing}

def _compute_all_metrics(solution_dir: str, prediction_dir: str):
    """Compute per-stem metrics, then macro-average across available stems."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    per_stem = {}
    available_stems = []

    for stem in STEMS:
        stem_ref = join(solution_dir, stem)
        stem_pred = join(prediction_dir, stem)
        if not os.path.isdir(stem_ref):
            LOGGER.warning(f"Reference stem folder missing: {stem_ref}")
        if not os.path.isdir(stem_pred):
            LOGGER.warning(f"Prediction stem folder missing: {stem_pred}")
        if not (os.path.isdir(stem_ref) and os.path.isdir(stem_pred)):
            continue

        res = _compute_stem_metrics(stem_ref, stem_pred, device)
        if res is None:
            continue
        per_stem[stem] = res
        available_stems.append(stem)

    if not available_stems:
        raise FileNotFoundError("No valid stems found in both reference and prediction directories.")

    # Macro average across stems (equal weight per stem)
    si_macro = float(np.mean([per_stem[s]["si_snr"] for s in available_stems]))
    fad_macro = float(np.mean([per_stem[s]["fad_clap"] for s in available_stems]))

    LOGGER.info(f"Stems scored: {available_stems}")
    LOGGER.info(f"Macro SI-SNR = {si_macro:.4f} dB; Macro FAD-CLAP = {fad_macro:.6f}")

    return si_macro, fad_macro, per_stem, available_stems

# -------------- Output --------------
def _write_scores_html(score_dir, auto_refresh=True, append=False):
    filename = 'detailed_results.html'
    html_head = ('<html><head> <meta http-equiv="refresh" content="5"> '
                 '</head><body><pre>') if auto_refresh else "<html><body><pre>"
    html_end = '</pre></body></html>'
    mode = 'a' if append else 'w'
    filepath = join(score_dir, filename)
    with open(filepath, mode) as html_file:
        html_file.write(html_head)
        html_file.write(html_end)
    LOGGER.debug(f"Wrote learning curve page to {filepath}")

def write_scores(score_dir, si_macro, fad_macro, duration, per_stem, stems_scored):
    """Write macro and per-stem metrics to score_dir/scores.txt"""
    score_filename = join(score_dir, 'scores.txt')
    with open(score_filename, 'w') as f:
        f.write(f'si_snr: {si_macro}\n')
        f.write(f'fad_clap: {fad_macro}\n')
        f.write(f'duration: {duration}\n')
        for stem in stems_scored:
            si = per_stem[stem]["si_snr"]
            fad = per_stem[stem]["fad_clap"]
            cnt = per_stem[stem]["count"]
            f.write(f'stem_{stem}_si_snr: {si}\n')
            f.write(f'stem_{stem}_fad_clap: {fad}\n')
            f.write(f'stem_{stem}_count: {cnt}\n')
    LOGGER.debug(f"Wrote scores (macro + per-stem) to {score_filename}")

# -------------- Ingestion wrappers --------------
def _init_scores_html(detailed_results_filepath):
    html_head = ('<html><head> <meta http-equiv="refresh" content="5"> '
                 '</head><body><pre>')
    html_end = '</pre></body></html>'
    with open(detailed_results_filepath, 'w') as html_file:
        html_file.write(html_head)
        html_file.write("Starting scoring process... <br> Waiting for predictions.")
        html_file.write(html_end)

def _update_scores(args, duration):
    si_macro, fad_macro, per_stem, stems_scored = _compute_all_metrics(
        solution_dir=args.solution_dir,
        prediction_dir=args.prediction_dir
    )
    _write_scores_html(args.score_dir)
    write_scores(args.score_dir, si_macro, fad_macro, duration, per_stem, stems_scored)
    LOGGER.info(f"SI-SNR (macro): {si_macro:.4f} dB | FAD-CLAP (macro): {fad_macro:.6f}")
    return si_macro, fad_macro

def _detect_ingestion_alive(args):
    start_filepath = join(args.prediction_dir, 'start.txt')
    lockfile = join(args.prediction_dir, 'start.txt.lock')
    if not os.path.exists(start_filepath):
        return False
    with FileLock(lockfile):
        with open(start_filepath, 'r') as ftmp:
            last_time = datetime.datetime.fromtimestamp(json.load(ftmp))
    current_time = datetime.datetime.now()
    return (current_time - last_time) <= MAX_TIME_DIFF

def _init(args):
    os.makedirs(args.score_dir, exist_ok=True)
    detailed_results_filepath = join(args.score_dir, 'detailed_results.html')
    _init_scores_html(detailed_results_filepath)
    LOGGER.info('===== wait for ingestion to start')
    for _ in range(WAIT_TIME):
        if _detect_ingestion_alive(args):
            LOGGER.info('===== detect alive ingestion')
            break
        time.sleep(1)
    else:
        raise RuntimeError("[-] Failed: scoring didn't detect the start of ingestion in time.")

def _finalize(args, scoring_start):
    duration = time.time() - scoring_start
    LOGGER.info(f"[+] Successfully finished scoring! Scoring duration: {duration:.2f} sec.")
    LOGGER.info("[Scoring terminated]")

def _exist_endfile(args): return isfile(join(args.prediction_dir, 'end.txt'))

def _get_ingestion_info(args):
    endfile = join(args.prediction_dir, 'end.txt')
    with open(endfile, 'r') as ftmp:
        ingestion_info = json.load(ftmp)
    return ingestion_info

# ---------------- CLI ----------------
def _parse_args():
    root_dir = _here(os.pardir)
    default_solution_dir = join(root_dir, "reference_data")
    default_prediction_dir = join(root_dir, "sample_result_submission")
    default_score_dir = join(root_dir, "scoring_output")
    parser = argparse.ArgumentParser()
    parser.add_argument('--solution_dir', type=str, default=default_solution_dir,
                        help="Root directory of reference stems")
    parser.add_argument('--prediction_dir', type=str, default=default_prediction_dir,
                        help="Root directory of predicted stems")
    parser.add_argument('--score_dir', type=str, default=default_score_dir,
                        help="Directory to write scores and details")
    parser.add_argument('--skip_ingestion', action='store_true',
                        help='Skip waiting for ingestion; use for local runs or result-only submissions.')
    parser.add_argument('--strict_structure', action='store_true',
                        help='If set, filename overlap mismatches become fatal (exit).')
    args = parser.parse_args()
    LOGGER.debug(f"Parsed args: {args}")
    return args

def main():
    scoring_start = time.time()
    LOGGER.info('===== init scoring program')
    args = _parse_args()

    expected = EXPECTED_FILES_PER_STEM

    # ---- HARD STOP structure checks before anything else ----
    preflight_validate_inputs(
        solution_dir=args.solution_dir,
        prediction_dir=args.prediction_dir,
        expected_files_per_stem=expected,
        strict_structure=args.strict_structure
    )

    if args.skip_ingestion:
        os.makedirs(args.score_dir, exist_ok=True)
        detailed_results_filepath = join(args.score_dir, 'detailed_results.html')
        _init_scores_html(detailed_results_filepath)
        duration = 0.0
        _update_scores(args, duration)
        _finalize(args, scoring_start)
        return

    _init(args)
    LOGGER.info('===== wait for the exit of ingestion or end.txt file')
    while _detect_ingestion_alive(args) and (not _exist_endfile(args)):
        time.sleep(1)

    if not _exist_endfile(args):
        raise RuntimeError("no end.txt exist, ingestion failed")
    else:
        LOGGER.info('===== end.txt detected, reading ingestion information')
        ingestion_info = _get_ingestion_info(args)
        duration = ingestion_info.get('ingestion_duration', 0.0)
        _update_scores(args, duration)

    _finalize(args, scoring_start)

if __name__ == "__main__":
    main()
