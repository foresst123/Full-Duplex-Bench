#!/usr/bin/env python3
"""Reproducible inference and validation harness for Vietnamese FDB v1.0."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time
import wave

EXPECTED_COUNTS = {
    ("v1.0", "backchannel"): 20,
    ("v1.0", "pause_handling"): 20,
    ("v1.0", "smooth_turn_taking"): 20,
    ("v1.0", "user_interruption"): 20,
    ("v1.5", "background_speech"): 20,
    ("v1.5", "talking_to_other"): 20,
    ("v1.5", "user_backchannel"): 20,
    ("v1.5", "user_interruption"): 20,
}
EXPANSION_EXPECTED_COUNTS = {group: 30 for group in EXPECTED_COUNTS}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return {
            "frames": frames,
            "sample_rate": rate,
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "duration": frames / rate,
        }


def tail_rms(path: Path, seconds: float = 0.1) -> float:
    with wave.open(str(path), "rb") as handle:
        width = handle.getsampwidth()
        channels = handle.getnchannels()
        count = min(handle.getnframes(), round(handle.getframerate() * seconds))
        handle.setpos(handle.getnframes() - count)
        raw = handle.readframes(count)
    if width != 2 or not raw:
        return math.nan
    samples = struct.unpack(f"<{len(raw) // 2}h", raw)
    mono_energy = sum(value * value for value in samples) / max(1, len(samples) * channels)
    return math.sqrt(mono_energy) / 32768.0


def source_suite_version(metadata: dict) -> str | None:
    """Recover the inherited English-FDB suite version after HF flattening."""
    source = str(metadata.get("source_methodology", "")).lower()
    if "v1.5" in source:
        return "v1.5"
    if "v1.0" in source:
        return "v1.0"
    return None


def load_manifest(dataset_root: Path) -> list[dict]:
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    rows = read_json(manifest_path)
    if not isinstance(rows, list):
        raise ValueError("manifest.json must contain an array")
    normalized = []
    for original in rows:
        row = dict(original)
        # Public release manifests flatten v1.0/v1.5 from their paths. Recover
        # the local suite directory without changing the released dataset.
        metadata_rel = row.get("metadata")
        if metadata_rel and not (dataset_root / metadata_rel).exists():
            release_id = str(row.get("id", ""))
            local_id = release_id.split("_", 1)[-1]
            local_metadata = str(Path(row["task"]) / local_id / "metadata.json")
            candidates = [
                prefix / local_metadata
                for prefix in (Path("v1_0"), Path("v1_5"))
                if (dataset_root / prefix / local_metadata).exists()
            ]
            variant = row.get("evaluation_variant")
            if len(candidates) > 1 and variant in {"standard", "counterfactual"}:
                wanted = "v1_0" if variant == "standard" else "v1_5"
                candidates = [candidate for candidate in candidates if candidate.parts[0] == wanted]
            if len(candidates) == 1:
                prefix = candidates[0].parts[0]
                base = Path(prefix) / row["task"] / local_id
                row["input"] = str(base / "input.wav")
                row["metadata"] = str(base / "metadata.json")
                if row.get("clean_input"):
                    row["clean_input"] = str(base / "clean_input.wav")
                metadata_rel = row["metadata"]
        if not row.get("version") and row.get("metadata"):
            metadata_path = dataset_root / row["metadata"]
            if metadata_path.exists():
                row["version"] = source_suite_version(read_json(metadata_path))
        normalized.append(row)
    return normalized


def validate_dataset(dataset_root: Path, expected_counts: dict[tuple[str, str], int] | None = EXPECTED_COUNTS) -> dict:
    rows = load_manifest(dataset_root)
    errors: list[str] = []
    warnings: list[str] = []
    counts: dict[tuple[str, str], int] = {}
    seen = set()
    for row in rows:
        key = (row.get("version"), row.get("task"), row.get("id"))
        if key in seen:
            errors.append(f"duplicate sample: {key}")
        seen.add(key)
        group = key[:2]
        counts[group] = counts.get(group, 0) + 1
        for field in ("input", "metadata"):
            if not row.get(field):
                errors.append(f"{key}: missing {field}")
        input_path = dataset_root / row["input"]
        metadata_path = dataset_root / row["metadata"]
        if not input_path.exists() or not metadata_path.exists():
            errors.append(f"{key}: missing input or metadata file")
            continue
        try:
            info = wav_info(input_path)
            metadata = read_json(metadata_path)
        except Exception as exc:  # validation must report every broken case
            errors.append(f"{key}: unreadable artifact: {exc}")
            continue
        timestamps = metadata.get("timestamps")
        if not (isinstance(timestamps, list) and len(timestamps) == 2):
            errors.append(f"{key}: timestamps must be [start, end]")
        elif not (0 <= timestamps[0] <= timestamps[1] <= info["duration"] + 0.02):
            errors.append(f"{key}: timestamps outside input duration")
        rms = tail_rms(input_path)
        if not math.isnan(rms) and rms > 0.01:
            warnings.append(f"{key}: final 100 ms is not quiet (RMS={rms:.4f})")
        clean = row.get("clean_input")
        if row.get("version") == "v1.5" and not clean:
            errors.append(f"{key}: v1.5 requires clean_input")
        if clean and not (dataset_root / clean).exists():
            errors.append(f"{key}: missing clean_input file")
    if expected_counts is not None:
        for group, expected in expected_counts.items():
            if counts.get(group) != expected:
                errors.append(f"{group}: expected {expected}, found {counts.get(group, 0)}")
    return {
        "ok": not errors,
        "samples": len(rows),
        "counts": {f"{v}/{t}": count for (v, t), count in sorted(counts.items())},
        "errors": errors,
        "warnings": warnings,
    }


def result_dir(run_root: Path, row: dict) -> Path:
    return run_root / row["version"].replace(".", "_") / row["task"] / row["id"]


def safe_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def run_one(node_cli: Path, dataset_root: Path, run_root: Path, row: dict, condition: str, overwrite: bool) -> dict:
    source_rel = row["input"] if condition == "event" else row.get("clean_input")
    if not source_rel:
        return {"status": "not_applicable", "condition": condition}
    folder = result_dir(run_root, row)
    folder.mkdir(parents=True, exist_ok=True)
    source = dataset_root / source_rel
    input_name = "input.wav" if condition == "event" else "clean_input.wav"
    output_name = "output.wav" if condition == "event" else "clean_output.wav"
    timing_name = "output_timing.json" if condition == "event" else "clean_output_timing.json"
    output = folder / output_name
    if output.exists() and not overwrite:
        return {"status": "skipped", "condition": condition, "output": str(output)}
    safe_link(source, folder / input_name)
    safe_link(dataset_root / row["metadata"], folder / "metadata.json")
    prefix = folder / f".{condition}_combined.wav"
    generated_audio = prefix.with_name(prefix.stem + "_gpt_response.wav")
    generated_timing = prefix.with_name(prefix.stem + "_timing.json")
    log_path = folder / f"{condition}.log"
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                ["node", str(node_cli), "--input", str(source), "--output", str(prefix)],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                check=False,
                # A failed WebSocket handshake can otherwise leave Node's socket
                # handle alive indefinitely. Normal runs finish well inside this.
                timeout=float(os.getenv("REALTIME_PROCESS_TIMEOUT_SECONDS", "180")),
            )
        except subprocess.TimeoutExpired:
            log.write("\nHarness timeout: Realtime subprocess did not exit.\n")
            return {"status": "failed", "condition": condition, "log": str(log_path), "returncode": "timeout"}
    if proc.returncode != 0 or not generated_audio.exists() or not generated_timing.exists():
        return {"status": "failed", "condition": condition, "log": str(log_path), "returncode": proc.returncode}
    generated_audio.replace(output)
    generated_timing.replace(folder / timing_name)
    return {"status": "completed", "condition": condition, "output": str(output), "wall_seconds": round(time.monotonic() - started, 3)}


def run_openai(args) -> int:
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
        print("OPENAI_API_KEY or OPENAI_KEY is required", file=sys.stderr)
        return 2
    dataset_root = args.dataset_root.resolve()
    run_root = args.run_root.resolve()
    rows = load_manifest(dataset_root)
    if args.limit:
        rows = rows[: args.limit]
    conditions = [args.condition] if args.condition != "both" else ["event", "clean"]
    jobs = [(row, condition) for row in rows for condition in conditions if condition == "event" or row.get("clean_input")]
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_config.json", {
        "dataset_root": str(dataset_root),
        "model": os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime"),
        "clock": "monotonic_from_first_input_frame",
        "conditions": conditions,
        "jobs": args.jobs,
    })
    counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, args.node_cli.resolve(), dataset_root, run_root, row, condition, args.overwrite) for row, condition in jobs]
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            status = result["status"]
            counts[status] = counts.get(status, 0) + 1
            print(f"[{index}/{len(jobs)}] {status}: {result.get('output') or result.get('log') or result['condition']}", flush=True)
    write_json(run_root / "run_summary.json", counts)
    print(json.dumps(counts, indent=2))
    return 1 if counts.get("failed") else 0


def validate_run(dataset_root: Path, run_root: Path, require_complete: bool) -> dict:
    rows = load_manifest(dataset_root)
    errors, warnings = [], []
    event_complete = clean_complete = 0
    for row in rows:
        folder = result_dir(run_root, row)
        expected = [("output.wav", "output_timing.json", "event")]
        if row.get("clean_input"):
            expected.append(("clean_output.wav", "clean_output_timing.json", "clean"))
        for audio_name, timing_name, condition in expected:
            audio, timing = folder / audio_name, folder / timing_name
            if not audio.exists() or not timing.exists():
                if require_complete:
                    errors.append(f"{row['version']}/{row['task']}/{row['id']} {condition}: missing output/timing")
                continue
            try:
                info, data = wav_info(audio), read_json(timing)
                if data.get("clock") != "monotonic_from_first_input_frame":
                    errors.append(f"{audio}: unsupported clock")
                if abs(info["duration"] - data.get("output_timeline_duration", -1)) > 0.03:
                    errors.append(f"{audio}: WAV and timing duration disagree")
                if info["duration"] + 0.02 < data.get("input_duration", 0):
                    errors.append(f"{audio}: output timeline shorter than input")
            except Exception as exc:
                errors.append(f"{audio}: {exc}")
                continue
            if condition == "event": event_complete += 1
            else: clean_complete += 1
    return {"ok": not errors, "event_complete": event_complete, "clean_complete": clean_complete, "errors": errors, "warnings": warnings}


def main() -> None:
    parser = argparse.ArgumentParser(prog="vi-fdb")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-dataset")
    validate.add_argument("--dataset-root", type=Path, required=True)
    validate.add_argument(
        "--profile",
        choices=("original-160", "expansion-240", "schema-only"),
        default="original-160",
        help="Expected per-task counts; schema-only skips count checks.",
    )
    run = sub.add_parser("run-openai")
    run.add_argument("--dataset-root", type=Path, required=True)
    run.add_argument("--run-root", type=Path, required=True)
    run.add_argument("--node-cli", type=Path, required=True)
    run.add_argument("--condition", choices=["event", "clean", "both"], default="both")
    run.add_argument("--jobs", type=int, default=4)
    run.add_argument("--limit", type=int)
    run.add_argument("--overwrite", action="store_true")
    fd = sub.add_parser(
        "run-fd-badcat",
        help="Run a local fd-badcat WebSocket server against Vi-FDB.",
    )
    fd.add_argument("--dataset-root", type=Path, required=True)
    fd.add_argument("--run-root", type=Path, required=True)
    fd.add_argument(
        "--ws-url",
        default=os.getenv("FDB_WS_URL", "ws://127.0.0.1:18000/realtime"),
    )
    fd.add_argument("--condition", choices=["event", "clean", "both"], default="both")
    fd.add_argument("--limit", type=int)
    fd.add_argument("--overwrite", action="store_true")
    fd.add_argument("--tail-seconds", type=float, default=3.0)
    fd.add_argument("--close-grace-seconds", type=float, default=2.0)
    fd.add_argument("--exp-name", default="vi_fdb")
    fd.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Kept at 1 so one WebSocket session owns one monotonic clock.",
    )
    check = sub.add_parser("validate-run")
    check.add_argument("--dataset-root", type=Path, required=True)
    check.add_argument("--run-root", type=Path, required=True)
    check.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.command == "validate-dataset":
        expected_counts = {
            "original-160": EXPECTED_COUNTS,
            "expansion-240": EXPANSION_EXPECTED_COUNTS,
            "schema-only": None,
        }[args.profile]
        result = validate_dataset(args.dataset_root, expected_counts)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        raise SystemExit(0 if result["ok"] else 1)
    if args.command == "run-openai":
        raise SystemExit(run_openai(args))
    if args.command == "run-fd-badcat":
        if args.jobs != 1:
            parser.error("run-fd-badcat requires --jobs 1 to preserve realtime timing")
        from run_fd_badcat import run_fd_badcat
        raise SystemExit(run_fd_badcat(args))
    result = validate_run(args.dataset_root, args.run_root, not args.allow_partial)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
