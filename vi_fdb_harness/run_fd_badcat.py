#!/usr/bin/env python3
"""Run fd-badcat against Vi-FDB using the harness shared-clock contract."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import time
import wave
from array import array
from collections import deque
from pathlib import Path
from typing import Any


SAMPLE_RATE = 16_000
CHUNK_SAMPLES = 256


def _read_pcm16(path: Path) -> list[int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        if channels != 1 or width != 2 or rate != SAMPLE_RATE:
            raise ValueError(
                f"{path}: expected mono PCM16 {SAMPLE_RATE} Hz, "
                f"got channels={channels}, width={width}, rate={rate}"
            )
        raw = handle.readframes(handle.getnframes())
    samples = array("h")
    samples.frombytes(raw)
    if sys.byteorder != "little":
        samples.byteswap()
    return list(samples)


def _write_pcm16(path: Path, samples: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = array("h", samples)
    if sys.byteorder != "little":
        payload.byteswap()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(payload.tobytes())


def _decode_audio(message: bytes) -> list[int]:
    if message[:4] == b"RIFF":
        # Legacy fd-badcat sends one complete WAV file per response.
        import io

        with wave.open(io.BytesIO(message), "rb") as handle:
            if handle.getframerate() != SAMPLE_RATE or handle.getsampwidth() != 2:
                raise ValueError("legacy WAV response is not 16 kHz PCM16")
            raw = handle.readframes(handle.getnframes())
        message = raw
    if len(message) % 2:
        raise ValueError("PCM16 response has an odd byte count")
    values = array("h")
    values.frombytes(message)
    if sys.byteorder != "little":
        values.byteswap()
    return list(values)


def _float32_frame(samples: list[int]) -> bytes:
    values = [sample / 32768.0 for sample in samples]
    return struct.pack(f"<{len(values)}f", *values)


def _manifest_audio(dataset_root: Path, row: dict, condition: str) -> Path | None:
    field = "input" if condition == "event" else "clean_input"
    value = row.get(field)
    if not value:
        return None
    path = dataset_root / value
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _map_event(name: str) -> str:
    """Expose fd-badcat lifecycle names in the harness' neutral event schema."""
    return {
        "vad_start": "input_audio_buffer.speech_started",
        "response_started": "response.created",
        "response_complete": "response.done",
    }.get(name, name)


async def _run_case(
    *,
    ws_url: str,
    dataset_root: Path,
    run_root: Path,
    row: dict,
    condition: str,
    exp_name: str,
    tail_seconds: float,
    close_grace_seconds: float,
    overwrite: bool,
) -> dict:
    from harness import result_dir, safe_link, write_json

    source = _manifest_audio(dataset_root, row, condition)
    if source is None:
        return {"status": "not_applicable", "condition": condition}

    folder = result_dir(run_root, row)
    folder.mkdir(parents=True, exist_ok=True)
    input_name = "input.wav" if condition == "event" else "clean_input.wav"
    output_name = "output.wav" if condition == "event" else "clean_output.wav"
    timing_name = "output_timing.json" if condition == "event" else "clean_output_timing.json"
    output = folder / output_name
    timing_path = folder / timing_name
    if output.exists() and timing_path.exists() and not overwrite:
        return {"status": "skipped", "condition": condition, "output": str(output)}

    safe_link(source, folder / input_name)
    safe_link(dataset_root / row["metadata"], folder / "metadata.json")
    input_samples = _read_pcm16(source)
    input_duration = len(input_samples) / SAMPLE_RATE
    events: list[dict[str, Any]] = []
    pending_legacy: deque[float] = deque()
    active_segment: dict[str, Any] | None = None
    assistant_segments: list[dict[str, Any]] = []
    receiver_error: str | None = None
    server_offset: float | None = None

    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("Install the adapter dependency with: uv sync --group fd-badcat") from exc

    async with websockets.connect(ws_url, max_size=None) as websocket:
        await websocket.send(json.dumps({
            "event": "config",
            "data": {"lang": row.get("task", "vi_fdb"), "exp": exp_name},
        }))

        def mapped_time(obj: dict, received_at: float) -> tuple[float, float | None]:
            nonlocal server_offset
            data = obj.get("data") or {}
            backend_timestamp = data.get("timestamp")
            try:
                backend_timestamp = float(backend_timestamp)
            except (TypeError, ValueError):
                backend_timestamp = None
            if backend_timestamp is not None and server_offset is None:
                server_offset = received_at - backend_timestamp
            if backend_timestamp is None or server_offset is None:
                return received_at, backend_timestamp
            return backend_timestamp + server_offset, backend_timestamp

        async def receiver() -> None:
            nonlocal active_segment, receiver_error

            def finish_segment() -> None:
                nonlocal active_segment
                segment = active_segment
                active_segment = None
                if not segment or not segment["audio"]:
                    return
                audio = []
                for chunk in segment["audio"]:
                    audio.extend(chunk)
                assistant_segments.append({
                    "start": float(segment["start"]),
                    "duration": len(audio) / SAMPLE_RATE,
                    "samples": audio,
                })

            try:
                while True:
                    message = await websocket.recv()
                    received_at = time.perf_counter() - clock_start
                    if isinstance(message, bytes):
                        if active_segment is not None:
                            active_segment["audio"].append(_decode_audio(message))
                        elif pending_legacy:
                            audio = _decode_audio(message)
                            assistant_segments.append({
                                "start": pending_legacy.popleft(),
                                "duration": len(audio) / SAMPLE_RATE,
                                "samples": audio,
                            })
                        else:
                            events.append({"type": "audio_without_response", "time": received_at})
                        continue

                    try:
                        obj = json.loads(message)
                    except json.JSONDecodeError:
                        events.append({"type": "unparsed_message", "time": received_at})
                        continue
                    name = str(obj.get("event", "unknown"))
                    event_time, backend_timestamp = mapped_time(obj, received_at)
                    data = obj.get("data") or {}
                    events.append({
                        "type": _map_event(name),
                        "event": name,
                        "time": round(event_time, 6),
                        "backend_timestamp": backend_timestamp,
                        "data": data,
                    })
                    if name == "tts_segment_start":
                        finish_segment()
                        active_segment = {"start": event_time, "audio": []}
                    elif name in {"tts_provider_ready", "tts_first_audio"}:
                        if active_segment is not None and not active_segment["audio"]:
                            active_segment["start"] = event_time
                    elif name == "tts_segment_end":
                        finish_segment()
                    elif name == "tts_done":
                        pending_legacy.append(event_time)
            except websockets.exceptions.ConnectionClosed:
                finish_segment()
            except Exception as exc:
                finish_segment()
                receiver_error = repr(exc)

        clock_start = time.perf_counter()
        receiver_task = asyncio.create_task(receiver())
        for offset in range(0, len(input_samples), CHUNK_SAMPLES):
            chunk = input_samples[offset:offset + CHUNK_SAMPLES]
            if len(chunk) < CHUNK_SAMPLES:
                chunk += [0] * (CHUNK_SAMPLES - len(chunk))
            target = offset / SAMPLE_RATE
            elapsed = time.perf_counter() - clock_start
            if target > elapsed:
                await asyncio.sleep(target - elapsed)
            await websocket.send(_float32_frame(chunk))

        # fd-badcat treats client_end as cancellation. Send real-time silence
        # first so VAD can finalize the last segment and the response can play.
        tail_samples = int(round(max(0.0, tail_seconds) * SAMPLE_RATE))
        tail_start = len(input_samples) / SAMPLE_RATE
        silent = [0] * CHUNK_SAMPLES
        for offset in range(0, tail_samples, CHUNK_SAMPLES):
            target = tail_start + offset / SAMPLE_RATE
            elapsed = time.perf_counter() - clock_start
            if target > elapsed:
                await asyncio.sleep(target - elapsed)
            await websocket.send(_float32_frame(silent))
        if close_grace_seconds > 0:
            await asyncio.sleep(close_grace_seconds)
        await websocket.send(json.dumps({"event": "end"}))
        await websocket.close()
        await receiver_task

    output_duration = input_duration
    for segment in assistant_segments:
        output_duration = max(output_duration, segment["start"] + segment["duration"])
    output_samples = [0] * max(1, int(output_duration * SAMPLE_RATE + 0.999))
    serializable_segments = []
    for segment in assistant_segments:
        start = max(0.0, float(segment["start"]))
        start_index = round(start * SAMPLE_RATE)
        audio = segment["samples"]
        for index, sample in enumerate(audio):
            target = start_index + index
            if target >= len(output_samples):
                break
            output_samples[target] = max(-32768, min(32767, output_samples[target] + sample))
        serializable_segments.append({
            "start": start,
            "duration": segment["duration"],
        })

    _write_pcm16(output, output_samples)
    write_json(timing_path, {
        "schema_version": 1,
        "clock": "monotonic_from_first_input_frame",
        "task": row.get("task"),
        "id": row.get("id"),
        "condition": condition,
        "input_duration": input_duration,
        "output_timeline_duration": len(output_samples) / SAMPLE_RATE,
        "sample_rate": SAMPLE_RATE,
        "events": events,
        "assistant_segments": serializable_segments,
        "receiver_error": receiver_error,
        "model": os.getenv("FDBBADCAT_MODEL_ID", "MiniCPM-o-4_5"),
        "ws_url": ws_url,
    })
    return {
        "status": "completed" if receiver_error is None else "partial",
        "condition": condition,
        "output": str(output),
        "wall_seconds": round(time.perf_counter() - clock_start, 3),
        "assistant_segments": len(assistant_segments),
    }


async def _run_all(args, rows: list[dict], dataset_root: Path, run_root: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    conditions = [args.condition] if args.condition != "both" else ["event", "clean"]
    jobs = [
        (row, condition)
        for row in rows
        for condition in conditions
        if condition == "event" or row.get("clean_input")
    ]
    for index, (row, condition) in enumerate(jobs, 1):
        try:
            result = await _run_case(
                ws_url=args.ws_url,
                dataset_root=dataset_root,
                run_root=run_root,
                row=row,
                condition=condition,
                exp_name=args.exp_name,
                tail_seconds=args.tail_seconds,
                close_grace_seconds=args.close_grace_seconds,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            result = {"status": "failed", "condition": condition, "error": repr(exc)}
        status = result["status"]
        counts[status] = counts.get(status, 0) + 1
        print(f"[{index}/{len(jobs)}] {status}: {result.get('output') or result.get('error') or condition}", flush=True)
    return counts


def run_fd_badcat(args) -> int:
    from harness import load_manifest, write_json

    dataset_root = args.dataset_root.resolve()
    run_root = args.run_root.resolve()
    rows = load_manifest(dataset_root)
    if args.limit:
        rows = rows[: args.limit]
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "run_config.json", {
        "adapter": "fd-badcat-websocket",
        "dataset_root": str(dataset_root),
        "ws_url": args.ws_url,
        "model": os.getenv("FDBBADCAT_MODEL_ID", "MiniCPM-o-4_5"),
        "clock": "monotonic_from_first_input_frame",
        "conditions": [args.condition] if args.condition != "both" else ["event", "clean"],
        "jobs": 1,
        "tail_seconds": args.tail_seconds,
        "close_grace_seconds": args.close_grace_seconds,
    })
    counts = asyncio.run(_run_all(args, rows, dataset_root, run_root))
    write_json(run_root / "run_summary.json", counts)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 1 if counts.get("failed") else 0
