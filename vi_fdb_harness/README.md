# Vietnamese FDB v1.0 Harness

This directory is the canonical harness for Vietnamese FDB v1.0. It keeps user audio and assistant audio on separate tracks with one monotonic clock, runs both event and clean conditions, normalizes Vietnamese ASR timestamps, applies a blinded behavior judge, and creates synchronized review HTML.

It is reserved for benchmark creation and evaluation. Dialogue-manager data
generation, training, and evaluation live in the sibling `dialogue_manager/`
project.

## Data and result terminology

- **Event condition** (`input.wav` → `output.wav`): the user question plus a later backchannel, interruption, side utterance, or background event.
- **Clean condition** (`clean_input.wav` → `clean_output.wav`): the same primary question and duration, with silence replacing the event. This exists for the 80 v1.5-derived cases.
- “Event condition” replaces the ambiguous earlier name “mixed.” User and assistant audio are never mono-mixed for scoring.

## Shared-clock contract

1. Connect to the live model and finish session configuration.
2. Set `t=0` immediately before transmitting the first user PCM frame.
3. Pace input against absolute source timestamps.
4. Position each assistant audio delta on the same monotonic clock.
5. Queue deltas for each response at audible playout speed.
6. Truncate queued audio at a response cancellation time.
7. Make the assistant timeline at least as long as the input timeline.
8. Save VAD and response lifecycle events in `*_timing.json`.

ASR runs independently on user and assistant tracks. Stereo/dual playback exists only for human review.

## Setup

Core validation and orchestration have no Python dependencies:

```bash
cd vi_fdb_harness
uv sync
```

GPU ASR:

```bash
uv sync --group asr
```

LLM judging:

```bash
uv sync --group judge
```

The GPT-Realtime JavaScript transport is installed separately:

```bash
cd ../v1_v1.5/model_inference/gpt-realtime
npm install
```

The fd-badcat WebSocket adapter uses one Python dependency:

```bash
uv sync --group fd-badcat
```

## 1. Validate a released dataset split

From `vi_fdb_harness/`:

```bash
uv run python harness.py validate-dataset \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710
```

The public release contains `pilot_160` and `expansion_240`. Validate each split
independently; together they contain 400 event cases and 120 paired clean
controls.

## 2. Run GPT-Realtime

Never store a key in the repository. Export it in the shell:

```bash
export OPENAI_API_KEY=...
```

One-sample check:

```bash
uv run python harness.py run-openai \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710 \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --node-cli ../v1_v1.5/model_inference/gpt-realtime/cli.js \
  --condition both --jobs 1 --limit 1
```

Run all event cases and any paired clean controls in a split:

```bash
uv run python harness.py run-openai \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710 \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --node-cli ../v1_v1.5/model_inference/gpt-realtime/cli.js \
  --condition both --jobs 4
```

The harness is resumable by default. Add `--overwrite` only to intentionally regenerate existing outputs.

Validate results:

```bash
uv run python harness.py validate-run \
  --dataset-root ../external/VieNeu-TTS/outputs/fdb_vi_pilot_160_20260710 \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime
```

## 3. Run fd-badcat on A100

Start the fd-badcat server first from its `a100-minicpm-vieneu-zipformer`
branch. Then run the same Vi-FDB dataset through the local WebSocket adapter:

```bash
uv run python harness.py run-fd-badcat \
  --dataset-root /path/to/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/vi_fdb_v1_0/fd_badcat_a100 \
  --ws-url ws://127.0.0.1:18000/realtime \
  --condition both --jobs 1 --limit 1
```

The adapter sends 16 ms float32 frames at realtime speed, adds trailing silence
so fd-badcat can finalize VAD, and only then closes the session. It accepts both
the legacy WAV response and fd-badcat's PCM16 streaming response. It writes the
standard `output.wav`/`output_timing.json` and paired clean artifacts under the
same result folders used by `validate-run`, `transcribe.py`, `judge.py`, and
`report.py`.

Run the full split after the smoke test:

```bash
uv run python harness.py validate-run \
  --dataset-root /path/to/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/vi_fdb_v1_0/fd_badcat_a100
```

### Complete two-A100 command checklist

Use this exact order when running the benchmark against the fd-badcat branch.

Terminal 1 — prepare fd-badcat and its Python environment:

```bash
git clone https://github.com/foresst123/fd-badcat.git
cd fd-badcat
git switch --track origin/a100-minicpm-vieneu-zipformer
conda create -n fd-badcat-a100 python=3.10 -y
conda activate fd-badcat-a100
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.a100.example .env.a100
set -a && source .env.a100 && set +a
```

Terminal 2 — start VieNeu on physical GPU 1:

```bash
cd /path/to/VieNeu-TTS
CUDA_VISIBLE_DEVICES=1 \
VIENEU_DEVICE=cuda \
VIENEU_PORT=19100 \
.venv/bin/python -m apps.openai_speech
curl http://127.0.0.1:19100/health
```

Terminal 3 — start MiniCPM/Zipformer/fd-badcat on GPUs 0 and 1:

```bash
cd /path/to/fd-badcat
conda activate fd-badcat-a100
set -a && source .env.a100 && set +a
python setup/a100_minicpm_server.py
```

Terminal 4 — prepare this harness and download the public dataset:

```bash
git clone https://github.com/foresst123/Full-Duplex-Bench.git
cd Full-Duplex-Bench
git switch --track origin/fd-badcat-a100-adapter
cd vi_fdb_harness
uv sync --group fd-badcat
hf download tuanamz/vi-fdb-v1 \
  --repo-type dataset \
  --local-dir ../data/vi-fdb-v1
```

Validate both dataset splits before inference:

```bash
uv run python harness.py validate-dataset \
  --dataset-root ../data/vi-fdb-v1/data/pilot_160 \
  --profile original-160

uv run python harness.py validate-dataset \
  --dataset-root ../data/vi-fdb-v1/data/expansion_240 \
  --profile expansion-240
```

Run one event/clean pair first:

```bash
uv run python harness.py run-fd-badcat \
  --dataset-root ../data/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/fd_badcat_a100_pilot \
  --ws-url ws://127.0.0.1:18000/realtime \
  --condition both --jobs 1 --limit 1

uv run python harness.py validate-run \
  --dataset-root ../data/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/fd_badcat_a100_pilot
```

Run the complete pilot, then the expansion split:

```bash
uv run python harness.py run-fd-badcat \
  --dataset-root ../data/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/fd_badcat_a100_pilot \
  --ws-url ws://127.0.0.1:18000/realtime \
  --condition both --jobs 1

uv run python harness.py run-fd-badcat \
  --dataset-root ../data/vi-fdb-v1/data/expansion_240 \
  --run-root ../outputs/fd_badcat_a100_expansion \
  --ws-url ws://127.0.0.1:18000/realtime \
  --condition event --jobs 1
```

Validate outputs and run the post-processing pipeline:

```bash
uv run python harness.py validate-run \
  --dataset-root ../data/vi-fdb-v1/data/pilot_160 \
  --run-root ../outputs/fd_badcat_a100_pilot

uv sync --group asr
uv run python transcribe.py \
  --root ../outputs/fd_badcat_a100_pilot \
  --backend phowhisper
uv run python align_phowhisper_vad.py \
  --root ../outputs/fd_badcat_a100_pilot

uv sync --group judge
export OPENAI_API_KEY=your_key_here
uv run python judge.py \
  --root ../outputs/fd_badcat_a100_pilot \
  --asr-backend phowhisper_vad
uv run python report.py \
  --run-root ../outputs/fd_badcat_a100_pilot \
  --asr-backend phowhisper_vad
```

Keep `--jobs 1`: one fd-badcat WebSocket session owns one realtime clock and
the A100 model/provider stack is intentionally single-session. Add
`--overwrite` only when deliberately regenerating completed samples.

## 4. Vietnamese ASR and observable speech boundaries

Generate word-level Vietnamese transcripts with PhoWhisper:

```bash
uv run python transcribe.py \
  --root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --backend phowhisper
```

Replace only the outer speech boundaries with boundaries measured directly from
the output audio by Silero VAD. Lexical units and inner word timestamps remain
from PhoWhisper:

```bash
uv run python align_phowhisper_vad.py \
  --root ../outputs/vi_fdb_v1_0/gpt_realtime
```

ChunkFormer remains available as an ASR comparison, but its phrase-level chunks
must not be passed to the original word-count timing rules as if they were
word-level chunks.

Export the selected ASR artifacts into the original English FDB directory
contract without changing its metric implementations:

```bash
uv run python export_english_fdb_compat.py \
  --source-data /path/to/normalized/vi-fdb \
  --system-run ../outputs/vi_fdb_v1_0/gpt_realtime \
  --output ../outputs/vi_fdb_v1_0/english_fdb_compat \
  --asr-backend phowhisper_vad
```

Every ASR backend writes the same normalized schema:

```json
{
  "schema_version": 1,
  "backend": "phowhisper",
  "audio": "output.wav",
  "text": "...",
  "chunks": [{"text": "...", "timestamp": [4.92, 5.24]}]
}
```

## 5. Blinded behavior judge

The judge sees the task, source text/event interval, and timestamped clean/event assistant transcripts. Dialogue-manager action tokens are not part of benchmark scoring. Internal API cancellation events are diagnostic only; the judge scores observable behavior.

```bash
export OPENAI_API_KEY=...
uv run python judge.py \
  --root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --asr-backend phowhisper_vad
```

Cases below 0.7 judge confidence and ASR-disagreement cases require human review.

## 6. Synchronized review report

```bash
uv run python report.py \
  --run-root ../outputs/vi_fdb_v1_0/gpt_realtime \
  --asr-backend phowhisper_vad
```

The generated `review.html` plays the event input and assistant output from the same timestamp. It does not alter scoring audio.

## Release gate

A run is eligible for scoring only when:

- Dataset validation reports zero errors.
- There are 160 event outputs and 80 clean outputs.
- Every output has a readable WAV and a corrected monotonic timing sidecar.
- ASR produces normalized timestamp chunks or explicitly records no speech.
- All ASR/judge disagreements are resolved or listed for human review.
- Model, voice, VAD settings, concurrency, and run date are recorded in `run_config.json`.
