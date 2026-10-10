# LLM-BENCH

Retro terminal CLI that benchmarks **local** LLMs on your machine — single-model profiles or multi-model comparisons — across **Ollama**, **llama.cpp**, **MLX**, and **OpenAI-compatible** servers.

## Summary

After each run you get:

- a keygen-style live TUI + ranking/profile board in the terminal
- a self-contained **HTML report** with raw metrics (profile), comparison charts, memory timeline, and tables
- two sub-indices — **performance** (how fast) and **capability** (how good) — plus a **reliability** figure (share of requests that did not error), a per-capability breakdown, per-category sub-scores, latency percentiles, and a drill-down list of wrong answers

## What gets measured

13 tasks in two groups. Everything is graded **offline and deterministically** — no LLM-as-judge, no network.

**Performance** (`--tasks performance`)

| Task | Measures |
|------|----------|
| `speed` | decode tok/s (mean, p50, p95), TTFT, inter-token latency (mean/p95/max) |
| `latency` | prefill tok/s and TTFT at 0.5k / 2k / 8k-character prompts |
| `throughput` | aggregate tok/s with 1, 2 and 4 concurrent requests, and the scaling factor |

**Capability** (`--tasks capability`)

| Task | Items | Measures | Sub-scores |
|------|------:|----------|------------|
| `reasoning` | 18 | word problems, arithmetic, logic; exact-answer match | arithmetic, word problem, counting, logic, pattern |
| `knowledge` | 22 | multiple-choice facts (answer position is balanced A–D) | science, geography, history, technology, math, language |
| `coding` | 10 | write a Python function; **pass@1 by running unit tests** | strings, lists, algorithms, basics |
| `instructions` | 14 | IFEval-style verifiable constraints (length, casing, format, forbidden words); partial credit per constraint | format, casing, length, structure, precision |
| `tools` | 6 | tool selection, argument accuracy (including types), and *not* calling a tool when none is needed | selection, arguments, restraint |
| `context` | 6 | needle retrieval at 30%/70% of `--context` and 10/50/90% depth; also reports how much of the prompt the server actually kept | by depth, by length |
| `hallucination` | 12 | abstains on unanswerable questions, corrects false premises, and still answers controls (so refusing everything scores badly) | unanswerable, false premise, control |
| `multi_turn` | 6 | conversation memory, corrections, system-prompt and instruction carry-over | memory, system prompt, carry-over |
| `consistency` | 5 | same question ×3 at temperature 0.8: agreement and accuracy | — |
| `quality` | 4 | tiny sanity smoke test | — |

`<think>…</think>` blocks from reasoning models are stripped before grading and counted separately; a response that runs out of tokens mid-thought scores zero and shows up in the **Truncated** column.

### Scoring

`overall = 0.3 × performance + 0.7 × capability`, renormalised over whatever you ran. Capability is a weighted mean of the categories that ran (reasoning 20%, coding 20%, knowledge 15%, instructions 12%, tools 10%, context 8%, hallucination 7%, multi-turn 4%, sanity 3%, consistency 1%). Performance blends decode/TTFT (70%) and prefill (30%); with several models these are normalised against each other, with one model against fixed reference bands. A request that errors counts as a wrong answer, not as a missing one.

The scores are for comparing models run on the same machine with this tool. They are small fixed banks (10–22 items per task), so differences of a few points are noise — read the per-category and per-item detail, and use the item banks in `src/llm_bench/benchmarks/tasks/` as a starting point if you need something larger.

Every run is sandboxed in a workspace. When the run finishes (or aborts), llm-bench tears down **processes it started**, **unloads models it loaded**, and **deletes the sandbox**. Only the durable report under `--report` remains (unless you pass `--keep-workspace`).

### CLI

![llm-bench CLI — boot sequence, live progress, and terminal summary](docs/images/cli.png)

### Report

![llm-bench HTML report — test specs, raw profile metrics, memory timeline, and results table](docs/images/report.png)

## The live run screen

While benchmarks run, the terminal switches to a tracker-style screen (tan desktop, black bevelled panels, yellow channel bars, a scrolling note pattern) instead of a progress bar:

- **VU bars and scope** react to the song that is playing; the **pattern view** scrolls its real note onsets.
- **ETA** next to the progress bar: each remaining sample is priced at its own task's observed average duration (so a quick knowledge question and a slow coding task are not treated alike); it shows `--:--` until the first sample finishes. Model load time is not included, so treat it as a lower-bound guide.
- **Header fields** show the run ID, models, current model/task/sample (`Order` / `Pattern` / `Row`), live decode speed and pass rate; a log panel streams each sample's PASS / FAIL / tok/s.
- It adapts to the terminal size (full layout from about 100×34, compact below 64×20).

**Music plays only while the benchmark runs** — not in the setup wizard, and it is stopped before the final report prints (also on Ctrl+C). A playlist of free-to-use tracks (`src/llm_bench/assets/music/keygen-*.mp3`) plays one song after the other and wraps around. Use your own with `--music PATH` (a file or a folder) or `$LLM_BENCH_MUSIC`; see that folder's README. `--no-music` disables sound and the beat-sync.

**Controls during the run** (when stdin is a terminal):

| Key | Action |
|-----|--------|
| `p` / `Space` | pause / resume the music (the visuals pause with it) |
| `+` / `-` (or `↑` / `↓`) | volume up / down in 5% steps; the header shows a volume bar |
| `n` / `→` | next song |
| `b` / `←` | previous song (restarts the current one if it is more than 3 s in) |
| `Ctrl+C` | abort the run (music stops, terminal is restored) |

Pause freezes the player process, so resume continues mid-song. Volume is a start-up option of the players, so a change restarts playback at the same position (a burst of key presses is coalesced into one restart). `--volume 0.0–1.0` sets the starting level (default 0.7). Playback shells out to `afplay` / `ffplay` / `mpv`; the visuals fall back to a synthetic animation if there is no track or `ffmpeg`.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[mlx]"       # Apple Silicon mlx-lm
pip install -e ".[llamacpp]"  # llama-cpp-python (optional; path mode prefers llama-server binary)
```

## Quick start

Make sure the provider is running first (Ollama: `ollama serve`, then `ollama pull <model>`). If the server is down, llm-bench exits non-zero with a clear error instead of an empty-looking report.

By default, `llm-bench` / `llm-bench run` opens a **setup wizard** (provider, discovered models or custom path/URL, tasks, context, …). Use `--yes` to skip it.

```bash
# Interactive wizard (recommended)
llm-bench run

# Non-interactive, full suite (the default)
llm-bench run --yes -p ollama -m llama3.2:3b --context 8192 --report ./reports

# Quick pass: 5 items per capability task
llm-bench run --yes -p ollama -m llama3.2:3b --max-items 5

# Pick tasks or presets (all | performance | capability)
llm-bench run --yes -p ollama -m llama3.2:3b --tasks speed,latency,reasoning,coding

# Compare two models (non-interactive)
llm-bench run --yes -p ollama -m llama3.2:3b -m mistral:7b --tasks capability

# llama.cpp GGUF path or server URL
llm-bench run --yes -p llamacpp -m ./models/model.Q4_K_M.gguf --tasks speed,context
llm-bench run --yes -p openai --base-url http://127.0.0.1:8080/v1 -m local-model

# YAML config still opens the wizard with those defaults; add --yes to skip
llm-bench run -c examples/config.example.yaml --yes
```

### Useful flags

| Flag | Meaning |
|------|---------|
| `-m / --model` | Model name, local path, or URL (**repeatable**; one = profile) |
| `-p / --provider` | `ollama`, `llamacpp`, `mlx`, `openai` |
| `--tasks` | comma list (see `llm-bench list-tasks`) or preset `all` / `performance` / `capability` |
| `--max-items N` | Cap items per capability task (balanced across categories); `0` = full bank |
| `--runs` | Repeats for the speed/latency tasks (capability tasks sweep their whole bank once) |
| `--tools on\|off` | Enable/skip tool-calling task |
| `--no-music` / `--music PATH` / `--volume V` | Disable, choose, or set the starting volume of the live-screen track |
| `--no-code-exec` | Skip the `coding` task so model-written code is never executed |
| `--context` | Context window hint for long-context task |
| `--mode sequential\|parallel` | Multi-model scheduling |
| `--report DIR` | Durable HTML/JSON output (not used as scratch) |
| `--workspace DIR` | Override sandbox root |
| `--keep-workspace` | Keep raw logs after run (default: wipe) |
| `--yes` / `-y` | Skip setup wizard; require flags/config |

```bash
llm-bench list-providers
llm-bench list-tasks
```

### Running model-written code

The `coding` task executes the model's Python to run unit tests. Each candidate runs in its own `python -I` subprocess in a throw-away temp directory with a hard timeout, a CPU limit, and a minimal environment. **That is not a security sandbox** — a hostile model could still read files your user can read. Pass `--no-code-exec` (or `code_exec: false` in YAML) to skip it.

## Workspace cleanup contract

Runs use:

```text
~/.cache/llm-bench/runs/<run_id>/
  raw/     # JSONL samples
  tmp/     # scratch only
  report/  # draft artifacts
```

On exit (success, failure, Ctrl+C):

1. Stop child processes **we spawned** (e.g. `llama-server`)
2. Unload models **we loaded** (e.g. Ollama `keep_alive=0`, MLX GC)
3. Delete sandbox `tmp/` and, by default, the whole run workspace
4. Leave only `--report/<run_id>/` (HTML + JSON)

User-owned servers addressed by URL are **not** killed.

## Providers

| Provider | Model value |
|----------|-------------|
| `ollama` | Model name (`llama3.2:3b`) |
| `llamacpp` | `.gguf` path (spawns `llama-server` if available) or server URL |
| `mlx` | HF id / local MLX path (Apple Silicon, `[mlx]` extra) |
| `openai` | Model id on an OpenAI-compatible server (`--base-url`) |

## Development

```bash
pip install -e ".[dev]"
pytest
```

Tests use a mock provider and assert the workspace is removed with no leftover owned processes. An "oracle" provider that answers every task with its gold response checks that all graders and the scoring pipeline end to end give a perfect score, and every reference solution in the coding bank is executed against its tests.

## License

MIT
