# pbstop

A `top`-like TUI for PBS/OpenPBS clusters that runs on the head node.

![preview](docs/preview.png) <!-- optional -->

## Highlights

- Low-overhead: **1 `qstat` call per refresh** (JSON; auto-repair; text fallback).
- **Portable**, no third-party dependencies; **all policies in JSON config**.
- Two toplines (like `top`):
  - `R, Users, Updated, Poll, Sort`
  - `CPU% (1/5/15m) [EWMA]` and **global CPU efficiency** (non-EWMA).
- Columns: `USER, EGROUP, JOBID, QUEUE, START, WALL, MEM, %MEM, CPU%, EFF%, HOSTS`.
  - `WALL`: right-aligned `"H:MM:SS"` (hours without left zeros), width=10.
  - `MEM`: `GiB` with `'G'` anchored to the right of the cell.
  - `%MEM`: computed as `used / requested * 100` when possible; missing values show `-` and **sort last**.
  - `HOSTS`: compacted like `adano[01-03,12]` (configurable).
- CPU% < threshold (20% default) is dimmed.
- Detailed pane (`Enter`) shows: script path (`PBS_O_WORKDIR` + `Submit_arguments`), Start and Submitted timestamps, plus raw `qstat -f`.

## Why another PBS viewer?

- Strong focus on **portability**, **robust parsing**, and **admin-centric metrics** (CPU efficiency, weighted aggregates).
- **No assumptions baked into code** — cluster-specific policies live in a JSON file.

## Installation

See [INSTALL.md](INSTALL.md) for two options:

1) **System Python** (with `curses`) + `venv`.
2) **No system Python nor conda**: use [`uv`](https://github.com/astral-sh/uv) (single binary that manages its own Python).

## Quickstart

```bash
# clone
git clone https://github.com/your-org/pbstop.git
cd pbstop

# run with defaults
./pbstop.py

# run with your config
./pbstop.py --config config.lovelace.json

