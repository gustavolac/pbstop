# README.md

# pbstop

A `top`-like TUI for PBS/OpenPBS clusters that runs on the head node.

## Highlights

* Low-overhead: **1 **\`\`** call per refresh** (JSON; auto-repair; text fallback).
* **Portable**, no third-party dependencies; **all policies are in a JSON config**.
* Two toplines (like `top`):

  * `R, Users, Updated, Poll, Sort`
  * `CPU% (1/5/15m) [EWMA]` and **global CPU efficiency** (non‑EWMA).
* Columns: `USER, EGROUP, JOBID, QUEUE, START, WALL, MEM, %MEM, CPU%, EFF%, HOSTS`.

  * `WALL`: right‑aligned `"H:MM:SS"` (hours without leading zeros), width = 10.
  * `MEM`: GiB with `'G'` anchored to the right edge of the cell.
  * `%MEM`: `used / requested * 100` when possible; if unknown shows `-` and **sorts last**.
  * `HOSTS`: compacted like `adano[01‑03,12]` (configurable).
* CPU% below a threshold (20% by default) is dimmed.
* Details pane (`Enter`): raw `qstat -f` plus script path (`PBS_O_WORKDIR` + `Submit_arguments`) and Start/Submitted timestamps.

## Why another PBS viewer?

* Focus on **portability**, **robust parsing**, and **admin‑centric metrics** (CPU efficiency, weighted aggregates).
* **No cluster policy hard‑coded** — everything lives in a JSON config.

## Quickstart

```bash
# 1) Clone
git clone https://github.com/gustavolac/pbstop.git
cd pbstop

# 2) Run with built‑in defaults (most portable)
python3 pbstop.py

# 3) Run with a site config (strict JSON)
python3 pbstop.py --config config.lovelace.json
# or via environment variable:
PBSTOP_CONFIG=config.lovelace.json python3 pbstop.py
```

> **Shebang note:** the file currently uses `#!/usr/bin/env python3.12`. For best portability across head nodes, prefer invoking with `python3 pbstop.py`. If you want to run it as an executable (`./pbstop.py`), ensure `python3.12` exists on PATH *or* change the shebang to `#!/usr/bin/env python3` in your local copy.

## Keys

* Navigation: `↑/↓`, `PgUp/PgDn`, `g/G`, `Enter` (details)
* Sort: `e/E` (EFF%), `c/C` (CPU% now), `m/M` (%MEM), `w/W` (WALL), `u/U` (USER), `j/J` (JOBID), `o/O` (QUEUE), `s/S` (START) — lowercase = ascending, UPPERCASE = descending
* Filter: `f` (user/queue regex), `/` (substring search), `a` (clear)
* Misc: `r` (refresh), `t` (poll seconds), `x` (export CSV), `E` (last raw error), `?` (help), `q` (quit)

## Configuration

* Examples:

  * `config.default.json`: **READING TEMPLATE** (commented for explanation). **Remove comments** before using it as a real config file.
  * `config.lovelace.json`: site example (strict JSON, ready to use).
* `queue_memory_defaults` lets you derive **mem\_req** when the PBS job lacks an explicit memory request — enabling consistent `%MEM`.
* `qstat_candidates` controls how the binary is discovered.
* You may also set the environment variable `PBSTOP_CONFIG=/path/to/your.json` and run without `--config`.

## Troubleshooting

* `_curses.error: addnwstr() returned ERR` — Some terminals error on the bottom‑right cell. pbstop avoids it, but if it persists, set `$TERM` (e.g., `xterm-256color`) and ensure enough width.
* `UnicodeDecodeError` on `qstat` — The tool decodes robustly (UTF‑8 → Latin‑1 → surrogateescape). If your locale is uncommon, try `export LC_ALL=C; export LANG=C`.
* Invalid `qstat` JSON — pbstop attempts to auto‑repair and falls back to the text parser. Press `E` to inspect the last raw failure.
* No `%MEM` — Provide `Resource_List.mem` or `select:...:mem=` in PBS, or define `queue_memory_defaults` in the config.

## Admin extras (optional)

If enabled via `root_extra`, the details pane can show privileged shortcuts (full name, e‑mail, groups, “PBS script”). Disabled by default in `config.default.json`.
