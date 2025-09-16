# INSTALL

Two supported paths:

## A) System Python (with `curses`)

> Works on most Linux head nodes. Requires Python ≥ 3.8 with `curses`.

```bash
# 1) Ensure Python has curses:
python3 -c "import curses; print('curses OK')"

# 2) Clone
git clone https://github.com/your-org/pbstop.git
cd pbstop

# 3) (Optional) Create a venv (recommended)
python3 -m venv .venv
. .venv/bin/activate

# 4) Run
./pbstop.py --config config.default.json
# or your site config:
./pbstop.py --config config.lovelace.json
```
# To put it on PATH:

```bash
sudo install -m 0755 pbstop.py /usr/local/bin/pbstop
sudo install -m 0644 config.default.json /etc/pbstop.json
# run with:
pbstop --config /etc/pbstop.json
```

## B) No system Python, no conda — use uv

uv is a single self-contained binary that manages its own Python

# 1) Install uv (single binary)
curl -fsSL https://astral.sh/uv/install.sh | sudo sh
# This installs `uv` to ~/.local/bin or /usr/local/bin depending on environment.

# 2) Clone
git clone https://github.com/your-org/pbstop.git
cd pbstop

# 3) Run using uv-managed Python
uv run ./pbstop.py --config config.default.json
# or:
uv run ./pbstop.py --config config.lovelace.json


To have a visible command on PATH without touching system Python:

sudo tee /usr/local/bin/pbstop <<'SH' >/dev/null
#!/usr/bin/env bash
# Wrapper that runs pbstop.py using uv-managed Python
exec uv run --quiet --project-dir /opt/pbstop /opt/pbstop/pbstop.py --config /etc/pbstop.json "$@"
SH
sudo chmod 0755 /usr/local/bin/pbstop
sudo mkdir -p /opt/pbstop
sudo cp pbstop.py /opt/pbstop/
sudo cp config.default.json /etc/pbstop.json

Adjust paths as needed (/opt/pbstop, /etc/pbstop.json).

Troubleshooting

_curses.error: addnwstr() returned ERR
Some terminals error when writing to the bottom-right cell. pbstop avoids it, but if you still see it, ensure $TERM is set (e.g., xterm-256color) and your terminal is wide enough.

UnicodeDecodeError in qstat
pbstop decodes qstat robustly (UTF-8 → Latin-1 → surrogateescape). If your locale is unusual, set:

export LC_ALL=C
export LANG=C


JSON from qstat is invalid
pbstop auto-repairs common breakages and falls back to a text parser. Press E to inspect the last raw failure.

No %MEM
Provide Resource_List.mem or select:...:mem= in PBS, or define queue_memory_defaults in your config.



