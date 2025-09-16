# INSTALL.md

# INSTALL

Two supported paths:

## A) System Python (with `curses`)

> Works on most Linux head nodes. Requires Python ≥ 3.8 with `curses`.

```bash
# 1) Check that Python has curses:
python3 -c "import curses; print('curses OK')"

# 2) Clone
git clone https://github.com/gustavolac/pbstop.git
cd pbstop

# 3) (Optional) Create a venv
python3 -m venv .venv
. .venv/bin/activate

# 4) Run
python3 pbstop.py --config config.lovelace.json
# or simply with embedded defaults:
python3 pbstop.py
```

> **About **\`\`**:** it contains explanatory comments. If you use it, **strip all comments** to make it strict JSON before passing it to `--config` or via `PBSTOP_CONFIG`.

### Put it on PATH (recommended wrapper)

Install a small wrapper that explicitly calls `python3` (more portable than relying on the shebang):

```bash
sudo tee /usr/local/bin/pbstop <<'SH' >/dev/null
#!/usr/bin/env bash
exec python3 /opt/pbstop/pbstop.py --config /etc/pbstop.json "$@"
SH
sudo chmod 0755 /usr/local/bin/pbstop
sudo install -d /opt/pbstop
sudo cp pbstop.py /opt/pbstop/
sudo cp config.lovelace.json /etc/pbstop.json
# Edit /etc/pbstop.json for your cluster (strict JSON)
```

> **Alternative:** If you insist on installing the script directly, either ensure `python3.12` exists on PATH or change the shebang in `pbstop.py` to `#!/usr/bin/env python3` before `install -m 0755`.

## B) No system Python / no conda — use `uv`

[`uv`](https://github.com/astral-sh/uv) is a single binary that manages its own Python.

```bash
# 1) Install uv (single binary)
curl -fsSL https://astral.sh/uv/install.sh | sudo sh
# This installs uv to ~/.local/bin or /usr/local/bin depending on your environment.

# 2) Clone
git clone https://github.com/gustavolac/pbstop.git
cd pbstop

# 3) Run using the uv‑managed Python
uv run ./pbstop.py --config config.lovelace.json
# or:
uv run ./pbstop.py
```

To expose a command on PATH without touching system Python:

```bash
sudo tee /usr/local/bin/pbstop <<'SH' >/dev/null
#!/usr/bin/env bash
# Wrapper that uses uv‑managed Python
exec uv run --quiet --project-dir /opt/pbstop /opt/pbstop/pbstop.py --config /etc/pbstop.json "$@"
SH
sudo chmod 0755 /usr/local/bin/pbstop
sudo install -d /opt/pbstop
sudo cp pbstop.py /opt/pbstop/
sudo cp config.lovelace.json /etc/pbstop.json
```

## Troubleshooting

* `_curses.error: addnwstr() returned ERR` Some terminals error when writing to the bottom‑right cell. pbstop avoids it, but if you still see it, ensure `$TERM` is set (e.g., `xterm-256color`) and your terminal has enough width.

* `UnicodeDecodeError` in `qstat` pbstop decodes `qstat` robustly (UTF‑8 → Latin‑1 → surrogateescape). If your locale is unusual, try:

  ```bash
  export LC_ALL=C
  export LANG=C
  ```

* `qstat` JSON is invalid pbstop auto‑repairs common breakages and falls back to a text parser. Press `E` to inspect the last raw failure.

* No `%MEM` Provide `Resource_List.mem` or `select:...:mem=` in PBS, or define `queue_memory_defaults` in your config.
