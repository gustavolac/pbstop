#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
pbstop.py — top-like view for PBS on the head node.

Features
--------
- Real-time, low-overhead view of running (R) jobs.
- One qstat call per refresh (JSON with auto-repair; fallback to text parser).
- Sort by any column (lowercase = ascending, UPPERCASE = descending).
- Two toplines (à la `top`) with cluster-wide metrics:
    * Line 1: Running jobs, unique users, last update, poll interval, sort.
    * Line 2: CPU% EWMA (1/5/15m) and global CPU efficiency (non-EWMA).
- Columns: USER, EGROUP, JOBID, QUEUE, START, WALL, MEM (GiB), %MEM, CPU%, EFF%, HOSTS.
- WALL formatted right-aligned as "H:MM:SS" (no zero-padding for hours), width = 10.
- MEM printed with 'G' on the right edge of the cell.
- %MEM sorted with missing ('-') always last (for both asc/desc); m/M sorts by %MEM.
- HOSTS compacted (e.g., adano[01-03,12]), width default = 30.
- Low CPU% (<20) dimmed to grey (unless selected).
- Detail pane (Enter): shows `qstat -f` plus script path (PBS_O_WORKDIR + Submit_arguments),
  Start and Submitted timestamps in YYYY-MM-DD-HH:MM.

Configurability
--------------
- All environment-specific policies live in a JSON config file:
  memory fallback per queue, host-compaction on/off, poll interval, columns & widths,
  narrow-mode hiding order, dimming threshold, EWMA windows, default sort, etc.
- Defaults are portable; see `config.default.json`.
- Lovelace example: `config.lovelace.json`.

Usage
-----
    ./pbstop.py                 # uses defaults or PBSTOP_CONFIG env var
    ./pbstop.py --config path/to/config.json
    ./pbstop.py 60              # override poll seconds
    ./pbstop.py 60 /opt/pbs/bin/qstat --config config.lovelace.json

Keys
----
  Navigation:  ↑/↓, PgUp/PgDn, g/G, Enter (job details)
  Sort:        e/E (EFF%), c/C (CPU% now), m/M (%MEM), w/W (WALL),
               u/U (USER), j/J (JOBID), o/O (QUEUE), s/S (START)
               (lowercase=asc, UPPERCASE=desc)
  Filter:      f (user/queue regex), / (search substring), a (clear)
  Misc:        r (refresh), t (change poll sec), x (export CSV),
               E (last raw error), ? (help), q (quit)

Requirements
------------
- Python ≥ 3.8 with `curses`. No third-party deps.

Author: You + ChatGPT (GPT-5 Pro)
"""

from __future__ import annotations

import shlex
import curses
import curses.ascii
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# -------------------------
# Configuration management
# -------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "poll_seconds": 120,
    "qstat_candidates": ["/opt/pbs/bin/qstat", "/usr/bin/qstat", "/usr/local/bin/qstat", "qstat"],
    "columns": ["user", "egroup", "jobid", "queue", "start", "wall", "mem", "mem_pct", "cpu_now", "cpu_eff", "hosts"],
    "column_widths": {
        "user": 10, "egroup": 8, "jobid": 6, "queue": 10,
        "start": 14, "wall": 10, "mem": 7, "mem_pct": 5,
        "cpu_now": 6, "cpu_eff": 6, "hosts": 30
    },
    "column_mins": {
        "user": 4, "egroup": 4, "jobid": 4, "queue": 6,
        "start": 14, "wall": 10, "mem": 5, "mem_pct": 4,
        "cpu_now": 5, "cpu_eff": 5, "hosts": 8
    },
    "narrow_hide_order": ["hosts", "mem_pct", "egroup"],
    "ui": {
        "dim_cpu_below_pct": 20.0,
        "ellipsis": "...",
        "avoid_lr_corner": True
    },
    "format": {
        "wall_right_align": True,  # H:MM:SS aligned right
        "wall_width": 10,
        "host_compaction": True
    },
    "sort": {"key": "cpu_eff", "ascending": True},
    "ewma": {
        "windows_seconds": [60, 300, 900],  # 1/5/15 min
        "metric": "cpu_now_pct"             # EWMA of CPU% (not CPUeff)
    },
    "egroup_attr_order": ["egroup", "group_list", "group"],
    "queue_memory_defaults": {
        # Fallbacks: per-node GiB and optional fixed nodes count.
        # Example portable defaults (empty map means "no fallback policies").
    }
}

CFG: Dict[str, Any] = {}

def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out

def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    env_path = os.environ.get("PBSTOP_CONFIG")
    use_path = path or env_path
    if use_path and os.path.exists(use_path):
        try:
            with open(use_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            cfg = deep_update(cfg, user_cfg)
        except Exception as e:
            sys.stderr.write(f"[WARN] Failed to load config '{use_path}': {e}\n")
    return cfg

# -------------------------
# Utilities
# -------------------------

class PbsError(Exception):
    def __init__(self, message: str, raw_output: Optional[str] = None, cause: Optional[Exception] = None):
        super().__init__(message)
        self.raw_output = raw_output
        self.cause = cause

def safe_decode(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        try:
            return b.decode("latin-1")
        except UnicodeDecodeError:
            return b.decode("utf-8", errors="surrogateescape")

def safe_addnstr(win, y, x, s, width, attr=0, avoid_lr_corner=False):
    """Robust addnstr: clip to width, optionally avoid bottom-right cell, swallow curses.error."""
    try:
        h, w = win.getmaxyx()
        if y < 0 or y >= h or x < 0 or x >= w:
            return
        maxw = min(max(0, width), w - x)
        if avoid_lr_corner and y == h - 1 and x + maxw >= w:
            maxw = max(0, maxw - 1)
        if maxw <= 0:
            return
        buf = s.ljust(maxw)[:maxw]
        try:
            win.addnstr(y, x, buf, maxw, attr)
        except curses.error:
            if maxw > 1:
                win.addnstr(y, x, buf[:maxw - 1], maxw - 1, attr)
    except Exception:
        pass

def which_qstat(candidates: List[str]) -> str:
    for cand in candidates:
        path = shutil.which(cand) if cand == "qstat" else (cand if os.path.exists(cand) else None)
        if path:
            return path
    return "qstat"

def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v

def now_ts() -> int:
    return int(time.time())

def parse_hms_to_seconds(hms: str) -> int:
    """Parse 'HH:MM:SS' (HH can be >24). Also accepts 'MM:SS' or 'SS'."""
    if not hms:
        return 0
    parts = [p for p in hms.strip().split(":") if p != ""]
    try:
        if len(parts) == 3:
            h, m, s = (int(parts[0]), int(parts[1]), int(parts[2]))
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = (int(parts[0]), int(parts[1]))
            return m * 60 + s
        elif len(parts) == 1:
            return int(parts[0])
    except ValueError:
        pass
    return 0

def fmt_wall_right(seconds: int, width: int = 10) -> str:
    """Format "H:MM:SS" (hours no left zero) right-aligned to `width`."""
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    core = f"{h}:{m:02d}:{s:02d}"
    return core[-width:].rjust(width) if len(core) > width else core.rjust(width)

_UNIT_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)([kmgtp]?b)?\s*$", re.IGNORECASE)

def parse_mem_to_gib(s: str) -> float:
    """'1523456kb'|'12gb'|'0b' -> GiB."""
    if not s:
        return 0.0
    m = _UNIT_RE.match(s)
    if not m:
        return 0.0
    val = float(m.group(1))
    unit = (m.group(2) or "b").lower()
    factor = {"b":1.0, "kb":1024.0, "mb":1024.0**2, "gb":1024.0**3, "tb":1024.0**4, "pb":1024.0**5}.get(unit, 1.0)
    return (val * factor) / (1024.0**3)

def fmt_gib_right(gib: float, width: int) -> str:
    """GiB with 1 decimal; 'G' anchored on the right border of the cell."""
    if gib < 0:
        gib = 0.0
    core = "999+" if gib >= 999.95 else f"{gib:>5.1f}".strip()
    wnum = max(0, width - 1)
    core = core[-wnum:].rjust(wnum)
    return core + "G"

def fmt_pct_right(val: Optional[float], width: int) -> str:
    """Integer percent with '%' at the right edge; '-' when None."""
    if val is None:
        return "-".rjust(width)
    wnum = max(0, width - 1)
    core = f"{int(round(val)):d}"[-wnum:].rjust(wnum)
    return core + "%"

def ellipsize(s: str, width: int, ellipsis: str = "...") -> str:
    if width <= 0:
        return ""
    if len(s) <= width:
        return s.ljust(width)
    if width <= len(ellipsis):
        return ellipsis[:width]
    return s[: width - len(ellipsis)] + ellipsis

def parse_stime_to_epoch(stime: str) -> Optional[int]:
    """Example: 'Mon Aug 26 17:56:38 2024' -> epoch (localtime)."""
    if not stime:
        return None
    for fmt in ("%a %b %d %H:%M:%S %Y", "%a %b %d %H:%M:%S %Z %Y"):
        try:
            return int(time.mktime(time.strptime(stime, fmt)))
        except Exception:
            continue
    return None

def fmt_dt14(ts: Optional[int]) -> str:
    """yy-mm-dd-hh:mm (14 chars)."""
    if not ts:
        return "-- -- -- --".replace(" ", "")
    return time.strftime("%y-%m-%d-%H:%M", time.localtime(ts))

def fmt_dt_full(ts: Optional[int]) -> str:
    """YYYY-MM-DD-HH:MM."""
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d-%H:%M", time.localtime(ts))

def s_int(x: Any, default: int = 0) -> int:
    try: return int(x)
    except Exception: return default

def s_float(x: Any, default: float = 0.0) -> float:
    try: return float(x)
    except Exception: return default

# -------------------------
# NCPUS / nodes / mem req
# -------------------------

def parse_ncpus_from_rl(rl: Dict[str, Any]) -> int:
    if not rl:
        return 1
    if rl.get("ncpus") is not None:
        return max(1, s_int(rl["ncpus"], 1))
    sel = str(rl.get("select") or "").strip()
    if not sel:
        return 1
    total = 0
    for seg in sel.split("+"):
        seg = seg.strip()
        if not seg: continue
        count = 1
        parts = seg.split(":")
        if parts and parts[0].isdigit():
            count = int(parts[0]); kvs = parts[1:]
        else:
            kvs = parts
        kv: Dict[str, str] = {}
        for p in kvs:
            if "=" in p:
                k, v = p.split("=", 1)
                kv[k.strip()] = v.strip()
        n_this = s_int(kv.get("ncpus") or kv.get("mpiprocs") or 1, 1)
        total += count * n_this
    return total if total > 0 else 1

def parse_nodes_from_rl(rl: Dict[str, Any]) -> Optional[int]:
    if not rl: return None
    sel = str(rl.get("select") or "").strip()
    if not sel: return None
    total = 0
    for seg in sel.split("+"):
        seg = seg.strip()
        if not seg: continue
        parts = seg.split(":")
        if parts and parts[0].isdigit():
            total += int(parts[0])
        else:
            total += 1
    return total if total > 0 else None

# -------------------------
# Hosts compaction
# -------------------------

_HOST_SPLIT_RE = re.compile(r"^(.+?)(\d+)$")

def extract_hosts(exec_host: str) -> List[str]:
    if not exec_host:
        return []
    hosts = []
    for part in exec_host.split("+"):
        part = part.strip()
        if not part: continue
        host = part.split("/")[0]
        if host and host not in hosts:
            hosts.append(host)
    return hosts

def compact_hosts(hosts: List[str], max_width: int, ellipsis: str = "...") -> str:
    if not hosts:
        return "-".rjust(max_width)
    groups: Dict[str, List[str]] = {}
    for h in hosts:
        m = _HOST_SPLIT_RE.match(h)
        pref, num = (m.group(1), m.group(2)) if m else (h, None)
        groups.setdefault(pref, []).append(num)
    parts = []
    for pref, nums in sorted(groups.items(), key=lambda kv: kv[0]):
        if nums and all(n is not None for n in nums):
            nums_clean = sorted(set(nums))
            width = max(len(n) for n in nums_clean)
            ns = sorted(int(n) for n in nums_clean)
            ranges = []
            start = prev = ns[0]
            for x in ns[1:]:
                if x == prev + 1:
                    prev = x; continue
                if start == prev:
                    ranges.append(f"{start:0{width}d}")
                else:
                    ranges.append(f"{start:0{width}d}-{prev:0{width}d}")
                start = prev = x
            if start == prev:
                ranges.append(f"{start:0{width}d}")
            else:
                ranges.append(f"{start:0{width}d}-{prev:0{width}d}")
            part = f"{pref}{','.join(ranges)}"
        else:
            part = pref
        parts.append(part)
    s = "+".join(parts)
    return s.ljust(max_width) if len(s) <= max_width else ellipsize(s, max_width, ellipsis)

# -------------------------
# Data models
# -------------------------

@dataclass
class JobRow:
    user: str
    egroup: str
    jobid: str
    queue: str
    start_ts: Optional[int]
    wall_s: int
    mem_gib: float
    cpu_now_pct: float
    cpu_eff_pct: float
    ncpus: int
    mem_req_gib: Optional[float]
    mem_pct: Optional[float]
    hosts_raw: str
    vmem_gib: Optional[float]
    raw: Dict[str, Any] = field(default_factory=dict)

# -------------------------
# PBS client
# -------------------------

class PbsClient:
    def __init__(self, qstat_path: str, timeout_secs: int = 30):
        self.qstat_path = qstat_path
        self.timeout_secs = max(5, timeout_secs)
        self.last_raw_error: Optional[str] = None

    def fetch_jobs(self) -> List[Dict[str, Any]]:
        try:
            return self._fetch_jobs_json()
        except PbsError as e_json:
            self.last_raw_error = e_json.raw_output or str(e_json)
            try:
                return self._fetch_jobs_text()
            except PbsError as e_text:
                combined = f"[JSON FAIL] {str(e_json)}\n\n[TEXT FAIL] {str(e_text)}"
                raw = (e_json.raw_output or "") + "\n\n---\n\n" + (e_text.raw_output or "")
                raise PbsError(combined, raw_output=raw, cause=e_text)

    def _fetch_jobs_json(self) -> List[Dict[str, Any]]:
        cmd = [self.qstat_path, "-f", "-F", "json"]
        try:
            env = os.environ.copy(); env.setdefault("LC_ALL","C"); env.setdefault("LANG","C")
            raw_bytes = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=self.timeout_secs, env=env)
            out = safe_decode(raw_bytes)
        except subprocess.CalledProcessError as e:
            raise PbsError(f"qstat JSON failed (exit {e.returncode}).",
                           raw_output=safe_decode(e.output) if e.output else None, cause=e)
        except subprocess.TimeoutExpired as e:
            raise PbsError(f"qstat JSON timeout after {self.timeout_secs}s.", raw_output=str(e), cause=e)
        raw = out.strip()

        try:
            data = json.loads(raw)
        except Exception:
            try:
                fixed = self._repair_broken_json(raw)
                data = json.loads(fixed)
            except Exception as e2:
                raise PbsError("Invalid qstat JSON and repair failed.", raw_output=raw, cause=e2)

        jobs = data.get("Jobs") or data.get("jobs") or {}
        if not isinstance(jobs, dict):
            raise PbsError("Unexpected qstat JSON structure.", raw_output=raw)

        norm: List[Dict[str, Any]] = []
        for jid_full, j in jobs.items():
            if not isinstance(j, dict): continue
            if (j.get("job_state") or "").strip() != "R": continue

            jobid  = str(j.get("Job_Id") or jid_full).split(".")[0]
            owner  = str(j.get("Job_Owner") or j.get("job_owner") or "")
            user   = owner.split("@")[0] if "@" in owner else owner

            egroup = "-"
            for attr in CFG.get("egroup_attr_order", ["egroup","group_list","group"]):
                val = j.get(attr) or ""
                if val:
                    egroup = str(val).split("@")[0]
                    break

            queue  = str(j.get("queue") or j.get("Queue") or "")
            ru     = j.get("resources_used") or j.get("Resources_Used") or {}
            rl     = j.get("Resource_List") or {}
            stime  = str(j.get("stime") or j.get("Start_Time") or "")
            qtime  = str(j.get("qtime") or "")
            exec_host = str(j.get("exec_host") or "")
            exec_vnode = str(j.get("exec_vnode") or "")
            var_list = str(j.get("Variable_List") or "")
            submit_args = str(j.get("Submit_arguments") or j.get("submit_arguments") or "")

            norm.append({
                "jobid": jobid, "user": user, "egroup": egroup, "queue": queue,
                "resources_used": ru, "Resource_List": rl, "stime": stime, "qtime": qtime,
                "exec_host": exec_host, "exec_vnode": exec_vnode,
                "Variable_List": var_list, "Submit_arguments": submit_args,
                "job_state": "R",
            })
        return norm

    def _repair_broken_json(self, s: str) -> str:
        s2 = "".join(ch for ch in s if (ord(ch) >= 32 or ch in "\n\t\r"))
        s2 = re.sub(r",\s*([}\]])", r"\1", s2)
        s2 = re.sub(
            r"(?P<pre>[^\\])'(Jobs|jobs|Resources_Used|resources_used|Resource_List|Job_Id|job_state|queue|Queue|Job_Owner|job_owner|exec_host|exec_vnode|Variable_List|Submit_arguments|submit_arguments|egroup|group_list|group|qtime|stime)'(?P<post>\s*:)",
            r'\g<pre>"\2"\g<post>', s2)
        def _esc(m: re.Match) -> str:
            inner = m.group(1).replace("\\", "\\\\").replace("\n","\\n").replace("\r","\\r").replace("\t","\\t")
            return '"' + inner + '"'
        s2 = re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', _esc, s2)
        return s2

    def _fetch_jobs_text(self) -> List[Dict[str, Any]]:
        cmd = [self.qstat_path, "-f"]
        try:
            env = os.environ.copy(); env.setdefault("LC_ALL","C"); env.setdefault("LANG","C")
            raw_bytes = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=self.timeout_secs, env=env)
            out = safe_decode(raw_bytes)
        except subprocess.CalledProcessError as e:
            raise PbsError(f"qstat text failed (exit {e.returncode}).",
                           raw_output=safe_decode(e.output) if e.output else None, cause=e)
        except subprocess.TimeoutExpired as e:
            raise PbsError(f"qstat text timeout after {self.timeout_secs}s.", raw_output=str(e), cause=e)

        blocks: List[str] = []
        cur: List[str] = []
        for line in out.splitlines():
            if line.startswith("Job Id:"):
                if cur: blocks.append("\n".join(cur)); cur = []
            cur.append(line.rstrip("\n"))
        if cur: blocks.append("\n".join(cur))

        norm: List[Dict[str, Any]] = []
        for b in blocks:
            jobid_full = ""
            attrs: Dict[str, Any] = {}
            for line in b.splitlines():
                if line.startswith("Job Id:"):
                    jobid_full = line.split(":", 1)[1].strip(); continue
                if " = " in line:
                    k, v = line.split(" = ", 1)
                    attrs[k.strip()] = v.strip()
            if not jobid_full or attrs.get("job_state") != "R":
                continue

            jobid = jobid_full.split(".")[0]
            owner = attrs.get("Job_Owner") or attrs.get("job_owner") or ""
            user = owner.split("@")[0] if "@" in owner else owner

            egroup = "-"
            for attr in CFG.get("egroup_attr_order", ["egroup","group_list","group"]):
                val = attrs.get(attr) or ""
                if val:
                    egroup = str(val).split("@")[0]
                    break

            queue = attrs.get("queue") or attrs.get("Queue") or ""
            ru: Dict[str, Any] = {}
            rl: Dict[str, Any] = {}
            for k, v in attrs.items():
                if k.startswith("resources_used."):
                    ru[k[len("resources_used."):]] = v
                elif k.startswith("Resource_List."):
                    rl[k[len("Resource_List."):]] = v

            stime = attrs.get("stime") or attrs.get("Start_Time") or ""
            qtime = attrs.get("qtime") or ""
            exec_host = attrs.get("exec_host") or ""
            exec_vnode = attrs.get("exec_vnode") or ""
            var_list = attrs.get("Variable_List") or ""
            submit_args = attrs.get("Submit_arguments") or attrs.get("submit_arguments") or ""

            norm.append({
                "jobid": jobid, "user": user, "egroup": egroup, "queue": queue,
                "resources_used": ru, "Resource_List": rl, "stime": stime, "qtime": qtime,
                "exec_host": exec_host, "exec_vnode": exec_vnode,
                "Variable_List": var_list, "Submit_arguments": submit_args,
                "job_state": "R",
            })
        return norm

# -------------------------
# Row builder
# -------------------------

def fallback_mem_req_from_queue(queue: str, rl: Dict[str, Any], exec_host: str) -> Optional[float]:
    """Policy: derive mem_req when not available in Resource_List using config queue rules."""
    rules = CFG.get("queue_memory_defaults", {})
    rule = rules.get(queue)
    if not rule:
        return None
    per_node = float(rule.get("per_node_gib", 0.0))
    if per_node <= 0:
        return None
    fixed_nodes = rule.get("nodes")  # optional
    if fixed_nodes is not None:
        nodes = int(fixed_nodes)
    else:
        nodes = parse_nodes_from_rl(rl) or len(set(extract_hosts(exec_host))) or 1
    return per_node * max(1, nodes)

def make_job_row(rec: Dict[str, Any], ts: Optional[int] = None) -> JobRow:
    if ts is None:
        ts = now_ts()

    user   = str(rec.get("user") or "")
    egroup = str(rec.get("egroup") or "-")
    jobid  = str(rec.get("jobid") or "")
    queue  = str(rec.get("queue") or "")

    ru = rec.get("resources_used") or {}
    rl = rec.get("Resource_List") or {}

    ncpus = parse_ncpus_from_rl(rl) or 1

    stime_s = str(rec.get("stime") or "")
    start_ts = parse_stime_to_epoch(stime_s)
    if start_ts is None:
        wt = s_int(ru.get("walltime"), 0)
        start_ts = ts - wt if wt > 0 else None

    wall_s = s_int(ru.get("walltime"), 0)
    if wall_s <= 0 and start_ts:
        wall_s = max(0, ts - start_ts)

    mem_gib = parse_mem_to_gib(str(ru.get("mem") or ""))
    vmem_gib = parse_mem_to_gib(str(ru.get("vmem") or "")) if "vmem" in ru else None

    cpupercent_total = s_float(ru.get("cpupercent"), 0.0)
    cpu_now_pct = clamp(cpupercent_total / ncpus, 0.0, 100.0) if ncpus > 0 else 0.0
    cput_s = parse_hms_to_seconds(str(ru.get("cput") or "0"))
    denom = wall_s * ncpus
    cpu_eff_pct = clamp((cput_s / denom) * 100.0 if denom > 0 else 0.0, 0.0, 100.0)

    # mem_req:
    mem_req_gib: Optional[float] = None
    if rl.get("mem"):
        mem_req_gib = parse_mem_to_gib(str(rl.get("mem")))
    if mem_req_gib is None:
        sel = str(rl.get("select") or "").strip()
        if sel:
            total_gib = 0.0; found = False
            for seg in sel.split("+"):
                seg = seg.strip()
                if not seg: continue
                count = 1
                parts = seg.split(":")
                if parts and parts[0].isdigit():
                    count = int(parts[0]); kvs = parts[1:]
                else:
                    kvs = parts
                kv: Dict[str, str] = {}
                for p in kvs:
                    if "=" in p:
                        k,v = p.split("=",1)
                        kv[k.strip()] = v.strip()
                if "mem" in kv:
                    total_gib += count * parse_mem_to_gib(kv["mem"])
                    found = True
            if found and total_gib > 0:
                mem_req_gib = total_gib
    if mem_req_gib is None:
        mem_req_gib = fallback_mem_req_from_queue(queue, rl, str(rec.get("exec_host") or ""))

    mem_pct = (mem_gib / mem_req_gib * 100.0) if (mem_req_gib and mem_req_gib > 0) else None

    return JobRow(
        user=user, egroup=egroup, jobid=jobid, queue=queue,
        start_ts=start_ts, wall_s=wall_s,
        mem_gib=mem_gib, cpu_now_pct=cpu_now_pct, cpu_eff_pct=cpu_eff_pct,
        ncpus=ncpus, mem_req_gib=mem_req_gib, mem_pct=mem_pct,
        hosts_raw=str(rec.get("exec_host") or ""), vmem_gib=vmem_gib, raw=rec
    )

# -------------------------
# Table model
# -------------------------

class TableModel:
    def __init__(self):
        self.rows: List[JobRow] = []
        self.filter_user: Optional[str] = None
        self.filter_queue: Optional[str] = None
        self.search_substr: Optional[str] = None
        self.sort_key = CFG.get("sort", {}).get("key", "cpu_eff")
        self.sort_asc = bool(CFG.get("sort", {}).get("ascending", True))

    def set_rows(self, rows: List[JobRow]) -> None:
        self.rows = rows

    def _key_missing_last(self, value: Optional[float], asc: bool) -> Tuple[int, float]:
        missing = 1 if value is None else 0
        v = 0.0 if value is None else float(value)
        return (missing, v if asc else -v)

    def current_view(self) -> List[JobRow]:
        out = self.rows
        if self.filter_user:
            rx = re.compile(self.filter_user, re.IGNORECASE)
            out = [r for r in out if rx.search(r.user)]
        if self.filter_queue:
            rx = re.compile(self.filter_queue, re.IGNORECASE)
            out = [r for r in out if rx.search(r.queue)]
        if self.search_substr:
            s = self.search_substr.lower()
            out = [r for r in out if (s in r.user.lower() or s in r.jobid.lower() or s in r.queue.lower())]

        if self.sort_key == "mem_pct":
            out.sort(key=lambda r: self._key_missing_last(r.mem_pct, self.sort_asc))
            return out

        keyfunc = {
            "user":   lambda r: r.user,
            "egroup": lambda r: r.egroup,
            "jobid":  lambda r: (s_int(r.jobid, 0), r.jobid),
            "queue":  lambda r: r.queue,
            "start":  lambda r: (r.start_ts or 0),
            "wall":   lambda r: r.wall_s,
            "mem":    lambda r: r.mem_gib,
            "cpu_now":lambda r: r.cpu_now_pct,
            "cpu_eff":lambda r: r.cpu_eff_pct,
            "hosts":  lambda r: r.hosts_raw,
        }.get(self.sort_key, lambda r: r.cpu_eff_pct)

        out.sort(key=keyfunc, reverse=not self.sort_asc)
        return out

    def set_sort(self, key: str, asc: bool):
        self.sort_key = key
        self.sort_asc = asc

# -------------------------
# EWMA for CPU% (1/5/15m)
# -------------------------

class EwmaTracker:
    def __init__(self, windows: List[int]):
        self.windows = windows[:]  # in seconds
        self.last_ts: Optional[int] = None
        self.values_map: Dict[int, float] = {w: 0.0 for w in self.windows}
        self.inited = False

    def update(self, cpu_now_frac: float, now: int):
        if not self.inited:
            for w in self.windows:
                self.values_map[w] = cpu_now_frac
            self.last_ts = now
            self.inited = True
            return
        dt = max(1, now - (self.last_ts or now))
        for w in self.windows:
            alpha = 1.0 - math.exp(-dt / float(w))
            old = self.values_map[w]
            self.values_map[w] = (1.0 - alpha) * old + alpha * cpu_now_frac
        self.last_ts = now

    def values(self) -> List[float]:
        return [self.values_map[w] for w in self.windows]

def _fmt_hms(seconds: int) -> str:
    """H:MM:SS with no left-zero for hours (not width-fixed)."""
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"

def print_snapshot(client: PbsClient, cfg: Dict[str, Any], qstat_path: str, fmt: str = "csv") -> None:
    """
    One-shot snapshot to STDOUT (no curses). Includes global metrics + per-job table.
    fmt in {"csv","tsv","table"}.
    """
    # fetch once
    recs = client.fetch_jobs()
    rows: List[JobRow] = []
    jobs_extra: List[Dict[str, Any]] = []
    for r in recs:
        jr = make_job_row(r)
        ru = r.get("resources_used") or {}
        cpupercent_raw = s_float(ru.get("cpupercent"), 0.0)         # as reported by PBS
        cput_s = parse_hms_to_seconds(str(ru.get("cput") or "0"))   # seconds
        jobs_extra.append({
            "cpupercent_raw": cpupercent_raw,
            "cput_s": cput_s,
            "wall_s": jr.wall_s,
        })
        rows.append(jr)

    # globals (weighted)
    sum_ncpus = sum(r.ncpus for r in rows) or 1
    cpu_now_global_pct = (sum((r.cpu_now_pct) * r.ncpus for r in rows) / sum_ncpus) if rows else 0.0
    num_cput = sum(j["cput_s"] for j in jobs_extra)
    den_wallxcpu = sum(j["wall_s"] * r.ncpus for j, r in zip(jobs_extra, rows)) or 1
    cpu_eff_global_pct = 100.0 * clamp(num_cput / den_wallxcpu, 0.0, 1.0)

    # meta lines
    snapshot_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts()))
    n_jobs = len(rows)
    n_users = len({r.user for r in rows})
    poll = cfg.get("poll_seconds", 120)

    # choose separator / printer
    sep = "," if fmt == "csv" else ("\t" if fmt == "tsv" else None)

    # meta
    if fmt in ("csv", "tsv"):
        print(f"metric{sep}value")
        print(f"snapshot_time{sep}{snapshot_time}")
        print(f"jobs_running{sep}{n_jobs}")
        print(f"users_running{sep}{n_users}")
        print(f"sum_ncpus{sep}{sum_ncpus}")
        print(f"cpu_now_global_pct{sep}{cpu_now_global_pct:.2f}")
        print(f"cpu_eff_global_pct{sep}{cpu_eff_global_pct:.2f}")
        print(f"poll_seconds{sep}{poll}")
        print("")  # blank line before table
    else:
        # pretty table header lines
        meta_lines = [
            f"SNAPSHOT   : {snapshot_time}",
            f"RUNNING    : jobs={n_jobs} users={n_users}",
            f"RESOURCES  : sum_ncpus={sum_ncpus}",
            f"CPU NOW    : {cpu_now_global_pct:5.2f} %",
            f"CPU EFF    : {cpu_eff_global_pct:5.2f} %",
            f"POLL       : {poll}s",
        ]
        for ln in meta_lines:
            print(ln)
        print("")

    # job table columns
    headers = [
        "user","egroup","jobid","queue",
        "ncpus","cpupercent_raw","cpu_now_pct",
        "cput_s","cput_hms","wall_s","wall_hms",
        "eff_pct","mem_gib","mem_req_gib","mem_pct",
        "start_ymdhm","hosts"
    ]

    # build rows
    table: List[List[str]] = []
    for jr, extra in zip(rows, jobs_extra):
        start_str = fmt_dt14(jr.start_ts)
        hosts_compact = compact_hosts(extract_hosts(jr.hosts_raw), 64).strip() if cfg.get("format",{}).get("host_compaction", True) else "+".join(extract_hosts(jr.hosts_raw))
        eff_pct = 100.0 * clamp((extra["cput_s"] / (extra["wall_s"] * jr.ncpus)) if (extra["wall_s"] * jr.ncpus) > 0 else 0.0, 0.0, 1.0)
        row = [
            jr.user,
            jr.egroup,
            jr.jobid,
            jr.queue,
            str(jr.ncpus),
            f"{extra['cpupercent_raw']:.3f}",
            f"{jr.cpu_now_pct:.3f}",
            str(extra["cput_s"]),
            _fmt_hms(extra["cput_s"]),
            str(extra["wall_s"]),
            _fmt_hms(extra["wall_s"]),
            f"{eff_pct:.3f}",
            f"{jr.mem_gib:.3f}",
            "" if jr.mem_req_gib is None else f"{jr.mem_req_gib:.3f}",
            "" if jr.mem_pct is None else f"{jr.mem_pct:.3f}",
            start_str,
            hosts_compact
        ]
        table.append(row)

    if fmt in ("csv", "tsv"):
        print(sep.join(headers))
        for row in table:
            print(sep.join(row))
    else:
        # simple aligned table
        widths = [max(len(h), *(len(r[i]) for r in table)) for i, h in enumerate(headers)]
        line = " ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        print(line)
        print("-" * len(line))
        for row in table:
            print(" ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
# -------------------------
# UI (curses)
# -------------------------

HELP_TEXT = """
Keys:
  Navigation: ↑/↓, PgUp/PgDn, g/G (top/bottom), Enter (job details)
  Sort:       e/E (EFF%), c/C (CPU% now), m/M (%MEM), w/W (WALL),
              u/U (USER), j/J (JOBID), o/O (QUEUE), s/S (START)
              (lowercase=ascending, UPPERCASE=descending)
  Filter:     f (user/queue regex), / (substring search), a (clear filters)
  Misc:       r (refresh), t (change poll sec), x (export CSV),
              E (show last raw error), ? (help), q (quit)
Note: rows with CPU% now < dim-threshold appear dimmed (grey).
"""

class CursesApp:
    def __init__(self, client: PbsClient, refresh_secs: int):
        self.client = client
        self.model = TableModel()
        self.refresh_secs = max(1, refresh_secs)
        self.last_fetch_ok_ts: Optional[int] = None
        self.last_fetch_err: Optional[str] = None
        self.last_raw_error: Optional[str] = None
        self.selection = 0
        self.top_idx = 0
        self.export_counter = 0
        # State for CPU% by Δcput/Δt
        self._prev_job_state: Dict[str, Tuple[int, int, int]] = {}  # jobid -> (cput_s, wall_s, ncpus)
        self.ewma = EwmaTracker(CFG.get("ewma", {}).get("windows_seconds", [60, 300, 900]))
        self.cpu_eff_global_pct: float = 0.0  # shown in topline
        self._pair_sel = None  # color pair id for selection (set in _main)

    def run(self):
        curses.wrapper(self._main)

    def _main(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(200)

        # ---- robust color init ----
        self._has_colors = False
        self._c_updated = curses.A_NORMAL
        self._c_now     = curses.A_NORMAL
        self._c_sort    = curses.A_NORMAL

        try:
            curses.start_color()
            if curses.has_colors():
                # tente usar cor de fundo "default" (-1); se falhar, caia p/ preto (0)
                try:
                    curses.use_default_colors()
                    bg = -1
                except curses.error:
                    bg = 0

                def initp(idx, fg):
                    try:
                        curses.init_pair(idx, fg, bg)
                        return True
                    except curses.error:
                        # fallback: fundo preto
                        try:
                            curses.init_pair(idx, fg, 0)
                            return True
                        except curses.error:
                            return False
                ok1 = initp(1, curses.COLOR_GREEN)   # Updated
                ok2 = initp(2, curses.COLOR_CYAN)    # Now (clock)
                ok3 = initp(3, curses.COLOR_YELLOW)  # Sort
# seleção: fg preto (ou branco) em fundo azul, com fallback
                sel_ok = False
                try:
                    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_BLUE)
                    sel_ok = True
                except curses.error:
                    try:
                        curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLUE)
                        sel_ok = True
                    except curses.error:
                        sel_ok = False
                if sel_ok:
                    self._pair_sel = 4

                if ok1 and ok2 and ok3:
                    self._has_colors = True
                    self._c_updated = curses.color_pair(1)
                    self._c_now     = curses.color_pair(2)
                    self._c_sort    = curses.color_pair(3)
        except curses.error:
            # sem cores; segue em monocromático
            self._has_colors = False

        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        next_deadline = 0
        while True:
            h, w = stdscr.getmaxyx()
            now = now_ts()
            if now >= next_deadline:
                self._refresh_data()
                next_deadline = now + self.refresh_secs
            self._render(stdscr)
            ch = stdscr.getch()
            if ch != -1:
                if not self._handle_key(ch, stdscr):
                    break

    def _refresh_data(self):
        try:
            timeout = min(60, max(5, int(self.refresh_secs * 0.8)))
            self.client.timeout_secs = timeout
            recs = self.client.fetch_jobs()
            rows = [make_job_row(r) for r in recs]

            # ---------- Compute CPU% per job via Δcput/Δt (fallback to cpupercent) ----------
            use_clip = CFG.get("cpu_now", {}).get("clip_0_100", True)
            divide_by_n = bool(CFG.get("cpu_now", {}).get("divide_by_ncpus", True))
            source = (CFG.get("cpu_now", {}).get("source") or "cput_delta").lower()

            new_state: Dict[str, Tuple[int, int, int]] = {}
            for r in rows:
                # Gather present values
                ru = r.raw.get("resources_used") or {}
                cput_s_now = parse_hms_to_seconds(str(ru.get("cput") or "0"))
                wall_s_now = int(r.wall_s)
                n_now = max(1, int(r.ncpus))
                new_state[r.jobid] = (cput_s_now, wall_s_now, n_now)

                cpu_now_pct = None

                if source == "cput_delta":
                    prev = self._prev_job_state.get(r.jobid)
                    if prev:
                        cput_prev, wall_prev, _ = prev
                        dcput = cput_s_now - cput_prev
                        dwall = wall_s_now - wall_prev
                        if dcput >= 0 and dwall > 0 and n_now > 0:
                            cpu_now_pct = 100.0 * (dcput / (dwall * n_now))
                    # fallback if first sample or invalid delta
                    if cpu_now_pct is None:
                        cpupercent_total = s_float(ru.get("cpupercent"), 0.0)
                        if divide_by_n and n_now > 0:
                            cpu_now_pct = cpupercent_total / n_now
                        else:
                            cpu_now_pct = cpupercent_total  # already 0..100 per job (some PBS variants)

                else:  # source == "cpupercent"
                    cpupercent_total = s_float(ru.get("cpupercent"), 0.0)
                    if divide_by_n and n_now > 0:
                        cpu_now_pct = cpupercent_total / n_now
                    else:
                        cpu_now_pct = cpupercent_total

                # clamp for display
                if cpu_now_pct is None:
                    cpu_now_pct = 0.0
                if use_clip:
                    cpu_now_pct = clamp(cpu_now_pct, 0.0, 100.0)
                r.cpu_now_pct = cpu_now_pct  # override with delta based metric

            # Commit state for next delta
            self._prev_job_state = new_state

            # ---------- Aggregates ----------
            self.model.set_rows(rows)

            sum_ncpus = sum(r.ncpus for r in rows) or 1

            # CPU% NOW global (weighted by ncpus) — using the delta metric
            cpu_now_frac = (sum((r.cpu_now_pct/100.0) * r.ncpus for r in rows) / sum_ncpus) if rows else 0.0

            # Global CPU efficiency (non-EWMA): sum(cput) / sum(wall*ncpus)
            num = sum(parse_hms_to_seconds(str((r.raw.get("resources_used") or {}).get("cput") or "0")) for r in rows)
            den = sum(r.wall_s * r.ncpus for r in rows) or 1
            eff_frac = clamp(num / den, 0.0, 1.0)

            now = now_ts()
            # EWMA on CPU% (delta‑based)
            self.ewma.update(cpu_now_frac, now)
            self.cpu_eff_global_pct = eff_frac * 100.0

            self.last_fetch_ok_ts = now
            self.last_fetch_err = None
            self.last_raw_error = self.client.last_raw_error
        except PbsError as e:
            self.last_fetch_err = str(e)
            self.last_raw_error = e.raw_output


    def _render(self, stdscr):
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        self._render_toplines(stdscr, w)

        header = self._make_header_line(w)
        safe_addnstr(stdscr, 2, 0, header, w, curses.A_REVERSE)

        view = self.model.current_view()
        n = len(view)

        body_rows = max(0, h - 5)  # 2 toplines + header + status + footer
        self.selection = int(clamp(self.selection, 0, max(0, n - 1)))
        if self.selection < self.top_idx: self.top_idx = self.selection
        elif self.selection >= self.top_idx + body_rows: self.top_idx = self.selection - body_rows + 1
        self.top_idx = max(0, min(self.top_idx, max(0, n - body_rows)))

        colspec = self._compute_columns(w)

        dim_threshold = CFG.get("ui", {}).get("dim_cpu_below_pct", 20.0)

        for i in range(body_rows):
            y = 3 + i
            idx = self.top_idx + i
            if idx >= n: break
            row = view[idx]
            line = self._format_row(row, colspec)
            # base
            attr = curses.A_NORMAL
            # dim só quando NÃO selecionada
            if idx != self.selection and row.cpu_now_pct < dim_threshold:
                attr |= curses.A_DIM

            # seleção com bg azul (ou reverse se sem cores)
            if idx == self.selection:
                attr = curses.color_pair(self._pair_sel) if self._pair_sel else curses.A_REVERSE
            safe_addnstr(stdscr, y, 0, line, w, attr)

        status = self._make_status_line(w, n, colspec)
        safe_addnstr(stdscr, h - 2, 0, status, w, curses.A_REVERSE, avoid_lr_corner=CFG.get("ui", {}).get("avoid_lr_corner", True))
        foot = "[?] help  [Enter] details  [q] quit"
        safe_addnstr(stdscr, h - 1, 0, foot, w, curses.A_DIM, avoid_lr_corner=CFG.get("ui", {}).get("avoid_lr_corner", True))
        stdscr.refresh()

    def _wrap_text_lines(self, text: str, width: int, indent: int = 0) -> List[str]:
        import textwrap
        wrapped: List[str] = []
        for ln in (text or "").splitlines():
            if not ln:
                wrapped.append("")
                continue
            wrapped.extend(textwrap.wrap(ln, width=width, subsequent_indent=" " * indent,
                                         break_long_words=False, break_on_hyphens=False) or [""])
        return wrapped

    def _kv_inline_wrap(self, kv: List[Tuple[str, str]], width: int) -> List[str]:
        segs = [f"{k}={v}" for k, v in kv if v not in (None, "")]
        line = "  ".join(segs)
        return self._wrap_text_lines(line, width=width)

    def _modal_sections(self, stdscr, title: str, sections: List[Tuple[str, str]]):
        """
        sections: list of (label, text). Only non-empty texts are shown.
        Tabs: [1..N] or TAB/Shift+TAB to switch. Scroll with ↑/↓/PgUp/PgDn. q to close.
        """
        # filter empty
        tabs = [(lab, txt) for (lab, txt) in sections if (txt and txt.strip())]
        if not tabs:
            tabs = [("Empty", "")]
        sel = 0
        scroll = 0
        while True:
            h, w = stdscr.getmaxyx()
            win_h = min(h - 4, max(12, h - 6))
            win_w = min(w - 4, max(70, w - 6))
            y = h // 2 - win_h // 2
            x = w // 2 - win_w // 2
            win = curses.newwin(win_h, win_w, y, x)
            win.keypad(True)
            win.border()
            # Tabs bar
            tab_titles = []
            for i, (lab, _) in enumerate(tabs, 1):
                t = f"[{i}] {lab}"
                if i-1 == sel:
                    tab_titles.append(t)
                else:
                    tab_titles.append(t)
            tabs_line = "  ".join(tab_titles)
            win.addnstr(0, 2, f" {title} ", max(0, win_w - 4), curses.A_REVERSE)
            win.addnstr(1, 2, tabs_line[: max(0, win_w - 4)], max(0, win_w - 4), curses.A_BOLD)
            # Content
            inner_h = win_h - 4
            inner_w = win_w - 4
            _, txt = tabs[sel]
            # wrap content to inner width
            lines = self._wrap_text_lines(txt, width=inner_w)
            max_scroll = max(0, len(lines) - inner_h)
            scroll = max(0, min(scroll, max_scroll))
            for i in range(inner_h):
                row = lines[scroll + i] if (scroll + i) < len(lines) else ""
                win.addnstr(2 + i, 2, row.ljust(inner_w)[:inner_w], inner_w)
            # Footer help
            help_line = "[TAB] next  [Shift+TAB] prev  [1..9] jump  [↑/↓/PgUp/PgDn] scroll  [q] close"
            win.addnstr(win_h - 1, 2, help_line[: max(0, win_w - 4)], max(0, win_w - 4), curses.A_DIM)
            win.refresh()
            ch = win.getch()
            if ch in (ord('q'), ord('Q')):
                break
            elif ch in (curses.KEY_RIGHT, ord('\t')):  # TAB
                sel = (sel + 1) % len(tabs)
                scroll = 0
            elif ch == curses.KEY_BTAB:  # Shift+TAB
                sel = (sel - 1) % len(tabs)
                scroll = 0
            elif ord('1') <= ch <= ord('9'):
                idx = ch - ord('1')
                if idx < len(tabs):
                    sel = idx
                    scroll = 0
            elif ch in (curses.KEY_UP, ord('k')):
                scroll = max(0, scroll - 1)
            elif ch in (curses.KEY_DOWN, ord('j')):
                scroll = min(max_scroll, scroll + 1)
            elif ch == curses.KEY_PPAGE:
                scroll = max(0, scroll - inner_h)
            elif ch == curses.KEY_NPAGE:
                scroll = min(max_scroll, scroll + inner_h)
            elif ch == curses.KEY_RESIZE:
                continue
            # ignore others


    def _render_toplines(self, stdscr, w: int):
        rows = self.model.current_view()
        n_jobs = len(rows)
        n_users = len({r.user for r in rows})
        updated_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_fetch_ok_ts or now_ts()))
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts()))
        sort_map = {"cpu_eff":"EFF","cpu_now":"CPU","mem":"MEM","mem_pct":"%MEM","wall":"WALL",
                    "user":"USER","egroup":"EGROUP","jobid":"JOBID","queue":"QUEUE","hosts":"HOSTS","start":"START"}
        sort_txt = f"{sort_map.get(self.model.sort_key, self.model.sort_key)}{'↑' if self.model.sort_asc else '↓'}"

        # Line 1: R, Users, Updated (green), Now (cyan), Poll, Sort (yellow)
        segs1: List[Tuple[str,int]] = []
        segs1.append((f"R={n_jobs} | Users={n_users} | Updated=", curses.A_NORMAL))
        segs1.append((updated_str, self._c_updated))
        segs1.append((" | Now=", curses.A_NORMAL))
        segs1.append((now_str, self._c_now))
        segs1.append((f" | Poll={self.refresh_secs}s | Sort=", curses.A_NORMAL))
        segs1.append((sort_txt, self._c_sort))
        self._draw_segments(stdscr, 0, segs1, w)

        # Line 2: CPU% (1/5/15m) EWMA and CPUeff
        e_values = [v*100.0 for v in self.ewma.values()]  # CPU%_Δ EWMA 1/5/15m
        e_str = " ".join(f"{v:4.1f}%" for v in e_values)
        line2 = f"CPU% (1/5/15m)={e_str} | CPUeff={self.cpu_eff_global_pct:4.1f}%"
        safe_addnstr(stdscr, 1, 0, ellipsize(line2, w, CFG.get("ui", {}).get("ellipsis","...")), w)


    # ----- columns, layout -----

    def _choose_columns(self, width: int) -> List[str]:
        base = CFG.get("columns", DEFAULT_CONFIG["columns"])
        desired_total = sum(CFG["column_widths"].get(c, 6) for c in base) + (len(base)-1)
        if width >= desired_total:
            return base[:]
        cols = base[:]
        for c in CFG.get("narrow_hide_order", []):
            if c in cols:
                cols.remove(c)
                desired_total = sum(CFG["column_widths"].get(x, 6) for x in cols) + (len(cols)-1)
                if width >= desired_total:
                    return cols
        return cols

    def _compute_columns(self, width: int) -> List[Tuple[str, int]]:
        cols = self._choose_columns(width)
        cw = CFG["column_widths"]; cm = CFG["column_mins"]
        desired = [cw.get(k, 6) for k in cols]
        minimum = [cm.get(k, 4) for k in cols]
        sep_count = len(cols) - 1
        total_desired = sum(desired) + sep_count
        if width >= total_desired:
            final = desired[:]
        else:
            final = desired[:]
            excess = total_desired - width
            shrink_keys = ["user","queue","jobid","mem","hosts","egroup"]
            shrink_order = [cols.index(k) for k in shrink_keys if k in cols]
            for idx in shrink_order:
                can = final[idx] - minimum[idx]
                if can <= 0: continue
                take = min(can, excess)
                final[idx] -= take
                excess -= take
                if excess <= 0: break
            if excess > 0:
                final = minimum[:]
        return list(zip(cols, final))

    def _format_row(self, r: JobRow, colspec: List[Tuple[str, int]]) -> str:
        vals: Dict[str, str] = {
            "user":   r.user,
            "egroup": r.egroup,
            "jobid":  r.jobid,
            "queue":  r.queue,
            "start":  fmt_dt14(r.start_ts).ljust(CFG["column_widths"].get("start",14)),
            "wall":   "",
            "mem":    "",
            "mem_pct": fmt_pct_right(r.mem_pct, CFG["column_widths"].get("mem_pct",5)),
            "cpu_now": f"{r.cpu_now_pct:4.1f}",
            "cpu_eff": f"{r.cpu_eff_pct:4.1f}",
            "hosts":  "",
        }
        parts = []
        ell = CFG.get("ui", {}).get("ellipsis","...")
        for key, width in colspec:
            if key == "mem":
                parts.append(fmt_gib_right(r.mem_gib, width))
            elif key == "wall":
                parts.append(fmt_wall_right(r.wall_s, width))
            elif key == "hosts":
                if CFG.get("format", {}).get("host_compaction", True):
                    parts.append(compact_hosts(extract_hosts(r.hosts_raw), width, ellipsis=ell))
                else:
                    s = "+".join(extract_hosts(r.hosts_raw))
                    parts.append(ellipsize(s, width, ell))
            elif key == "mem_pct":
                parts.append(fmt_pct_right(r.mem_pct, width))
            elif key == "start":
                s = fmt_dt14(r.start_ts)
                parts.append(ellipsize(s, width, ell))
            else:
                parts.append(ellipsize(vals.get(key, ""), width, ell))
        return " ".join(parts)

    def _make_header_line(self, w: int) -> str:
        colspec = self._compute_columns(w)
        names = {
            "user":"USER","egroup":"EGROUP","jobid":"JOBID","queue":"QUEUE","start":"START",
            "wall":"WALL","mem":"MEM","mem_pct":"%MEM","cpu_now":"CPU%","cpu_eff":"EFF%","hosts":"HOSTS",
        }
        parts = [ellipsize(names[k], width, CFG.get("ui", {}).get("ellipsis","...")) for k, width in colspec]
        return " ".join(parts)

    def _make_status_line(self, w: int, n_items: int, colspec: List[Tuple[str, int]]) -> str:
        left = []
        if self.last_fetch_ok_ts:
            left.append("OK@" + time.strftime("%H:%M:%S", time.localtime(self.last_fetch_ok_ts)))
        else:
            left.append("Waiting…")
        left.append(f"jobs:{n_items}")
        # hidden cols
        all_cols = set(CFG.get("columns", []))
        shown = {k for k,_ in colspec}; hidden = sorted(list(all_cols - shown))
        if hidden: left.append("hidden[" + ",".join(hidden) + "]")
        # filters
        filt = []
        if self.model.filter_user:  filt.append(f"user=/{self.model.filter_user}/")
        if self.model.filter_queue: filt.append(f"queue=/{self.model.filter_queue}/")
        if self.model.search_substr: filt.append(f"find='{self.model.search_substr}'")
        if filt: left.append("filt[" + ", ".join(filt) + "]")
        left.append(f"sort={self.model.sort_key}{'^' if self.model.sort_asc else 'v'}")
        left.append(f"poll={self.refresh_secs}s")
        right = []
        if self.last_fetch_err: right.append("ERROR! (E for details)")
        msg_left = " | ".join(left); msg_right = " | ".join(right)
        if msg_right:
            pad = max(1, w - len(msg_left) - len(msg_right) - 1)
            return (msg_left + " " * pad + msg_right)[:w]
        return msg_left[:w]
    def _draw_segments(self, stdscr, y: int, segments: List[Tuple[str, int]], w: int):
        """
        Draw a list of (text, attr) segments left-to-right, clipped to width w.
        """
        x = 0
        for text, attr in segments:
            if x >= w: break
            remaining = w - x
            s = text if len(text) <= remaining else text[:remaining]
            safe_addnstr(stdscr, y, x, s, len(s), attr)
            x += len(s)
        # pad if needed
        if x < w:
            safe_addnstr(stdscr, y, x, " " * (w - x), w - x, curses.A_NORMAL)


    # ----- input -----

    def _handle_key(self, ch: int, stdscr) -> bool:
        if ch in (curses.KEY_UP, ord('k')): self.selection = max(0, self.selection - 1)
        elif ch in (curses.KEY_DOWN, ord('j')): self.selection += 1
        elif ch == curses.KEY_PPAGE: self.selection = max(0, self.selection - 10)
        elif ch == curses.KEY_NPAGE: self.selection += 10
        elif ch in (ord('g'),): self.selection = 0
        elif ch in (ord('G'),): self.selection = max(0, len(self.model.current_view()) - 1)
        elif ch in (ord('q'),): return False

        elif ch in (10, 13): self._show_job_details(stdscr)  # Enter
        elif ch in (ord('r'),): self._refresh_data()
        elif ch in (ord('x'),): self._export_csv()
        elif ch in (ord('?'),): self._modal_text(stdscr, "Help", HELP_TEXT)
        elif ch in (ord('E'),):
            raw = self.last_raw_error or "(no recent error)"
            self._modal_text(stdscr, "Last qstat/raw", self._clip_big_text(raw))

        elif ch in (ord('/'),):
            s = self._prompt(stdscr, "Search substring (Enter=clear): ")
            self.model.search_substr = s or None; self.selection = 0
        elif ch in (ord('f'),):
            u = self._prompt(stdscr, "Filter user (regex, empty=keep): ")
            q = self._prompt(stdscr, "Filter queue (regex, empty=keep): ")
            if u is not None: self.model.filter_user = u or None
            if q is not None: self.model.filter_queue = q or None
            self.selection = 0
        elif ch in (ord('a'),):
            self.model.filter_user = None; self.model.filter_queue = None; self.model.search_substr = None
            self.selection = 0

        elif ch in (ord('t'),):
            s = self._prompt(stdscr, f"Poll (s) current {self.refresh_secs}: ")
            if s:
                try:
                    val = int(s)
                    if val >= 1: self.refresh_secs = val
                except Exception: pass

        # Sorting
        elif ch in (ord('e'), ord('E')): self.model.set_sort("cpu_eff", ch == ord('e'))
        elif ch in (ord('c'), ord('C')): self.model.set_sort("cpu_now", ch == ord('c'))
        elif ch in (ord('m'), ord('M')): self.model.set_sort("mem_pct", ch == ord('m'))
        elif ch in (ord('w'), ord('W')): self.model.set_sort("wall", ch == ord('w'))
        elif ch in (ord('u'), ord('U')): self.model.set_sort("user", ch == ord('u'))
        elif ch in (ord('j'), ord('J')): self.model.set_sort("jobid", ch == ord('j'))
        elif ch in (ord('o'), ord('O')): self.model.set_sort("queue", ch == ord('o'))
        elif ch in (ord('s'), ord('S')): self.model.set_sort("start", ch == ord('s'))

        return True

    # ----- details -----
    def _show_job_details(self, stdscr):
        view = self.model.current_view()
        if not view:
            return
        idx = int(clamp(self.selection, 0, len(view) - 1))
        job = view[idx]
        jobid = job.jobid

        try:
            raw = self._qstat_f_one(jobid)

            # Timestamps
            st_ts = job.start_ts
            qt_ts = parse_stime_to_epoch(_extract_attr(raw, "qtime") or "") \
                    or parse_stime_to_epoch(str(job.raw.get("qtime") or ""))

            # ---------- SUMMARY as aligned label:value lines ----------
            items = [
                ("Start",     fmt_dt_full(st_ts)),
                ("Submitted", fmt_dt_full(qt_ts)),
                ("ncpus",     str(job.ncpus)),
                ("mem_used",  f"{job.mem_gib:.1f}G"),
                ("mem_req",   f"{job.mem_req_gib:.1f}G" if job.mem_req_gib else "-"),
                ("%MEM",      f"{int(round(job.mem_pct))}%" if job.mem_pct is not None else "-"),
                ("vmem_used", f"{job.vmem_gib:.1f}G" if job.vmem_gib is not None else "-"),
            ]
            hosts_compact = compact_hosts(extract_hosts(job.hosts_raw), 80).strip()
            label_w = max(len(k) for k, _ in (items + [("hosts","")]))

            summary_lines = [f"{k:<{label_w}}: {v}" for k, v in items]
            summary_lines.append(f"{'hosts':<{label_w}}: {hosts_compact}")
            summary_block = "\n".join(summary_lines)

            # ---------- root extras (only if actually present) ----------
            extras_block = ""
            if os.geteuid() == 0 and CFG.get("root_extra", {}).get("enabled", False):
                extras = self._root_extra_info(job.user, job.jobid)  # returns only non-empty fields
                # single-line fields
                single_lines = []
                for k in ("Full name", "Email", "Groups"):
                    v = extras.get(k)
                    if v:
                        single_lines.append(f"{k:<{label_w}}: {v}")
                # PBS script (raw block), only if printjob -s returned something
                scr = (extras.get("PBS script") or "").strip()
                if single_lines or scr:
                    parts = []
                    if single_lines:
                        parts.append("\n".join(single_lines))
                    if scr:
                        parts.append("PBS script:\n" + scr)
                    extras_block = "\n".join(parts)

            # ---------- Variable_List ----------
            vlist_text = ""
            try:
                vlist_text = job.raw.get("Variable_List") or ""
            except Exception:
                vlist_text = ""
            if not vlist_text:
                vl_lines = [m.group(0) for m in re.finditer(r'^\s*Variable_List\s*=\s*(.*)$', raw, re.MULTILINE)]
                if vl_lines:
                    vlist_text = "\n".join(vl_lines)

            # ---------- Assemble tabs (only for non-empty sections) ----------
            tabs: List[Tuple[str, str]] = []
            tabs.append(("Summary", summary_block))
            if extras_block.strip():
                # Name tab "Script" if it contains a script block; otherwise "Extras"
                tab_name = "Script" if "PBS script:" in extras_block else "Extras"
                tabs.append((tab_name, extras_block))
#            if vlist_text.strip():
#                tabs.append(("Vars", vlist_text))
            tabs.append(("Raw", f"qstat -f {jobid}\n{raw}"))

            # Show modal with tabs and scrolling
            self._modal_sections(stdscr, f"Job {jobid}", tabs)

        except Exception as e:
            self._modal_text(stdscr, f"Job {jobid}", f"Failed to get details: {e}")
    

    def _qstat_f_one(self, jobid: str) -> str:
        cmd = [which_qstat(CFG.get("qstat_candidates", DEFAULT_CONFIG["qstat_candidates"])), "-f", jobid]
        env = os.environ.copy(); env.setdefault("LC_ALL","C"); env.setdefault("LANG","C")
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=min(30, self.client.timeout_secs), env=env)
        return safe_decode(out)

    # ----- small UI helpers -----

    def _format_cmd(self, template: str, user: str, jobid: str) -> str:
        # Safe shell substitution with quoting
        return (template
                .replace("{user}", shlex.quote(user))
                .replace("{jobid}", shlex.quote(jobid)))

    def _run_shell(self, cmd: str, timeout: int) -> str:
        env = os.environ.copy(); env.setdefault("LC_ALL","C"); env.setdefault("LANG","C")
        out = subprocess.run(cmd, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             timeout=timeout, check=False)
        text = safe_decode(out.stdout).strip()
        # avoid excessively long blobs
        if len(text) > 8000:
            text = text[:4000] + "\n...[truncated]...\n" + text[-4000:]
        return text

    def _root_extra_info(self, user: str, jobid: str) -> Dict[str, str]:
        """
        Return root-only extras as a dict. Only include non-empty fields.
        For the 'PBS script' entry, include output strictly from 'printjob -s {jobid}'
        (already provided via config), and only if it prints something.
        """
        cfg = CFG.get("root_extra", {}) or {}
        if os.geteuid() != 0 or not cfg.get("enabled", False):
            return {}

        timeout = int(cfg.get("timeout_secs", 6))
        extras: Dict[str, str] = {}

        cmds = cfg.get("commands") or []
        for item in cmds:
            label = str(item.get("label") or "").strip()
            tmpl  = str(item.get("cmd")   or "").strip()
            if not label or not tmpl:
                continue

            try:
                cmd = self._format_cmd(tmpl, user, jobid)  # safe quoting with shlex
                out = self._run_shell(cmd, timeout).rstrip()
            except Exception:
                out = ""

            # Only include non-empty outputs.
            if not out:
                continue

            # No massaging: store as-is. The caller decides layout (e.g., header + block).
            extras[label] = out

        return extras


    def _prompt(self, stdscr, title: str) -> Optional[str]:
        h, w = stdscr.getmaxyx()
        win_h = 3; win_w = min(w - 4, max(40, len(title) + 20))
        y = h // 2 - win_h // 2; x = w // 2 - win_w // 2
        win = curses.newwin(win_h, win_w, y, x)
        win.border(); win.addnstr(0, 2, " " + title + " ", win_w - 4, curses.A_REVERSE)
        curses.curs_set(1); buf = ""
        while True:
            win.addnstr(1, 2, (buf + " ").ljust(win_w - 4), win_w - 4)
            win.move(1, 2 + len(buf)); win.refresh()
            ch = win.getch()
            if ch in (27,): curses.curs_set(0); return None
            elif ch in (curses.KEY_ENTER, 10, 13): curses.curs_set(0); return buf
            elif ch in (curses.KEY_BACKSPACE, 127, 8): buf = buf[:-1]
            elif 0 < ch < 256 and (chr(ch).isprintable()): buf += chr(ch)

    def _modal_text(self, stdscr, title: str, text: str):
        lines = textwrap.dedent(text).splitlines() or [""]
        pos = 0
        while True:
            h, w = stdscr.getmaxyx()
            win_h = min(h - 4, max(8, min(40, len(lines) + 4)))
            win_w = min(w - 4, max(60, min(120, max(len(l) for l in lines) + 4)))
            y = h // 2 - win_h // 2; x = w // 2 - win_w // 2
            win = curses.newwin(win_h, win_w, y, x)
            win.keypad(True)  # <- essencial para setas não virarem ESC
            win.border()
            hdr = f" {title} (q to close) "
            win.addnstr(0, 2, hdr[: win_w - 4], win_w - 4, curses.A_REVERSE)
            view_lines = lines[pos : pos + (win_h - 2)]
            for i, ln in enumerate(view_lines, start=1):
                win.addnstr(i, 1, ln[: win_w - 2].ljust(win_w - 2), win_w - 2)
            win.refresh()
            ch = win.getch()
            if ch in (ord('q'), ord('Q')):
                break
            elif ch in (curses.KEY_UP, ord('k')):
                pos = max(0, pos - 1)
            elif ch in (curses.KEY_DOWN, ord('j')):
                pos = min(max(0, len(lines) - (win_h - 2)), pos + 1)
            elif ch == curses.KEY_PPAGE:
                pos = max(0, pos - (win_h - 2))
            elif ch == curses.KEY_NPAGE:
                pos = min(max(0, len(lines) - (win_h - 2)), pos + (win_h - 2))
            elif ch in (curses.KEY_RESIZE,):
                continue  # redesenha
            else:
                 # ignora outras teclas (não fechar!)
                pass


    def _clip_big_text(self, s: str, limit: int = 8000) -> str:
        if not s: return ""
        if len(s) <= limit: return s
        head = s[:limit // 2]; tail = s[-limit // 2:]
        return head + "\n\n[... truncated ...]\n\n" + tail

    def _export_csv(self):
        view = self.model.current_view()
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.export_counter += 1
        fn = f"pbstop-{ts}-{self.export_counter:02d}.csv"
        try:
            with open(fn, "w", encoding="utf-8") as f:
                f.write("user,egroup,jobid,queue,start,wall_s,wall,mem_gib,mem_req_gib,mem_pct,cpu_now_pct,cpu_eff_pct,ncpus,hosts\n")
                for r in view:
                    wall = fmt_wall_right(r.wall_s, CFG["column_widths"].get("wall",10)).strip()
                    hosts_compact = compact_hosts(extract_hosts(r.hosts_raw), 64).strip() if CFG.get("format",{}).get("host_compaction", True) else "+".join(extract_hosts(r.hosts_raw))
                    start_s = fmt_dt14(r.start_ts)
                    mempct = "" if r.mem_pct is None else f"{r.mem_pct:.1f}"
                    f.write(f"{r.user},{r.egroup},{r.jobid},{r.queue},{start_s},{r.wall_s},{wall},{r.mem_gib:.3f},{(r.mem_req_gib or 0):.3f},{mempct},{r.cpu_now_pct:.1f},{r.cpu_eff_pct:.1f},{r.ncpus},{hosts_compact}\n")
            self.last_fetch_err = f"Exported: {fn}"
        except Exception as e:
            self.last_fetch_err = f"CSV export failed: {e}"

# ----- helpers for details parsing -----

_VAR_LIST_RE = re.compile(r"^ *Variable_List *= *(.*)$", re.MULTILINE)
_ATTR_RE_TMPL = r"^ *{attr} *= *(.*)$"

def _extract_attr(qstat_f_text: str, attr: str) -> Optional[str]:
    m = re.search(_ATTR_RE_TMPL.format(attr=re.escape(attr)), qstat_f_text, re.MULTILINE)
    if not m: return None
    return m.group(1).strip()

def _extract_var_from_variable_list(qstat_f_text: str, varname: str) -> Optional[str]:
    m = _VAR_LIST_RE.search(qstat_f_text)
    if not m: return None
    blob = m.group(1)
    pat = re.compile(rf"{re.escape(varname)}=([^,]+)")
    m2 = pat.search(blob)
    if not m2: return None
    return m2.group(1).strip()

# -------------------------
# Entrypoint
# -------------------------

def parse_args(argv: List[str]) -> Tuple[int, Optional[str], Optional[str], bool, str]:
    """
    Returns (poll_seconds_override, qstat_path_override, config_path, snapshot, snapshot_format).
    Positional compatibility: script [poll] [qstat_path]
    Options: --config PATH, --snapshot, --snapshot-format {csv,tsv,table}
    """
    poll_override = None
    qstat_override = None
    config_path = None
    snapshot = False
    snapshot_format = "csv"

    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]; i += 2; continue
        if a == "--snapshot":
            snapshot = True; i += 1; continue
        if a == "--snapshot-format" and i + 1 < len(argv):
            snapshot_format = argv[i + 1].lower()
            if snapshot_format not in ("csv","tsv","table"):
                snapshot_format = "csv"
            i += 2; continue
        # collect legacy positionals
        break

    # legacy positionals after options:
    rest = argv[i:]
    if len(rest) >= 1:
        try:
            poll_override = max(1, int(rest[0]))
        except Exception:
            pass
    if len(rest) >= 2:
        qstat_override = rest[1]

    return (poll_override if poll_override is not None else -1, qstat_override, config_path, snapshot, snapshot_format)

def main():
    global CFG
    poll_override, qstat_override, config_path, snapshot, snapshot_format = parse_args(sys.argv)
    CFG = load_config(config_path)

    qstat_path = which_qstat(CFG.get("qstat_candidates", DEFAULT_CONFIG["qstat_candidates"]))
    if qstat_override:
        qstat_path = qstat_override

    refresh = CFG.get("poll_seconds", 120)
    if poll_override and poll_override > 0:
        refresh = poll_override

    client = PbsClient(qstat_path=qstat_path, timeout_secs=min(60, max(5, int(refresh * 0.8))))

    # NEW: one-shot snapshot to STDOUT
    if snapshot:
        try:
            print_snapshot(client, CFG, qstat_path, fmt=snapshot_format)
        except PbsError as e:
            sys.stderr.write("PBS ERROR (snapshot): " + str(e) + "\n")
            if e.raw_output:
                sys.stderr.write("\n--- RAW OUTPUT ---\n")
                sys.stderr.write(e.raw_output[:8000] + ("\n...[truncated]...\n" if len(e.raw_output) > 8000 else ""))
            sys.exit(2)
        return

    # otherwise run curses UI
    app = CursesApp(client=client, refresh_secs=refresh)
    try:
        app.run()
    except PbsError as e:
        sys.stderr.write("PBS ERROR: " + str(e) + "\n")
        if e.raw_output:
            sys.stderr.write("\n--- RAW OUTPUT ---\n")
            sys.stderr.write(e.raw_output[:8000] + ("\n...[truncated]...\n" if len(e.raw_output) > 8000 else ""))
        sys.exit(2)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

