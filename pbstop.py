#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pbstop — Top-like TUI for PBS/OpenPBS (minimal TUI + CLI utilities)
-------------------------------------------------------------------
This version adds:
- argparse-based CLI with: --config, --poll-seconds, --qstat, --debug,
  --check-config, --persist-write, --at, --quiet.
- Config validation with actionable messages.
- CPU% "trust" plumbing (delta vs fallback) and global trust ratio.
- Optional sqlite persistence writer (for cron) and CLI time viewer.
- Minimal curses TUI that shows trust badge and essential columns.
  (Key map trimmed but includes: q, r, ?, c/C, e/E; plus history keys when persistence is enabled)

Stdlib-only. Assumes OpenPBS/PBSPro supports: `qstat -f -F json`.
Falls back to empty result if JSON is unavailable.

NOTE: This file intentionally keeps the TUI minimal to prioritize the
requested CLI/persistence features. It can be wired to your richer TUI
later by reusing the same data model (JobRow) and helpers.
"""
import os
import sys
import re
import json
import time
import math
import argparse
import sqlite3
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import curses
except Exception:
    curses = None  # allow CLI-only usage on systems without curses

# -------------------------
# Defaults & Config
# -------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "columns": ["user", "egroup", "jobid", "queue", "start", "wall", "mem", "mem_pct", "cpu_now", "cpu_eff", "hosts"],
    "column_widths": {
        "user": 10, "egroup": 10, "jobid": 8, "queue": 10, "start": 16,
        "wall": 10, "mem": 7, "mem_pct": 6, "cpu_now": 6, "cpu_eff": 6, "hosts": 30
    },
    "poll_seconds": 120,
    "ewma": {"windows_seconds": [60, 300, 900], "metric": "cpu_now_pct"},
    "ui": {"dim_cpu_below_pct": 20.0, "ellipsis": "..."},
    "qstat_candidates": ["/opt/pbs/bin/qstat", "/usr/bin/qstat", "qstat"],
    "cpu_now": {"source": "cput_delta", "divide_by_ncpus": True, "clip_0_100": True},
    "egroup_attr_order": ["egroup", "group_list", "group"],
    "persistence": {"enabled": False, "path": "/var/lib/pbstop", "mode": "aggregate", "top_k": 50, "retention_days": 30},
    "queue_memory_defaults": {}
}

CFG: Dict[str, Any] = {}

# -------------------------
# Small utils
# -------------------------

_DEBUG_FH = None
def dlog(msg: str):
    global _DEBUG_FH
    if _DEBUG_FH:
        try:
            _DEBUG_FH.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n"); _DEBUG_FH.flush()
        except Exception:
            pass

def now_ts() -> int:
    return int(time.time())

def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x

def parse_hms_to_seconds(s: str) -> int:
    # Accept "HH:MM:SS" or "DAYS+HH:MM:SS"
    s = str(s).strip()
    if not s:
        return 0
    if '+' in s:
        days, rest = s.split('+', 1)
        d = int(days) if days.isdigit() else 0
        hh, mm, ss = rest.split(':')
        return d*86400 + int(hh)*3600 + int(mm)*60 + int(ss)
    parts = s.split(':')
    if len(parts) == 3:
        hh, mm, ss = parts
        return int(hh)*3600 + int(mm)*60 + int(ss)
    # seconds as int fallback
    try:
        return int(float(s))
    except Exception:
        return 0

def parse_size_to_bytes(s: str) -> int:
    """Parse PBS size strings like '123kb', '4gb', '1024', '12345kb'."""
    if s is None:
        return 0
    s = str(s).strip().lower()
    m = re.match(r'^([\d\.]+)\s*([kmgtp]?b)?$', s)
    if not m:
        return 0
    val = float(m.group(1))
    unit = m.group(2) or 'b'
    mult = {'b':1, 'kb':1024, 'mb':1024**2, 'gb':1024**3, 'tb':1024**4, 'pb':1024**5}.get(unit, 1)
    return int(val * mult)

def fmt_dt_full(ts: Optional[int]) -> str:
    if not ts:
        return '-'
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))

def fmt_hms(seconds: int) -> str:
    if seconds < 0:
        seconds = 0
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"

def fmt_pct_right(p: Optional[float], width: int) -> str:
    if p is None:
        return "-".rjust(width)
    return (f"{int(round(p))}%").rjust(width)

def fmt_gib_right(gib: Optional[float], width: int) -> str:
    if gib is None:
        return "-".rjust(width)
    s = f"{gib:.1f}G"
    return s.rjust(width)

_HOST_SPLIT_RE = re.compile(r"^([a-zA-Z0-9\-\._]+?)(\d+)$")
def extract_hosts(exec_host: str) -> List[str]:
    if not exec_host:
        return []
    # PBS exec_host might be like: "node1/0*4+node2/0*4"
    parts = []
    for token in str(exec_host).split('+'):
        node = token.split('/')[0]
        parts.append(node)
    return parts

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
        nums = [n for n in nums if n is not None]
        if nums:
            ns = sorted(set(int(n) for n in nums))
            ranges = []
            start = prev = ns[0]
            for x in ns[1:]:
                if x == prev + 1:
                    prev = x; continue
                ranges.append((start, prev)); start = prev = x
            ranges.append((start, prev))
            frag = pref + "[" + ",".join(f"{a}-{b}" if a!=b else f"{a:02d}" for a,b in ranges) + "]"
        else:
            frag = pref
        parts.append(frag)
    out = ",".join(parts)
    if len(out) <= max_width:
        return out.rjust(max_width)
    if max_width <= len(ellipsis):
        return ellipsis[:max_width]
    return (out[:max_width-len(ellipsis)] + ellipsis)

# -------------------------
# Config I/O
# -------------------------

def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)  # strict JSON
        # shallow merge
        for k, v in user_cfg.items():
            cfg[k] = v
    return cfg

def validate_config(cfg: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    allowed_cols = {"user","egroup","jobid","queue","start","wall","mem","mem_pct","cpu_now","cpu_eff","hosts"}
    cols = cfg.get("columns", [])
    if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
        errors.append("`columns` must be a list of strings.")
    else:
        unknown = [c for c in cols if c not in allowed_cols]
        if unknown:
            errors.append(f"`columns` has unknown keys: {', '.join(unknown)}")
    wins = (cfg.get("ewma") or {}).get("windows_seconds", [])
    if not isinstance(wins, list) or not all(isinstance(x, int) and x>0 for x in wins):
        errors.append("`ewma.windows_seconds` must be positive integers (seconds).")
    v = (cfg.get("ui") or {}).get("dim_cpu_below_pct", 20.0)
    try:
        vf = float(v)
        if not (0.0 <= vf <= 100.0): raise ValueError()
    except Exception:
        errors.append("`ui.dim_cpu_below_pct` must be a number in [0,100].")
    src = (cfg.get("cpu_now") or {}).get("source")
    if src and src not in ("cput_delta","cpupercent"):
        errors.append("`cpu_now.source` must be 'cput_delta' or 'cpupercent'.")
    pers = cfg.get("persistence", {})
    if pers:
        if not isinstance(pers, dict):
            errors.append("`persistence` must be an object.")
        else:
            if "enabled" in pers and not isinstance(pers["enabled"], bool):
                errors.append("`persistence.enabled` must be true/false.")
            if "path" in pers and not isinstance(pers["path"], str):
                errors.append("`persistence.path` must be a string path.")
            if "mode" in pers and pers["mode"] not in ("aggregate","full"):
                errors.append("`persistence.mode` must be 'aggregate' or 'full'.")
    return (len(errors)==0, errors)

def which_qstat(candidates: List[str]) -> str:
    for p in candidates:
        try:
            r = subprocess.run([p, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=2)
            if r.returncode == 0 or r.returncode == 2:  # PBS qstat may return 2 for --version
                return p
        except Exception:
            continue
    # fallback to "qstat" and hope PATH resolves it
    return "qstat"

# -------------------------
# Data model
# -------------------------

@dataclass
class JobRow:
    user: str
    egroup: str
    jobid: str
    queue: str
    start_ts: Optional[int]
    wall_s: int
    mem_gib: Optional[float]
    cpu_now_pct: float
    cpu_eff_pct: float
    ncpus: int
    mem_req_gib: Optional[float]
    mem_pct: Optional[float]
    hosts_raw: str
    vmem_gib: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    cpu_now_src: str = "fallback"  # "delta" or "fallback"

# -------------------------
# PBS client (JSON first, text fallback minimal)
# -------------------------

class PbsClient:
    def __init__(self, qstat_path: str):
        self.qstat_path = qstat_path
        self.last_raw_error: Optional[str] = None

    def _run(self, args: List[str]) -> Tuple[int, str]:
        try:
            cp = subprocess.run([self.qstat_path] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
            return cp.returncode, cp.stdout
        except Exception as e:
            return 1, str(e)

    def fetch_jobs(self) -> List[Dict[str, Any]]:
        # Try JSON
        rc, out = self._run(["-f", "-F", "json"])
        if rc == 0:
            try:
                data = json.loads(out)
                # PBS JSON format nests jobs under "Jobs"
                jobs = list((data.get("Jobs") or {}).values())
                return jobs
            except Exception as e:
                self.last_raw_error = f"JSON parse error: {e}\n{out[:2000]}"
        else:
            self.last_raw_error = out

        # Fallback: return empty (keep stdlib; robust text parser omitted for brevity)
        return []

# -------------------------
# Transformations
# -------------------------

def _get_str(d: Dict[str, Any], *keys, default="") -> str:
    x = d
    for k in keys:
        if isinstance(x, dict):
            x = x.get(k, {})
        else:
            return default
    if isinstance(x, (str, int, float)):
        return str(x)
    return default

def _get_float(d: Dict[str, Any], *keys) -> Optional[float]:
    s = _get_str(d, *keys, default="").strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None

def make_job_row(rec: Dict[str, Any], ts: Optional[int] = None) -> JobRow:
    owner = _get_str(rec, "Job_Owner", default="")
    user = owner.split("@")[0] if owner else _get_str(rec, "euser", default="-")
    egroup = _get_str(rec, "egroup", default="-")
    jobid = _get_str(rec, "id", default=_get_str(rec, "Job_Id", default="-"))
    queue = _get_str(rec, "queue", default="-")
    stime = _get_str(rec, "stime", default=_get_str(rec, "start_time", default="")).strip()
    start_ts = None
    if stime:
        try:
            # PBS stime like "Mon Sep 16 10:11:12 2025"
            start_ts = int(time.mktime(time.strptime(stime, "%a %b %d %H:%M:%S %Y")))
        except Exception:
            start_ts = None
    wall_s = 0
    if start_ts:
        wall_s = max(0, (ts or now_ts()) - start_ts)
    # resources
    ru = rec.get("resources_used") or {}
    rl = rec.get("Resource_List") or {}
    cput_s = parse_hms_to_seconds(str((ru.get("cput") or "0")))
    mem_used_b = parse_size_to_bytes(str(ru.get("mem") or "0b"))
    vmem_used_b = parse_size_to_bytes(str(ru.get("vmem") or "0b"))
    mem_req_b = parse_size_to_bytes(str(rl.get("mem") or "0b"))
    ncpus = int(rl.get("ncpus") or rl.get("mpiprocs") or 1)
    cpupercent_total = float(ru.get("cpupercent") or 0.0)
    cpu_now_pct = (cpupercent_total / ncpus) if ncpus > 0 else 0.0
    cpu_now_pct = clamp(cpu_now_pct, 0.0, 100.0)
    # efficiency
    cpu_eff_pct = 0.0
    if wall_s > 0 and ncpus > 0:
        cpu_eff_pct = clamp((cput_s / float(wall_s * ncpus)) * 100.0, 0.0, 100.0)
    # mem GiB & pct
    mem_gib = mem_used_b / (1024**3) if mem_used_b > 0 else None
    mem_req_gib = mem_req_b / (1024**3) if mem_req_b > 0 else None
    mem_pct = None
    if mem_gib is not None and mem_req_gib:
        mem_pct = clamp((mem_gib / mem_req_gib) * 100.0, 0.0, 999.0)
    hosts_raw = _get_str(rec, "exec_host", default="")
    return JobRow(
        user=user, egroup=egroup, jobid=jobid, queue=queue, start_ts=start_ts, wall_s=wall_s,
        mem_gib=mem_gib, cpu_now_pct=cpu_now_pct, cpu_eff_pct=cpu_eff_pct, ncpus=ncpus,
        mem_req_gib=mem_req_gib, mem_pct=mem_pct, hosts_raw=hosts_raw, vmem_gib=vmem_used_b/(1024**3) if vmem_used_b>0 else None,
        raw=rec, cpu_now_src="fallback"
    )

def apply_cpu_now_delta(rows: List[JobRow], prev: Dict[str, int], poll_seconds: int) -> Tuple[List[JobRow], float, Dict[str,int]]:
    """Replace cpu_now_pct using Δcput when possible. Returns (rows, trust_ratio, new_prev_map)."""
    if poll_seconds <= 0 or not rows:
        return rows, 0.0, prev
    delta_ok = 0
    new_prev = dict(prev)
    for r in rows:
        ru = r.raw.get("resources_used") or {}
        cput_s = parse_hms_to_seconds(str(ru.get("cput") or "0"))
        prev_cput = new_prev.get(r.jobid)
        if prev_cput is not None and r.ncpus > 0 and cput_s >= prev_cput:
            d = cput_s - prev_cput
            r.cpu_now_pct = clamp((d / float(poll_seconds * r.ncpus)) * 100.0, 0.0, 100.0)
            r.cpu_now_src = "delta"; delta_ok += 1
        else:
            r.cpu_now_src = "fallback"
        new_prev[r.jobid] = cput_s
    trust = delta_ok / float(len(rows))
    return rows, trust, new_prev

# -------------------------
# EWMA helpers
# -------------------------

class Ewma:
    def __init__(self, window_seconds: int, poll_seconds: int):
        self.window = float(max(1, window_seconds))
        self.poll = float(max(1, poll_seconds))
        # continuous-time smoothing factor
        self.alpha = 1.0 - math.exp(-self.poll / self.window)
        self.value = None  # type: Optional[float]

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value
        return self.value

# -------------------------
# Persistence (sqlite)
# -------------------------

def _open_db(path_dir: str) -> sqlite3.Connection:
    os.makedirs(path_dir, exist_ok=True)
    dbpath = os.path.join(path_dir, "pbstop.sqlite")
    conn = sqlite3.connect(dbpath, timeout=5)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("CREATE TABLE IF NOT EXISTS meta(schema_version int, created_ts int)")
    conn.execute("""CREATE TABLE IF NOT EXISTS polls(
        ts integer primary key, ewma1 real, ewma5 real, ewma15 real, cpu_eff real, trust real)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS top_jobs(
        ts integer, rank integer, jobid text, user text, egroup text, queue text,
        start_ts integer, wall integer, mem_gib real, cpu_now real, cpu_eff real,
        ncpus integer, mem_req real, mem_pct real, hosts text, cpu_src text,
        primary key(ts, rank))""")
    return conn

def persist_snapshot(cfg: Dict[str,Any], ts: int,
                     ewma_triplet: Tuple[float,float,float], cpu_eff: float,
                     trust_ratio: float, rows: List[JobRow]) -> None:
    p = (cfg.get("persistence") or {}).get("path") or "/var/lib/pbstop"
    mode = (cfg.get("persistence") or {}).get("mode") or "aggregate"
    top_k = int((cfg.get("persistence") or {}).get("top_k") or 50)
    keep = int((cfg.get("persistence") or {}).get("retention_days") or 30)
    conn = _open_db(p)
    with conn:
        conn.execute("INSERT OR REPLACE INTO polls(ts,ewma1,ewma5,ewma15,cpu_eff,trust) VALUES(?,?,?,?,?,?)",
                     (ts,)+tuple(ewma_triplet)+(cpu_eff,trust_ratio))
        sel = sorted(rows, key=lambda x: x.cpu_now_pct, reverse=True)
        if mode == "aggregate":
            sel = sel[:top_k]
        for rank, r in enumerate(sel, start=1):
            hosts = compact_hosts(extract_hosts(r.hosts_raw), CFG.get("column_widths",{}).get("hosts",30), CFG.get("ui",{}).get("ellipsis","..."))
            conn.execute("""INSERT OR REPLACE INTO top_jobs
                (ts,rank,jobid,user,egroup,queue,start_ts,wall,mem_gib,cpu_now,cpu_eff,ncpus,mem_req,mem_pct,hosts,cpu_src)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts,rank,r.jobid,r.user,r.egroup,r.queue,r.start_ts or 0,r.wall_s,r.mem_gib,
                 r.cpu_now_pct,r.cpu_eff_pct,r.ncpus,r.mem_req_gib or 0.0,r.mem_pct if r.mem_pct is not None else None,hosts,r.cpu_now_src))
        cutoff = now_ts() - keep*86400
        conn.execute("DELETE FROM polls WHERE ts < ?", (cutoff,))
        conn.execute("DELETE FROM top_jobs WHERE ts < ?", (cutoff,))
    conn.close()

def read_snapshot_nearest(cfg: Dict[str,Any], when_ts: int) -> Tuple[Optional[int], List[JobRow], Tuple[float,float,float], float, float]:
    """Return (ts, rows, (ew1,ew5,ew15), cpu_eff, trust)."""
    p = (cfg.get("persistence") or {}).get("path") or "/var/lib/pbstop"
    dbpath = os.path.join(p, "pbstop.sqlite")
    if not os.path.exists(dbpath):
        return None, [], (0.0,0.0,0.0), 0.0, 0.0
    conn = sqlite3.connect(dbpath, timeout=5)
    try:
        cur = conn.cursor()
        cur.execute("SELECT ts, ewma1, ewma5, ewma15, cpu_eff, trust FROM polls WHERE ts <= ? ORDER BY ts DESC LIMIT 1", (when_ts,))
        row = cur.fetchone()
        if not row:
            return None, [], (0.0,0.0,0.0), 0.0, 0.0
        ts, ew1, ew5, ew15, ce, trust = row
        jobs: List[JobRow] = []
        for r in conn.execute("""SELECT rank,jobid,user,egroup,queue,start_ts,wall,mem_gib,cpu_now,cpu_eff,ncpus,mem_req,mem_pct,hosts,cpu_src
                                 FROM top_jobs WHERE ts=? ORDER BY rank ASC""", (ts,)):
            _rank, jobid,user,egroup,queue,start_ts,wall,mem_gib,cpu_now,cpu_eff,ncpus,mem_req,mem_pct,hosts,cpu_src = r
            jobs.append(JobRow(
                user=user, egroup=egroup, jobid=str(jobid), queue=queue,
                start_ts=int(start_ts) if start_ts else None, wall_s=int(wall or 0),
                mem_gib=float(mem_gib or 0.0), cpu_now_pct=float(cpu_now or 0.0),
                cpu_eff_pct=float(cpu_eff or 0.0), ncpus=int(ncpus or 1),
                mem_req_gib=(float(mem_req) if mem_req is not None else None),
                mem_pct=(float(mem_pct) if mem_pct is not None else None),
                hosts_raw=str(hosts or ""), vmem_gib=None, raw={}, cpu_now_src=str(cpu_src or "fallback")
            ))
        return int(ts), jobs, (float(ew1),float(ew5),float(ew15)), float(ce), float(trust)
    finally:
        conn.close()

# -------------------------
# Curses TUI (minimal)
# -------------------------

SORT_KEYS = {
    'c': ('cpu_now_pct', False), 'C': ('cpu_now_pct', True),
    'e': ('cpu_eff_pct', False), 'E': ('cpu_eff_pct', True),
}

def draw_tui(stdscr, cfg: Dict[str,Any], qstat_path: str, poll_seconds: int):
    curses.use_default_colors()
    has_colors = curses.has_colors()
    if has_colors:
        curses.init_pair(1, curses.COLOR_YELLOW, -1)  # badge ?
        curses.init_pair(2, curses.COLOR_GREEN, -1)   # badge ✓
        curses.init_pair(3, curses.COLOR_CYAN, -1)    # header

    client = PbsClient(qstat_path)
    prev_cput: Dict[str,int] = {}
    ewmas = [Ewma(w, poll_seconds) for w in cfg.get("ewma",{}).get("windows_seconds",[60,300,900])]
    sort_key, sort_desc = 'cpu_now_pct', True
    history_ts: Optional[int] = None
    persistence_enabled = bool((cfg.get("persistence") or {}).get("enabled"))

    while True:
        ts_now = now_ts()
        if history_ts is not None and persistence_enabled:
            ts = history_ts
            # read snapshot
            snap_ts, rows, (ew1,ew5,ew15), mean_eff, trust = read_snapshot_nearest(cfg, ts)
            if snap_ts is None:
                rows, ew1,ew5,ew15, mean_eff, trust = [], 0.0,0.0,0.0, 0.0, 0.0
                ts = history_ts
            ew_vals = [ew1, ew5, ew15]
        else:
            ts = ts_now
            recs = client.fetch_jobs()
            rows = [make_job_row(r, ts=ts) for r in recs]
            rows, trust, prev_cput = apply_cpu_now_delta(rows, prev_cput, poll_seconds)
            # global metrics live
            mean_cpu_now = (sum(r.cpu_now_pct for r in rows)/max(1,len(rows)))
            for ew in ewmas:
                ew.update(mean_cpu_now)
            ew_vals = [ew.value if ew.value is not None else 0.0 for ew in ewmas]
            mean_eff = (sum(r.cpu_eff_pct for r in rows)/max(1,len(rows)))

        # sort
        rows.sort(key=lambda r: getattr(r, sort_key), reverse=sort_desc)

        stdscr.erase()
        h, w = stdscr.getmaxyx()
        # topline 1
        hist_txt = f" | HIST @ {fmt_dt_full(ts)} (0=now)" if (history_ts is not None and persistence_enabled) else ""
        hdr1 = f"R={len(rows)}  Updated={fmt_dt_full(ts)}  Poll={poll_seconds}s  Sort={sort_key}{'↓' if sort_desc else '↑'}{hist_txt}"
        if has_colors: stdscr.attron(curses.color_pair(3))
        stdscr.addnstr(0,0,hdr1,w-1); 
        if has_colors: stdscr.attroff(curses.color_pair(3))
        # topline 2 with badge
        badge = "✓" if trust >= 0.6 else "?"
        badge_attr = curses.color_pair(2) if (has_colors and badge=="✓") else (curses.color_pair(1) if has_colors else curses.A_DIM)
        ew_txt = f"CPU% (1/5/15m)={ew_vals[0]:.1f}% {ew_vals[1]:.1f}% {ew_vals[2]:.1f}% "
        eff_txt = f"| CPUeff={mean_eff:.1f}% "
        stdscr.addnstr(1,0,ew_txt,w-1)
        stdscr.addnstr(1,len(ew_txt), "[", 1)
        stdscr.attron(badge_attr); stdscr.addnstr(1,len(ew_txt)+1, badge, 1); stdscr.attroff(badge_attr)
        stdscr.addnstr(1,len(ew_txt)+2, "] ", 2)
        stdscr.addnstr(1,len(ew_txt)+4, eff_txt, w - (len(ew_txt)+4))

        # table header
        widths = cfg.get("column_widths", {})
        y = 3
        header = f"{'USER':<{widths.get('user',10)}} {'JOBID':>{widths.get('jobid',8)}} {'QUEUE':<{widths.get('queue',10)}} {'MEM':>{widths.get('mem',7)}} {'CPU%':>{widths.get('cpu_now',6)}} {'EFF%':>{widths.get('cpu_eff',6)}} HOSTS"
        stdscr.addnstr(y,0,header,w-1); y += 1

        # rows
        for r in rows[:max(0, h - y - 1)]:
            cpu_now_txt = f"{int(round(r.cpu_now_pct))}%"
            if r.cpu_now_src != "delta":
                cpu_now_txt += "?"
            line = f"{r.user:<{widths.get('user',10)}} {r.jobid:>{widths.get('jobid',8)}} {r.queue:<{widths.get('queue',10)}} {fmt_gib_right(r.mem_gib,widths.get('mem',7))} {cpu_now_txt:>{widths.get('cpu_now',6)}} {fmt_pct_right(r.cpu_eff_pct,widths.get('cpu_eff',6))} {compact_hosts(extract_hosts(r.hosts_raw), widths.get('hosts',30), cfg.get('ui',{}).get('ellipsis','...'))}"
            attr = curses.A_DIM if (r.cpu_now_pct < float(cfg.get("ui",{}).get("dim_cpu_below_pct",20.0))) else curses.A_NORMAL
            stdscr.addnstr(y,0,line,w-1, attr)
            y += 1
            if y >= h-1: break

        footer = "q=quit  r=refresh  c/C=sort CPU now  e/E=sort EFF  ?=help"
        if persistence_enabled:
            footer += "  [ ]=±1m  { }=±5m  ( )=±1h  0=now"
        stdscr.addnstr(h-1,0,footer, w-1, curses.A_DIM)
        stdscr.refresh()

        # input with timeout (no timeout while in history mode)
        stdscr.timeout(0 if (history_ts is not None and persistence_enabled) else poll_seconds*1000)
        ch = stdscr.getch()
        if ch == -1:
            continue
        if ch in (ord('q'), ord('Q')):
            break
        if ch in (ord('r'), ord('R')):
            # exit history mode on refresh
            history_ts = None
            continue
        if ch in (ord('?'),):
            show_help(stdscr); continue
        cch = chr(ch) if 0 <= ch < 256 else ''
        if cch in SORT_KEYS:
            sort_key, sort_desc = SORT_KEYS[cch]
            continue
        if persistence_enabled:
            if cch == '[':   history_ts = (history_ts or ts_now) - 60
            elif cch == ']': history_ts = (history_ts or ts_now) + 60
            elif cch == '{': history_ts = (history_ts or ts_now) - 300
            elif cch == '}': history_ts = (history_ts or ts_now) + 300
            elif cch == '(': history_ts = (history_ts or ts_now) - 3600
            elif cch == ')': history_ts = (history_ts or ts_now) + 3600
            elif cch == '0': history_ts = None
            else:
                pass
            continue

def show_help(stdscr):
    lines = [
        "pbstop — minimal help",
        "",
        "CPU% now (delta): Δcput / (poll_seconds * ncpus) * 100  [trusted ✓]",
        "CPU% now (fallback): cpupercent / ncpus                [untrusted ?]",
        "CPUeff: cput / (wall * ncpus) * 100 (since start)",
        "",
        "Legend: ? = using fallback (less trustworthy), ✓ = delta-based,",
        "        dim values = below ui.dim_cpu_below_pct threshold.",
        "",
        "Keys: q quit | r refresh | c/C sort CPU now | e/E sort EFF | ? help",
        "      When persistence.enabled=true: [ ]=±1m  { }=±5m  ( )=±1h  0=now",
        "",
        "Press any key to return…",
    ]
    stdscr.erase()
    h, w = stdscr.getmaxyx()
    for i, s in enumerate(lines):
        if i >= h: break
        stdscr.addnstr(i, 0, s, w-1)
    stdscr.refresh()
    stdscr.getch()

# -------------------------
# CLI entrypoints
# -------------------------

def parse_when(s: Optional[str]) -> Optional[int]:
    if not s: return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%H:%M"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%H:%M":
                today = datetime.now().date()
                dt = datetime.combine(today, dt.time())
            return int(dt.timestamp())
        except Exception:
            continue
    return None

def get_args():
    p = argparse.ArgumentParser(prog="pbstop", description="Top-like TUI for PBS/OpenPBS.")
    p.add_argument("--config", help="Path to strict JSON config")
    p.add_argument("--poll-seconds", type=int, help="Override poll interval seconds")
    p.add_argument("--qstat", help="Path to qstat binary")
    p.add_argument("--debug", action="store_true", help="Enable verbose debug logging to /tmp/pbstop-debug-<pid>.log")
    p.add_argument("--check-config", nargs="?", const="__USE_CFG__", help="Validate config and exit (optionally pass a path)")
    p.add_argument("--persist-write", action="store_true", help="Poll once and write a snapshot to sqlite (cron mode)")
    p.add_argument("--at", help="Start at the given datetime (if persistence is enabled). Accepts 'HH:MM' or 'YYYY-MM-DD HH:MM'")
    p.add_argument("--quiet", action="store_true", help="Reduce non-error chatter")
    return p.parse_args()

def run_persist_writer_once(cfg: Dict[str,Any], qstat_path: str, poll_seconds: int, debug: bool=False) -> int:
    client = PbsClient(qstat_path)
    recs = client.fetch_jobs()
    ts = now_ts()
    rows = [make_job_row(r, ts=ts) for r in recs]
    rows, trust, _ = apply_cpu_now_delta(rows, {}, poll_seconds)
    cpu_eff = (sum(r.cpu_eff_pct for r in rows) / max(1, len(rows)))
    # EWMA: store instantaneous mean to all three (simple snapshot)
    ew_now = sum(r.cpu_now_pct for r in rows) / max(1, len(rows))
    persist_snapshot(cfg, ts, (ew_now, ew_now, ew_now), cpu_eff, trust, rows)
    if debug: dlog(f"persist-write: ts={ts} rows={len(rows)} trust={trust:.2f} cpu_eff={cpu_eff:.1f} ew_now={ew_now:.1f}")
    return 0

def run_cli_timeview(cfg: Dict[str,Any], at_ts: int) -> int:
    ts, rows, (ew1,ew5,ew15), cpu_eff, trust = read_snapshot_nearest(cfg, at_ts)
    if ts is None:
        print("No snapshot found at or before the requested time.", file=sys.stderr)
        return 1
    print(time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)),
          f"| EWMA 1/5/15m = {ew1:.1f}% {ew5:.1f}% {ew15:.1f}% [{'✓' if trust>=0.6 else '?'}] | CPUeff={cpu_eff:.1f}%")
    w_user = CFG.get("column_widths",{}).get("user",10)
    w_job  = CFG.get("column_widths",{}).get("jobid",8)
    w_q    = CFG.get("column_widths",{}).get("queue",10)
    w_mem  = CFG.get("column_widths",{}).get("mem",7)
    w_cnow = CFG.get("column_widths",{}).get("cpu_now",6)
    w_ceff = CFG.get("column_widths",{}).get("cpu_eff",6)
    print(f"{'USER':<{w_user}} {'JOBID':>{w_job}} {'QUEUE':<{w_q}} {'MEM':>{w_mem}} {'CPU%':>{w_cnow}} {'EFF%':>{w_ceff}} HOSTS")
    for r in rows:
        cpu_now_txt = f"{int(round(r.cpu_now_pct))}%{'?' if r.cpu_now_src!='delta' else ''}"
        print(f"{r.user:<{w_user}} {r.jobid:>{w_job}} {r.queue:<{w_q}} {fmt_gib_right(r.mem_gib,w_mem)} {cpu_now_txt:>{w_cnow}} {fmt_pct_right(r.cpu_eff_pct,w_ceff)} {r.hosts_raw}")
    return 0

def main():
    global CFG, _DEBUG_FH
    args = get_args()
    cfg_path = args.config if args.config else os.environ.get("PBSTOP_CONFIG")
    try:
        CFG = load_config(cfg_path)
    except json.JSONDecodeError as e:
        print(f"[CONFIG] JSON decode error in {cfg_path or '(env PBSTOP_CONFIG)'}: {e}", file=sys.stderr)
        sys.exit(2)
    except FileNotFoundError:
        # ok: will use defaults if no config path supplied; but if explicitly passed, fail
        if cfg_path:
            print(f"[CONFIG] File not found: {cfg_path}", file=sys.stderr); sys.exit(2)
        CFG = dict(DEFAULT_CONFIG)

    if args.check_config and args.check_config != "__USE_CFG__":
        try:
            CFG = load_config(args.check_config)
        except Exception as e:
            print(f"[CONFIG] Failed to load {args.check_config}: {e}", file=sys.stderr); sys.exit(2)

    ok, errs = validate_config(CFG)
    if not ok:
        for e in errs: print(f"[CONFIG] {e}", file=sys.stderr)
        sys.exit(2)

    poll_seconds = int(args.poll_seconds or CFG.get("poll_seconds",120))
    qstat_path = args.qstat or which_qstat(CFG.get("qstat_candidates", []))

    if args.debug:
        _DEBUG_FH = open(f"/tmp/pbstop-debug-{os.getpid()}.log","a", encoding="utf-8")
        dlog(f"qstat={qstat_path} poll_seconds={poll_seconds} cfg={cfg_path or '(defaults)'}")

    if args.persist_write:
        sys.exit(run_persist_writer_once(CFG, qstat_path, poll_seconds, debug=args.debug))

    if args.at:
        ts = parse_when(args.at)
        if ts is None:
            print("Invalid --at format. Use 'HH:MM' or 'YYYY-MM-DD HH:MM'.", file=sys.stderr); sys.exit(2)
        sys.exit(run_cli_timeview(CFG, ts))

    # TUI
    if curses is None:
        print("curses is not available. Use --persist-write or --at.", file=sys.stderr)
        sys.exit(1)
    curses.wrapper(draw_tui, CFG, qstat_path, poll_seconds)

if __name__ == "__main__":
    main()

