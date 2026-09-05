"""Append-only, resumable results table.

One row per (shape_id, direction, resolution) job, holding both the descriptor
value and its elapsed time (handoff sec 7.4 item 6). Every later question in the
paper -- which subset, how many, selected how, at what cost -- is a query
against this table, so a run must survive a crash, a logout or a reboot without
recomputing what it already has.

CSV rather than parquet: appends are atomic-enough line writes, the file stays
readable with `tail`, and at ~10^5 rows the size is trivial. A sidecar
`<name>.meta.json` records the protocol and machine state.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess

FIELDS = [
    "shape_id", "cls", "p", "q", "angle_deg", "norm2",
    "resolution", "E", "seconds",
    "host", "governor", "pinned", "git_commit",
]


def _git_commit(repo_dir: str = ".") -> str:
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        dirty = subprocess.run(
            ["git", "-C", repo_dir, "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        sha = out.stdout.strip() or "unknown"
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def cpu_governor() -> str:
    """Read the scaling governor of cpu0, or 'unknown' off Linux.

    Timings are the paper's result, so the governor must be recorded with them
    (handoff sec 1.3). Set it to `performance` before any measurement run.
    """
    path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    try:
        with open(path) as fh:
            return fh.read().strip()
    except Exception:
        return "unknown"


class ResultStore:
    def __init__(self, path: str, repo_dir: str = ".", pinned: bool = False):
        self.path = path
        self.repo_dir = repo_dir
        self.pinned = pinned
        self.host = platform.node()
        self.governor = cpu_governor()
        self.commit = _git_commit(repo_dir)
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="") as fh:
                csv.writer(fh).writerow(FIELDS)

    # -- resume support ----------------------------------------------------

    def done_keys(self) -> set:
        """(shape_id, p, q, resolution) already present in the table."""
        keys = set()
        if not os.path.exists(self.path):
            return keys
        with open(self.path, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    keys.add((row["shape_id"], int(row["p"]), int(row["q"]), int(row["resolution"])))
                except (KeyError, ValueError):
                    continue        # tolerate a torn final line from a crash
        return keys

    # -- writing -----------------------------------------------------------

    def append(self, rows) -> int:
        """Append rows. Each row is a dict with at least shape_id, cls, p, q,
        angle_deg, norm2, resolution, E, seconds."""
        n = 0
        with open(self.path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            for r in rows:
                r = dict(r)
                r.setdefault("host", self.host)
                r.setdefault("governor", self.governor)
                r.setdefault("pinned", self.pinned)
                r.setdefault("git_commit", self.commit)
                w.writerow(r)
                n += 1
            fh.flush()
            os.fsync(fh.fileno())
        return n

    def write_meta(self, meta: dict) -> None:
        meta = dict(meta)
        meta.update(
            host=self.host, governor=self.governor, pinned=self.pinned,
            git_commit=self.commit, python=platform.python_version(),
            platform=platform.platform(),
        )
        with open(os.path.splitext(self.path)[0] + ".meta.json", "w") as fh:
            json.dump(meta, fh, indent=2, sort_keys=True)


def load_table(path: str):
    """Read the table back as a dict keyed by (shape_id, p, q, resolution)."""
    out = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                key = (row["shape_id"], int(row["p"]), int(row["q"]), int(row["resolution"]))
                out[key] = {"cls": row["cls"], "E": float(row["E"]), "seconds": float(row["seconds"])}
            except (KeyError, ValueError):
                continue
    return out
