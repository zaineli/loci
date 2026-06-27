"""E3 - the theory out of sample: verify the protocol's hashes and score it against docs/PREREG.md.

The test itself was run from a frozen copy of the development code (results/oos/frozen/), with every
prediction written and hashed before any measurement. The full protocol, in order and with UTC
times, is in results/oos/REPORT.md, FROZEN.txt and PREDICTIONS.txt. This script re-checks those
hashes, checks that the joined results carry the hashed predictions unchanged, and computes the
registered criteria:
  T1  over all out-of-sample conditions, MAE <= 0.02 and >= 90% within 0.05
  T2  no condition family has MAE > 0.04

    uv run python bench/oos.py
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OOS = Path(__file__).resolve().parents[1] / "results" / "oos"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recorded() -> dict[str, str]:
    """The last hash each protocol file records for each file it names."""
    out = {}
    for name in ("FROZEN.txt", "PREDICTIONS.txt"):
        for digest, file in re.findall(r"^([0-9a-f]{64})\s+(\S+)$", (OOS / name).read_text(), re.M):
            out[file] = digest
    return out


def main() -> None:
    hashes = {file: _sha(OOS / file) == digest for file, digest in recorded().items() if (OOS / file).exists()}
    def key(row: dict) -> tuple:  # the recorded ids omit the load (the logged bug); the setting has it
        return row["family"], json.dumps(row["setting"], sort_keys=True), row["seed"], row["placement"]

    predictions = {key(row): row["T-relu"]["address"] for row in json.loads((OOS / "predictions.json").read_text())}
    rows = json.loads((OOS / "results.json").read_text())["rows"]
    unchanged = len(predictions) == len(rows) and all(
        abs(predictions[key(r)] - r["T-relu"]["address"]) < 1e-12 for r in rows)
    errors = np.array([r["T-relu"]["address"] - r["measured"] for r in rows])
    families = defaultdict(list)
    for r, e in zip(rows, errors, strict=True):
        families[r["family"]].append(abs(e))
    family_mae = {f: float(np.mean(v)) for f, v in sorted(families.items())}
    report = {
        "hashes verified": hashes,
        "predictions carried unchanged into results": unchanged,
        "conditions": len(rows),
        "MAE": float(np.abs(errors).mean()), "bias": float(errors.mean()), "max": float(np.abs(errors).max()),
        "within 0.05": float(np.mean(np.abs(errors) <= 0.05)),
        "correlation": float(np.corrcoef([r["T-relu"]["address"] for r in rows], [r["measured"] for r in rows])[0, 1]),
        "family MAE": family_mae,
        "T1": bool(np.abs(errors).mean() <= 0.02 and np.mean(np.abs(errors) <= 0.05) >= 0.9),
        "T2": bool(max(family_mae.values()) <= 0.04),
    }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
