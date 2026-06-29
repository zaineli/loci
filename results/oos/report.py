"""Step 5: compare predictions with measurements, per family, against the bars in PREDICTIONS.txt."""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def kendall(x, y):
    pairs = list(combinations(range(len(x)), 2))
    s = sum(np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]) for i, j in pairs)
    return s / len(pairs) if pairs else float("nan")


def main():
    uid = lambda r: f"{r['family']}|load {r['setting']['load']}|seed {r['seed']}|{r['placement']}"  # noqa: E731
    pred = {uid(r): {**r, "id": uid(r)} for r in json.loads((HERE / "predictions.json").read_text())}
    meas = {r["uid"]: r for r in json.loads((HERE / "measured.json").read_text())}
    assert len(pred) == 156
    assert set(pred) == set(meas), "predicted and measured conditions differ"
    rows = []
    for rid, p in pred.items():
        m = meas[rid]
        draws = np.array(m["address_draws"])
        rows.append({**p, "measured": m["address"], "measured_modules": m["modules"],
                     "err": p["T-relu"]["address"] - m["address"], "err_E": p["E"]["address"] - m["address"],
                     "module_err": np.abs(np.array(p["T-relu"]["modules"]) - np.array(m["modules"])).mean(),
                     # sampling floor of a 3-draw mean: binomial over P items, per draw
                     "floor": float(np.sqrt(max(m["address"] * (1 - m["address"]), 1e-9) / p["P"] / len(draws)))})
    fam = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    out = ["| family | rows | MAE | bias | max \\|err\\| | within 0.05 | module MAE | MAE, error law alone (E) | sampling floor | flagged (MAE > 0.05) |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for name, rs in list(fam.items()) + [("ALL", rows)]:
        e = np.array([r["err"] for r in rs])
        row = {"rows": len(rs), "MAE": float(np.abs(e).mean()), "bias": float(e.mean()),
               "max": float(np.abs(e).max()), "within": float(np.mean(np.abs(e) <= 0.05)),
               "module_MAE": float(np.mean([r["module_err"] for r in rs])),
               "MAE_E": float(np.mean([abs(r["err_E"]) for r in rs])),
               "floor": float(np.mean([r["floor"] for r in rs]))}
        summary[name] = row
        out.append(f"| {name} | {row['rows']} | {row['MAE']:.3f} | {row['bias']:+.3f} | {row['max']:.3f} | "
                   f"{row['within']:.0%} | {row['module_MAE']:.3f} | {row['MAE_E']:.3f} | {row['floor']:.3f} | "
                   f"{'**yes**' if name != 'ALL' and row['MAE'] > 0.05 else ''} |")
    # placement order within each (family, seed): does the theory rank placements as measured?
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["seed"])].append(r)
    taus = [kendall([r["T-relu"]["address"] for r in g], [r["measured"] for r in g]) for g in groups.values() if len(g) > 2]
    worst = sorted(rows, key=lambda r: -abs(r["err"]))[:10]
    out += ["", f"Kendall tau of the placement order within each (family, seed), mean over {len(taus)} groups: "
            f"{np.nanmean(taus):.2f}", "", "Worst 10 conditions (T-relu predicted vs measured address accuracy):", "",
            "| condition | P | predicted | measured | error | error-law-only (E) |", "|---|---|---|---|---|---|"]
    out += [f"| {r['id']} | {r['P']} | {r['T-relu']['address']:.3f} | {r['measured']:.3f} | {r['err']:+.3f} | "
            f"{r['E']['address']:.3f} |" for r in worst]
    overall = summary["ALL"]
    passed = overall["MAE"] <= 0.03 and overall["within"] >= 0.90
    out += ["", f"Pre-stated bars: overall MAE <= 0.03 and >= 90% of rows within 0.05. "
            f"Result: MAE {overall['MAE']:.3f}, {overall['within']:.0%} within 0.05 -> **{'PASS' if passed else 'FAIL'}**."]
    (HERE / "results.md").write_text("\n".join(out) + "\n")
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "rows": rows, "tau": taus}, indent=1, default=float))
    print("\n".join(out))


if __name__ == "__main__":
    main()
