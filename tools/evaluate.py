#!/usr/bin/env python
"""Score a results file against a reference with the brief's three checks.

    python tools/evaluate.py --ref ref.jsonl --pred results.jsonl

Only videos present in the reference are scored. Missing seconds in the prediction count as
"no objects, 0 people".
"""
from __future__ import annotations

import argparse
import gzip
import json


def load(path):
    op = gzip.open if str(path).endswith(".gz") else open
    secs, errors = {}, set()
    with op(path, "rt") as f:
        for line in f:
            d = json.loads(line)
            if "error" in d:
                errors.add(d["video"])
                continue
            secs[(d["video"], d["second"])] = (frozenset(d["objects"]), d["people"])
    return secs, errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    ref, ref_err = load(args.ref)
    pred, pred_err = load(args.pred)
    videos = {v for v, _ in ref}
    tp = fp = fn = 0
    exact = within1 = n = 0
    missing = extra = 0
    for key, (robj, rp) in ref.items():
        if key in pred:
            pobj, pp = pred[key]
        else:
            missing += 1
            pobj, pp = frozenset(), 0
        tp += len(robj & pobj)
        fp += len(pobj - robj)
        fn += len(robj - pobj)
        n += 1
        exact += pp == rp
        within1 += abs(pp - rp) <= 1
    for key in pred:
        if key[0] in videos and key not in ref:
            extra += 1
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    res = {
        "videos": len(videos), "seconds": n, "f1": round(f1, 4),
        "people_exact": round(exact / max(1, n), 4), "people_within1": round(within1 / max(1, n), 4),
        "missing_seconds": missing, "extra_seconds": extra,
        "ref_errors": len(ref_err), "pred_errors": len(pred_err),
    }
    res["pass"] = res["f1"] >= 0.95 and res["people_exact"] >= 0.85 and res["people_within1"] >= 0.95
    print(json.dumps(res))


if __name__ == "__main__":
    main()
