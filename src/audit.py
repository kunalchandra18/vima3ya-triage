"""Compare every parseable train label with the spec checker, field by field.

usage: python -I audit.py <data_dir> [field_to_dump] [max_examples]
"""
import collections
import json
import sys

import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speccheck as S  # noqa: E402

D = sys.argv[1]
DUMP = sys.argv[2] if len(sys.argv) > 2 else None
NMAX = int(sys.argv[3]) if len(sys.argv) > 3 else 15
NONTXN = {"account_access", "kyc", "other"}


def load(n):
    return [json.loads(l) for l in open(f"{D}/{n}.jsonl", encoding="utf-8")]


def issues(lab, c):
    """Return {field: (label_value, expected_or_hint)} for disagreements."""
    out = {}
    cat = lab.get("category")
    pri = lab.get("priority", lab.get("prio"))
    # transaction fields must be null for non-transaction categories
    if cat in NONTXN:
        for k in ("amount_paise", "txn_date", "txn_id", "channel"):
            if lab.get(k) is not None:
                out[k + "_nonnull"] = (lab.get(k), None)
    else:
        ids = c["txn_ids"]
        exp = ids[0] if len(ids) == 1 else (None if not ids else ids)
        if lab.get("txn_id") != exp:
            out["txn_id"] = (lab.get("txn_id"), exp)
        am = c["amounts"]
        exp = am[0] if len(am) == 1 else (None if not am else am)
        if lab.get("amount_paise") != exp:
            out["amount_paise"] = (lab.get("amount_paise"), exp)
        ds = [d for d, _ in c["dates"]]
        exp = ds[0] if len(set(ds)) == 1 else (None if not ds else ds)
        if lab.get("txn_date") != exp:
            out["txn_date"] = (lab.get("txn_date"), exp)
        ch = c["channels"]
        exp = ch[0] if len(ch) == 1 else (None if not ch else ch)
        if lab.get("channel") != exp:
            out["channel"] = (lab.get("channel"), exp)
    # priority recomputed from the label's own category + amount + text legal threat
    amt = lab.get("amount_paise") if isinstance(lab.get("amount_paise"), int) else None
    exp_p = S.derive_priority(cat, amt, c["legal"])
    if pri != exp_p:
        out["priority"] = (pri, exp_p)
    exp_h = S.derive_needs_human(exp_p, c["repeat"])
    if lab.get("needs_human") != exp_h:
        out["needs_human"] = (lab.get("needs_human"), exp_h)
    exp_l = "hi-en" if c["hinglish"] >= 0.05 else "en"
    if lab.get("language") != exp_l:
        out["language"] = (lab.get("language"), f"{exp_l} (score {c['hinglish']:.2f})")
    return out


def main():
    rows = load("train")
    agg = collections.Counter()
    dump = []
    for r in rows:
        try:
            lab = json.loads(r["messages"][2]["content"])
        except json.JSONDecodeError:
            agg["unparseable"] += 1
            continue
        c = S.check(r["messages"][1]["content"])
        iss = issues(lab, c)
        for k in iss:
            agg[k] += 1
        if DUMP and any(k.startswith(DUMP) for k in iss):
            dump.append((r["id"], {k: v for k, v in iss.items() if k.startswith(DUMP)}, c["own"]))
    print("disagreements per field:", dict(agg.most_common()))
    for rid, iss, own in dump[:NMAX]:
        print(f"\n## {rid} {iss}\n{own.strip()[:600]}")
    if DUMP:
        print(f"\n[{len(dump)} rows flagged for {DUMP}]")


if __name__ == "__main__":
    main()
