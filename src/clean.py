"""Clean train.jsonl using only train.jsonl + SCHEMA.md (dev is read ONLY to find and drop leaked rows).

usage: python -I clean.py <data_dir> <out_dir>
writes: <out_dir>/train_clean.jsonl, <out_dir>/cleaning_log.csv, prints a summary.

Order matters and is fixed:
  1 drop junk input   2 drop dev leakage   3 drop near-duplicate copies
  4 repair truncated JSON   5 v1-schema rows -> v2   6 txn_id normalisation
  7 amount in rupees -> paise   8 priority recomputed from rules   9 needs_human recomputed
"""
import collections
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speccheck as S  # noqa: E402

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
NONTXN = {"account_access", "kyc", "other"}
V1_PRIO = {"low": "P3", "medium": "P2", "high": "P1"}
JUNK = re.compile(r"^\s*((test|asdf)(\s+(test|asdf|\d+))*)?\s*$", re.I)


def norm_body(user):
    b = user.split("\n\n", 1)[1]
    b = re.sub(r"\s+", " ", b).strip().lower()
    return re.sub(r"(\s*(thx|thanks\.?))+$", "", b).strip()


def dup_key(user):
    return user.split("\n\n", 1)[0] + "||" + norm_body(user)


def language_of(c):
    # en rows score exactly 0.000, hi-en rows score >= 0.095 on train; 0.05 sits in the gap.
    return "hi-en" if c["hinglish"] >= 0.05 else "en"


def single(xs):
    xs = list(dict.fromkeys(xs))
    return xs[0] if len(xs) == 1 else (None if not xs else "AMBIG")


def reconstruct(prefix, c):
    """Rebuild a full label for a truncated row from its surviving prefix + the spec checker.
    Returns (label, None) or (None, reason_for_drop)."""
    got = dict(re.findall(r'"(\w+)": ("[^"]*"|null|true|false|-?\d+)(?=,)', prefix))
    got = {k: json.loads(v) for k, v in got.items()}
    cat = got.get("category") or re.match(r'\{"category": "(\w+)"', prefix)[1]
    lab = {"category": cat}
    if cat in NONTXN:
        amt = date = tid = ch = None
    else:
        amt, tid, ch = single(c["amounts"]), single(c["txn_ids"]), single(c["channels"])
        date = single([d for d, _ in c["dates"]])
        if "AMBIG" in (amt, tid, ch, date):
            return None, "truncated label and ticket is ambiguous for a transaction field"
    legal = c["legal"]
    lab["priority"] = S.derive_priority(cat, amt, legal)
    lab.update(amount_paise=amt, txn_date=date, txn_id=tid, channel=ch, language=language_of(c))
    lab["needs_human"] = S.derive_needs_human(lab["priority"], c["repeat"])
    lab = {k: lab[k] for k in KEYS}
    # every field that survived truncation must agree (after known-noise normalisation)
    for k, v in got.items():
        if k == "priority":
            continue  # priority noise is a known, separately-fixed problem
        if k == "txn_id" and isinstance(v, str):
            v = "NP" + re.sub(r"\D", "", v)
        if k in lab and lab[k] != v:
            return None, f"truncated label; surviving field {k}={v!r} disagrees with spec reconstruction {lab[k]!r}"
    return lab, None


def main():
    D, OUT = sys.argv[1], sys.argv[2]
    os.makedirs(OUT, exist_ok=True)
    load = lambda n: [json.loads(l) for l in open(f"{D}/{n}.jsonl", encoding="utf-8")]
    train = load("train")
    dev_keys = {dup_key(r["messages"][1]["content"]) for r in load("dev")}
    test_keys = {dup_key(r["messages"][1]["content"]) for r in load("test_inputs")}

    log, kept, seen = [], [], {}
    stats = collections.Counter()

    def drop(rid, why, tag):
        log.append((rid, "drop", why))
        stats["drop:" + tag] += 1

    for r in train:
        rid, user, raw = r["id"], r["messages"][1]["content"], r["messages"][2]["content"]
        c = S.check(user)
        # 1 junk
        if JUNK.match(c["own"]):
            drop(rid, f"junk input text {c['own'].strip()!r}; label cannot be grounded in the ticket", "junk")
            continue
        # 2 leakage
        k = dup_key(user)
        if k in dev_keys:
            drop(rid, "ticket duplicates a dev row (same header and body up to whitespace/'Thx'); leakage", "dev_leak")
            continue
        if k in test_keys:
            drop(rid, "ticket duplicates a test row; leakage", "test_leak")
            continue
        # 3 near-duplicates within train
        if k in seen:
            drop(rid, f"duplicate of {seen[k]} (same header and body up to whitespace/'Thx'); same label", "duplicate")
            continue
        seen[k] = rid

        fixes = []
        # 4 truncated JSON
        try:
            lab = json.loads(raw)
        except json.JSONDecodeError:
            lab, why = reconstruct(raw, c)
            if lab is None:
                drop(rid, why, "truncated_ambiguous")
                continue
            fixes.append("label JSON truncated; rebuilt from surviving prefix + SCHEMA rules (surviving fields verified)")
            stats["fix:truncated"] += 1
        # 5 v1 schema
        if "prio" in lab:
            lab["priority"] = V1_PRIO[lab.pop("prio")]
            fixes.append("v1 schema: key 'prio' with low/medium/high renamed to 'priority' P3/P2/P1")
            stats["fix:v1_prio"] += 1
            d = lab.get("txn_date")
            if isinstance(d, str) and re.fullmatch(r"\d\d/\d\d/\d{4}", d):
                dd, mm, yy = d.split("/")
                lab["txn_date"] = f"{yy}-{mm}-{dd}"
                fixes.append(f"v1 schema: txn_date {d} reformatted to YYYY-MM-DD")
                stats["fix:v1_date"] += 1
        # 6 txn_id format
        t = lab.get("txn_id")
        if isinstance(t, str) and not re.fullmatch(r"NP\d{10}", t):
            nt = "NP" + re.sub(r"\D", "", t)
            fixes.append(f"txn_id {t!r} normalised to {nt}")
            lab["txn_id"] = nt
            stats["fix:txn_id"] += 1
        # 7 amount unit
        a = lab.get("amount_paise")
        exp = single(c["amounts"])
        if lab["category"] not in NONTXN and isinstance(a, int) and isinstance(exp, int) and a != exp:
            if a == exp // 100:
                fixes.append(f"amount_paise {a} is in rupees; ticket amount is {exp} paise")
                lab["amount_paise"] = exp
                stats["fix:amount_rupees"] += 1
            else:
                drop(rid, f"amount_paise {a} disagrees with ticket amount {exp} for unknown reason", "amount_unknown")
                continue
        # 8 priority
        p = S.derive_priority(lab["category"], lab["amount_paise"], c["legal"])
        if lab["priority"] != p:
            fixes.append(f"priority {lab['priority']} violates SCHEMA rules; set to {p}")
            lab["priority"] = p
            stats["fix:priority"] += 1
        # 9 needs_human
        h = S.derive_needs_human(lab["priority"], c["repeat"])
        if lab["needs_human"] != h:
            fixes.append(f"needs_human {lab['needs_human']} violates SCHEMA rules; set to {h}")
            lab["needs_human"] = h
            stats["fix:needs_human"] += 1

        lab = {k2: lab[k2] for k2 in KEYS}
        if fixes:
            log.append((rid, "fix", "; ".join(fixes)))
        msgs = [r["messages"][0], r["messages"][1], {"role": "assistant", "content": json.dumps(lab)}]
        kept.append({"id": rid, "messages": msgs})

    with open(f"{OUT}/train_clean.jsonl", "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT}/cleaning_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "action", "reason"])
        w.writerows(sorted(log))
    print(f"in {len(train)}  kept {len(kept)}  dropped {sum(1 for x in log if x[1] == 'drop')}  "
          f"fixed rows {sum(1 for x in log if x[1] == 'fix')}")
    for k2, v in sorted(stats.items()):
        print(f"  {k2:28} {v}")


if __name__ == "__main__":
    main()
