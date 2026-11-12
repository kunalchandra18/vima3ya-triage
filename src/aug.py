"""Spec-driven augmentation (Part D, experiment 4). Uses ONLY train_clean.jsonl + SCHEMA.md.

Why: several rules in SCHEMA.md are rare or absent in train (parso 5 rows, kal 13, fee distractors 9,
lawyer/legal-notice 12, Visa/Mastercard 4, order numbers 0, year-less dates that roll back to the
previous year 0, quoted threads 36). We create variants of clean train tickets that exercise exactly
those rules, and recompute every non-category field with speccheck.py (which agrees with 100% of
dev labels on those fields). Category is copied from the seed; no transform changes the intent.

usage: python -I aug.py <train_clean.jsonl> <dev.jsonl> <test_inputs.jsonl> <out.jsonl> [n] [seed]
"""
import datetime as dt
import json
import random
import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import speccheck as S  # noqa: E402

KEYS = ["category", "priority", "amount_paise", "txn_date", "txn_id", "channel", "language", "needs_human"]
NONTXN = {"account_access", "kyc", "other"}
MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_FULL = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]

# ---- sentence banks (English / Hinglish). Most are harvested from train; the few marked SPEC are
# written from SCHEMA.md wording because train has no example of that rule.
LEGAL = {
    "en": ["I will complain to the RBI ombudsman.", "I will escalate this to the RBI.", "I will go to consumer court.",
           "My lawyer will send a legal notice.", "Otherwise I will approach the consumer court.",
           "I will take legal action.",  # SPEC
           "I will file a complaint with the banking ombudsman.",  # SPEC
           "You will receive a legal notice from my lawyer."],  # SPEC
    "hi-en": ["Consumer court jaunga.", "RBI mein complaint kar dunga.", "Lawyer se legal notice bhejunga.",  # SPEC
              "Ombudsman ke paas complaint karunga.", "Legal action lunga."],  # SPEC
}
REPEAT = {
    "en": ["This is the third time I am contacting you.", "I keep raising this again and again.",
           "This is my second complaint on the same issue.", "I have contacted support about this before.",
           "I have already written {n} times about this."],
    "hi-en": ["Teesri baar bol raha hoon.", "Baar baar complaint kar raha hoon.", "Pehle bhi complaint kar chuka hoon."],
}
DISTRACT = {
    "en": ["My account balance is {a} right now.", "FYI my daily limit is {a}.", "There was a fee of {a} on it.",  # fee: SPEC
           "The convenience fee was {a}.", "My monthly limit is {a}."],
    "hi-en": ["Account mein abhi {a} pade hain.", "Meri daily limit {a} hai.", "Fee {a} lagi thi."],
}
REFNUM = {
    "en": ["Bank ref no {b}.", "My earlier ticket number is #{t}.", "My order number is #{o}.",  # order: SPEC
           "Order ID {o}.", "UTR {b}."],
    "hi-en": ["Bank ref {b} hai.", "Pichla ticket number #{t} tha.", "Order number #{o} hai."],
}
CARDNET = {"en": ["I paid with my Visa card.", "It was a Mastercard payment."],
           "hi-en": ["Visa card se payment kiya tha.", "Mastercard se hua tha."]}
DATE_ABS = {"en": ["This happened on {d}.", "Transaction date {d}.", "Dated {d}.", "It took place on {d}.",
                   "Date of transaction: {d}."],
            "hi-en": ["Yeh {d} ko hua.", "{d} ki transaction hai.", "Transaction date {d} hai."]}
DATE_REL = {"en": {0: ["It was today.", "This happened today."], 1: ["It took place yesterday.", "That was yesterday."],
                   2: ["This happened day before yesterday.", "That was day before yesterday."],
                   "n": ["This happened {n} days ago.", "It was {n} days ago."]},
            "hi-en": {0: ["Yeh aaj hua."], 1: ["kal hi hua yeh.", "kal ki baat hai.", "Yeh kal hua."],
                      2: ["Yeh parso hua.", "parso ki baat hai."], "n": ["{n} din pehle ki baat hai.", "Yeh {n} din pehle hua."]}}
QUOTE = ("> On {d}, NimbusPay Support wrote:\n> Thank you for contacting NimbusPay. Your request has been registered "
         "with ticket #{t}. Our team will get back to you within 48 working hours. Please do not share your OTP, PIN or "
         "card details with anyone, including people who claim to be from NimbusPay. You can track the status of your "
         "request from the Help section of the app. If you have already received a resolution, please ignore this message. "
         "We value your trust and thank you for your patience while we look into this for you.")


def inr(x):  # Indian digit grouping, x in rupees (int)
    s = str(x)
    if len(s) <= 3:
        return s
    head, tail = s[:-3], s[-3:]
    head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
    return head + "," + tail


def render_amount(rng, paise):
    """Render an amount in one of the surface forms seen in train (labels are recomputed from text later)."""
    r, p = divmod(paise, 100)
    opts = [lambda: f"Rs {inr(r)}" + (f".{p:02d}" if p else ""), lambda: f"₹{inr(r)}", lambda: f"INR {inr(r)}",
            lambda: f"Rs.{r}/-", lambda: f"{r} rupees", lambda: f"Rs. {r}"]
    if p:  # keep paise exact: only forms that can show decimals
        opts = [lambda: f"Rs {inr(r)}.{p:02d}", lambda: f"₹{inr(r)}.{p:02d}", lambda: f"INR {inr(r)}.{p:02d}"]
    if p == 0 and r % 100 == 0 and r >= 1000:
        opts += [lambda: f"{r / 1000:g}k"] * 3
    if p == 0 and r % 1000 == 0 and r >= 100000:
        opts += [lambda: f"{r / 100000:g} lakh"] * 4
    return rng.choice(opts)()


def pick_amount(rng):
    """Biased to the 5k and 50k priority thresholds and to lakh/k renderings."""
    kind = rng.random()
    if kind < 0.2:
        r = rng.choice([4999, 5000, 5001, 4950, 5100, 49999, 50000, 50001, 49500, 51000])
    elif kind < 0.3:
        r = rng.choice([1, 1.2, 1.5, 2, 2.5, 3.25, 0.5, 4.8]) * 100000
    elif kind < 0.65:
        r = rng.choice(range(1000, 99000, 100))
    else:
        r = round(10 ** rng.uniform(1, 5.5))  # log-uniform 10 .. ~3 lakh rupees, like train
    p = rng.choice([0, 0, 0, 50, 75, 99, 25]) if r < 100000 and rng.random() < 0.3 else 0
    return int(r) * 100 + p


def render_txn(rng, tid):
    d = tid[2:]
    f = rng.choice([lambda: f"np {d[:5]} {d[5:]}", lambda: f"NP-{d}", lambda: f"N P {d[:4]} {d[4:7]} {d[7:]}",
                    lambda: f"np#{d}", lambda: f"NP {d[:5]}-{d[5:]}", lambda: f"Np.{d}", lambda: f"np{d}"])
    return f()


def render_date(rng, d, recv, lang):
    """Render date d (<= recv). Returns sentence or None."""
    delta = (recv - d).days
    if delta in (0, 1, 2) and rng.random() < 0.6:
        return rng.choice(DATE_REL[lang][delta])
    if 3 <= delta <= 9 and rng.random() < 0.5:
        return rng.choice(DATE_REL[lang]["n"]).format(n=delta)
    yearless = rng.random() < 0.55 or d.year < recv.year
    if yearless:
        txt = rng.choice([f"{d.day} {MON[d.month - 1]}", f"{MON[d.month - 1]} {d.day}"])
        if S._most_recent(d.day, d.month, recv) != d:
            return None
    else:
        txt = rng.choice([f"{d.day:02d}-{d.month:02d}-{d.year % 100:02d}", f"{d.day:02d}/{d.month:02d}/{d.year}",
                          f"{d.day}{'th' if 10 <= d.day % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d.day % 10, 'th')} {MONTH_FULL[d.month - 1]} {d.year}"])
    return rng.choice(DATE_ABS[lang]).format(d=txt)


def label_for(cat, user):
    """Full label from text using the spec checker; None if any field is ambiguous."""
    c = S.check(user)
    lab = {"category": cat}
    if cat in NONTXN:
        amt = date = tid = ch = None
    else:
        def one(xs):
            xs = list(dict.fromkeys(xs))
            return xs[0] if len(xs) == 1 else (None if not xs else "AMBIG")
        amt, tid, ch = one(c["amounts"]), one(c["txn_ids"]), one(c["channels"])
        date = one([d for d, _ in c["dates"]])
        if "AMBIG" in (amt, tid, ch, date):
            return None
    lab["priority"] = S.derive_priority(cat, amt, c["legal"])
    lab.update(amount_paise=amt, txn_date=date, txn_id=tid, channel=ch,
               language="hi-en" if c["hinglish"] >= 0.05 else "en")
    lab["needs_human"] = S.derive_needs_human(lab["priority"], c["repeat"])
    return {k: lab[k] for k in KEYS}


SENT_SPLIT = re.compile(r"(?<![Rr]s\.)(?<=[.!?])\s+")


def edit_body(user, fn):
    """Apply fn(list_of_sentences)->list to the main body line of the ticket (not quoted lines)."""
    head, body = user.split("\n\n", 1)
    lines = body.split("\n")
    idx = max((i for i, l in enumerate(lines) if not l.startswith(">")), key=lambda i: len(lines[i]))
    sents = [s for s in SENT_SPLIT.split(lines[idx]) if s]
    lines[idx] = " ".join(fn(sents))
    return head + "\n\n" + "\n".join(lines)


def find_sent(sents, pred):
    return [i for i, s in enumerate(sents) if pred(s)]


def transform(rng, row, lang, recv):
    user = row["messages"][1]["content"]
    cat = json.loads(row["messages"][2]["content"])["category"]
    applied = []
    txn = cat not in NONTXN
    # weighted, without replacement; legal/repeat kept rare so the P1/needs_human mix stays close to train
    w = {"legal": 0.6, "repeat": 0.6, "refnum": 2, "quote": 1.5, "distract": 2 if txn else 0.7}
    if txn:
        w.update(amount=3, date=3, txnid=2, card=0.8)
    k = rng.choice([1, 2, 2, 3])
    chosen = []
    while len(chosen) < k and w:
        op = rng.choices(list(w), weights=list(w.values()))[0]
        chosen.append(op)
        w.pop(op)
    for op in chosen:
        if op == "amount":
            def f(s):
                hit = find_sent(s, lambda x: S.amounts(x)[0])
                new = render_amount(rng, pick_amount(rng))
                if hit:
                    i = hit[0]
                    masked = S.TXN.sub(lambda m: "#" * len(m[0]), s[i])
                    m = S.AMT.search(masked)
                    if m:
                        s[i] = s[i][:m.start()] + new + s[i][m.end():]
                        return s
                s.insert(rng.randrange(1, len(s) + 1), rng.choice(["Amount: {a}.", "It was for {a}.", "Rakam {a} hai."
                                                                    if lang == "hi-en" else "{a} is the amount."]).format(a=new))
                return s
            user = edit_body(user, f)
        elif op == "date":
            # biased toward relative dates and year-less dates that resolve to the previous year
            if rng.random() < 0.35:
                # a calendar day later in the year than Received -> a year-less mention means LAST year
                later = [dt.date(recv.year - 1, m, dd) for m in range(recv.month, 13) for dd in range(1, 29)
                         if (m, dd) > (recv.month, recv.day)]
                d = rng.choice(later)
            else:
                d = recv - dt.timedelta(days=rng.choice([0, 1, 1, 2, 2, 3, 4, 5, 6, 8, 12, 20, 35, 60]))
            sent = render_date(rng, d, recv, lang)
            if not sent:
                continue

            def f(s, sent=sent):
                hit = find_sent(s, lambda x: S.txn_dates(x, recv))
                if hit:
                    s[hit[0]] = sent
                else:
                    s.insert(rng.randrange(1, len(s) + 1), sent)
                return s
            user = edit_body(user, f)
        elif op == "txnid":
            tid = "NP" + "".join(rng.choice("0123456789") for _ in range(10))
            txt = render_txn(rng, tid)

            def f(s, txt=txt):
                hit = find_sent(s, lambda x: S.txn_ids(x))
                if hit:
                    s[hit[0]] = S.TXN.sub(txt, s[hit[0]], count=1)
                else:
                    s.insert(rng.randrange(1, len(s) + 1), rng.choice(["Transaction ID {t}.", "Ref: {t}", "{t} check kar lo."
                                                                        if lang == "hi-en" else "The txn id is {t}."]).format(t=txt))
                return s
            user = edit_body(user, f)
        elif op == "card":
            def f(s):
                s = [x for x in s if not S.channels(x)]
                s.insert(rng.randrange(1, len(s) + 1), rng.choice(CARDNET[lang]))
                return s
            user = edit_body(user, f)
        elif op == "legal":
            user = edit_body(user, lambda s: s[:1] + [rng.choice(LEGAL[lang])] + s[1:] if rng.random() < 0.5
                             else s + [rng.choice(LEGAL[lang])])
        elif op == "repeat":
            user = edit_body(user, lambda s: s + [rng.choice(REPEAT[lang]).format(n=rng.randint(2, 5))])
        elif op == "distract":
            kind = rng.choice(["balance", "limit", "fee"] if txn else ["balance", "limit"])
            amt = {"balance": rng.randrange(1000, 20000000, 1), "limit": rng.randrange(5, 300) * 100000,
                   "fee": rng.randrange(500, 50000, 25)}[kind]
            pool = [t for t in DISTRACT[lang] if kind in t.lower() or (kind == "balance" and "pade" in t)]
            sent = rng.choice(pool).format(a=render_amount(rng, amt))
            user = edit_body(user, lambda s, sent=sent: (lambda i: s[:i] + [sent] + s[i:])(rng.randrange(1, len(s) + 1)))
        elif op == "refnum":
            sent = rng.choice(REFNUM[lang]).format(b=rng.randrange(10 ** 11, 10 ** 12), t=rng.randrange(10000, 99999),
                                                   o=rng.randrange(10 ** 7, 10 ** 9))
            user = edit_body(user, lambda s: (lambda i: s[:i] + [sent] + s[i:])(rng.randrange(1, len(s) + 1)))
        elif op == "quote":
            if "Source: email" not in user or "\n>" in user:
                continue
            n = rng.randint(1, 6)
            qs = []
            for _ in range(n):
                qd = recv - dt.timedelta(days=rng.randint(1, 60))
                qs.append(QUOTE.format(d=f"{qd.day:02d} {MON[qd.month - 1]} {qd.year}", t=rng.randint(10000, 99999)))
            user = user.rstrip("\n") + "\n\n" + "\n\n".join(qs)
        applied.append(op)
    return user, applied


def main():
    src, dev_p, test_p, out = sys.argv[1:5]
    n_target = int(sys.argv[5]) if len(sys.argv) > 5 else 700
    rng = random.Random(int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    load = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
    train = load(src)
    norm = lambda u: re.sub(r"\s+", " ", u).strip().lower()
    forbidden = {norm(r["messages"][1]["content"]) for r in load(dev_p) + load(test_p)}
    seen = {norm(r["messages"][1]["content"]) for r in train}
    synth, tries, opcount = [], 0, {}
    while len(synth) < n_target and tries < n_target * 20:
        tries += 1
        seed = rng.choice(train)
        lab0 = json.loads(seed["messages"][2]["content"])
        recv = S.split_ticket(seed["messages"][1]["content"])[0]
        user, ops = transform(rng, seed, lab0["language"], recv)
        if not ops or norm(user) in seen or norm(user) in forbidden:
            continue
        lab = label_for(lab0["category"], user)
        if lab is None or lab["language"] != lab0["language"]:
            continue
        seen.add(norm(user))
        for o in ops:
            opcount[o] = opcount.get(o, 0) + 1
        synth.append({"id": f"aug-{len(synth) + 1:05d}", "seed": seed["id"], "ops": ops,
                      "messages": [seed["messages"][0], {"role": "user", "content": user},
                                   {"role": "assistant", "content": json.dumps(lab)}]})
    with open(out, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for r in synth:
            f.write(json.dumps({"id": r["id"], "messages": r["messages"]}, ensure_ascii=False) + "\n")
    with open(out.replace(".jsonl", "_meta.jsonl"), "w", encoding="utf-8") as f:
        for r in synth:
            f.write(json.dumps({"id": r["id"], "seed": r["seed"], "ops": r["ops"]}) + "\n")
    print(f"clean rows {len(train)} + synthetic {len(synth)} (tries {tries}) -> {out}")
    print("ops applied:", dict(sorted(opcount.items())))


if __name__ == "__main__":
    main()
