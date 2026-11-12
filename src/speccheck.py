"""Rule-based implementation of SCHEMA.md (v2), used ONLY to audit train labels.

It is never used at inference time (the brief forbids post-processing).
Every rule below is traceable to a line in SCHEMA.md; anything the spec does not
pin down is marked AMBIGUOUS and is decided by the train-label majority, not by dev.
"""
import datetime as dt
import re

MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}

HINDI_WORDS = set("""
hai hain tha thi nahi nahin kar karo karna karun kaise kya kyun mera mere meri mujhe
ho gaya gayi gaye raha rahi rahe hua hui se ka ki ke ko bhi abhi yeh ye aur par lekin
paise paisa rupaye kat kaat liye diya wapas chahiye kab tak baar baat pehle mein
jaldi bhai dost usko mile mila aaya aayi bata batao kahan wala wali dikha karke liya
""".split())

LEGAL = re.compile(r"(?i)\b(rbi|ombudsman|consumer\s+court|lawyer|legal\s+notice|legal\s+action|vakil|court)\b")
REPEAT = re.compile(
    r"(?i)(third time|second time|again and again|baar baar|teesri baar|dusri baar|doosri baar|"
    r"contacted (?:support|you) (?:about this )?before|second complaint|already written|"
    r"pehle bhi complaint|already (?:raised|complained|contacted)|kai baar|phir se complaint|"
    r"\d+(?:st|nd|rd|th)? time)")


def split_ticket(user):
    head, body = user.split("\n\n", 1)
    m = re.match(r"Received: (\d{4}-\d\d-\d\d) \(\w{3}\)\nSource: (\S+)", head)
    recv = dt.date.fromisoformat(m[1])
    lines = body.split("\n")
    own = "\n".join(l for l in lines if not l.lstrip().startswith(">"))
    return recv, m[2], own


# ---------------- txn_id ----------------
TXN = re.compile(r"(?i)(?<![a-z])n[\s\-_.#:]*p[\s\-_.#:]*((?:\d[\s\-]*){10})(?!\d)")


def txn_ids(text):
    out = []
    for m in TXN.finditer(text):
        d = re.sub(r"\D", "", m[1])
        if len(d) == 10:
            out.append("NP" + d)
    return out


# ---------------- amount ----------------
NUM = r"(\d[\d,]*(?:\.\d+)?)"
MULT = r"(k|lakhs?)\b"
AMT = re.compile(
    r"(?i)(?:(?:\brs\.?|\binr|₹)\s*" + NUM + r"(?:\s*/-)?(?:\s*" + MULT + r")?)"
    r"|(?:(?<![\d,.])" + NUM + r"\s*" + MULT + r")"
    r"|(?:(?<![\d,.])" + NUM + r"\s*(?:rupees|rupaye|rupay|rs)\b)")
DISTRACT = re.compile(r"(?i)(balance|limit|fee\b|fees|pade hain|bacha|minimum|maximum)")


def _val(num, mult):
    x = float(num.replace(",", ""))
    if mult and mult.lower() == "k":
        x *= 1000
    elif mult and mult.lower().startswith("lakh"):
        x *= 100000
    return int(round(x * 100))


def amounts(text):
    """Return (kept, distractor) amounts in paise.

    An amount is a distractor if a balance/limit/fee word appears between the previous
    amount (or sentence start) and it, or in the 15 characters right after it.
    NP transaction ids are masked first so their digits are never read as money.
    """
    masked = TXN.sub(lambda m: "#" * len(m[0]), text)
    keep, drop = [], []
    sents = re.split(r"(?<![Rr]s\.)(?<=[.!?])\s+|\n", masked)
    for s in sents:
        prev = 0
        for m in AMT.finditer(s):
            if m[1]:
                v = _val(m[1], m[2])
            elif m[3]:
                v = _val(m[3], m[4])
            else:
                v = _val(m[5], None)
            ctx = s[prev:m.start()] + " " + s[m.end():m.end() + 15]
            (drop if DISTRACT.search(ctx) else keep).append(v)
            prev = m.end()
    return keep, drop


# ---------------- txn_date ----------------
def _most_recent(day, month, recv):
    for y in (recv.year, recv.year - 1):
        try:
            d = dt.date(y, month, day)
        except ValueError:
            continue
        if d <= recv:
            return d
    return None


DATE_PATTERNS = [
    (re.compile(r"\b(\d{4})-(\d\d)-(\d\d)\b"), "iso"),
    (re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"), "dmy4"),
    (re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2})\b"), "dmy2"),
    (re.compile(r"(?i)\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?(?:,?\s+(20\d\d))?\b"), "dMy"),
    (re.compile(r"(?i)\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b"), "Mdy"),
    (re.compile(r"(?i)\bday before yesterday\b|\bparso\b"), "rel2"),
    (re.compile(r"(?i)\byesterday\b|\bkal\b"), "rel1"),
    (re.compile(r"(?i)\btoday\b|\baaj\b"), "rel0"),
    (re.compile(r"(?i)\b(\d+)\s+(?:days?\s+ago|din\s+pehle)\b"), "relN"),
    (re.compile(r"(?i)\b(?:a|one|ek)\s+(?:week\s+ago|hafte\s+pehle|hafta\s+pehle)\b"), "rel7"),
]


def txn_dates(text, recv):
    found = []
    taken = []
    for pat, kind in DATE_PATTERNS:
        for m in pat.finditer(text):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            d = None
            try:
                if kind == "iso":
                    d = dt.date(int(m[1]), int(m[2]), int(m[3]))
                elif kind == "dmy4":
                    d = dt.date(int(m[3]), int(m[2]), int(m[1]))
                elif kind == "dmy2":
                    d = dt.date(2000 + int(m[3]), int(m[2]), int(m[1]))
                elif kind == "dMy":
                    day, mon = int(m[1]), MONTHS[m[2][:3].lower()]
                    d = dt.date(int(m[3]), mon, day) if m[3] else _most_recent(day, mon, recv)
                elif kind == "Mdy":
                    day, mon = int(m[2]), MONTHS[m[1][:3].lower()]
                    d = dt.date(int(m[3]), mon, day) if m[3] else _most_recent(day, mon, recv)
                elif kind == "rel0":
                    d = recv
                elif kind == "rel1":
                    d = recv - dt.timedelta(days=1)
                elif kind == "rel2":
                    d = recv - dt.timedelta(days=2)
                elif kind == "relN":
                    d = recv - dt.timedelta(days=int(m[1]))
                elif kind == "rel7":
                    d = recv - dt.timedelta(days=7)
            except ValueError:
                d = None
            if d:
                taken.append((m.start(), m.end()))
                found.append((m.start(), d.isoformat(), kind))
    found.sort()
    return [(d, k) for _, d, k in found]


# ---------------- channel ----------------
CHAN = [
    ("netbanking", re.compile(r"(?i)\b(net\s?banking|internet banking)\b")),
    ("upi", re.compile(r"(?i)\bupi\b")),
    ("wallet", re.compile(r"(?i)\bwallet\b")),
    # Train labels (235/235 rows) treat ANY mention of "card" as channel=card, except
    # "scratch card" (a reward, not a payment mode). AMBIGUOUS in spec; train majority decides.
    ("card", re.compile(r"(?i)(?<!scratch )(?<!srcatch )\bcards?\b|\bvisa\b|\bmastercard\b")),
]


def channels(text):
    return [c for c, p in CHAN if p.search(text)]


# ---------------- language ----------------
def hinglish_score(text):
    toks = re.findall(r"[a-z]+", text.lower())
    if not toks:
        return 0.0
    return sum(t in HINDI_WORDS for t in toks) / len(toks)


# ---------------- derived fields ----------------
def derive_priority(cat, amount, legal):
    if cat == "fraud" or (amount is not None and amount >= 50_000 * 100) or legal:
        return "P1"
    if (cat in ("payment_failed", "refund") and amount is not None and amount >= 5_000 * 100) or cat == "account_access":
        return "P2"
    return "P3"


def derive_needs_human(priority, repeat):
    return priority == "P1" or bool(repeat)


def check(user):
    """Extract everything the spec lets us compute from text alone (category excluded)."""
    recv, src, own = split_ticket(user)
    keep, drop = amounts(own)
    return {
        "recv": recv, "source": src, "own": own,
        "txn_ids": txn_ids(own),
        "amounts": keep, "distractor_amounts": drop,
        "dates": txn_dates(own, recv),
        "channels": channels(own),
        "legal": bool(LEGAL.search(own)),
        "repeat": bool(REPEAT.search(own)),
        "hinglish": hinglish_score(own),
    }
