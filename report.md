# NimbusPay ticket triage: report

**Final model:** `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (r=16, α=32, all linear layers, 18.46M trainable params), fp16 base, free Colab T4.
**Dev result (`score.py`, 200 rows, greedy, batch 1):** **93.5% exact match, 98.9% mean field accuracy, 100% valid JSON.**
The base model scores 0% under the strict scorer. All 400 test predictions are valid JSON, and re-loading the adapter from
disk reproduced 25/25 randomly sampled test predictions exactly.

## 1. Data audit (Part A)

The brief said the training data had problems, so before training I wanted a way to check every label rather than eyeball a
sample. Every field except `category` is a deterministic function of the ticket text under SCHEMA.md, so I wrote those rules as
code (`src/speccheck.py`) and diffed them against every train label (`src/audit.py`). I then read each disagreement. If the
checker was wrong I fixed the checker; if the label was wrong it went into the log. Category has no rule to check against, so I
grouped tickets by their core intent sentence to look for minority labels, and trained an out-of-fold TF-IDF + logistic
regression to flag labels the rest of the data disagrees with. Where the spec was ambiguous I followed what the majority of
train labels did, not dev. Only after the cleaning was frozen did I run the checker once on dev, as a sanity check: it agrees
with **200/200 dev labels on all 7 non-category fields**.

| # | Problem | Rows | How I found it | What I did |
|---|---|---|---|---|
| 1 | Label JSON cut off mid-object | 25 | `json.loads` fails; every one starts `{"category": "` and never closes | **Fixed:** rebuilt from the surviving prefix + the spec rules, requiring every surviving field to agree. The same rebuild reproduces 1274/1274 known-good labels |
| 2 | Old schema v1: `prio` = low/medium/high, 30 also with `DD/MM/YYYY` dates | 70 (all IVR) | key-set check; the low/medium/high → P3/P2/P1 mapping matches the v2 rules in 70/70 | **Fixed:** renamed + mapped, ISO dates |
| 3 | `txn_id` not normalized (`NP-…`, `np 12345 67890`) | 20 | regex `^NP\d{10}$` | **Fixed** per spec |
| 4 | `amount_paise` holding rupees (`Rs 1,68,787.25` → `168787`) | 25 | checker ≠ label, and label = ⌊checker/100⌋ every time | **Fixed** with the exact paise from the text |
| 5 | `priority` contradicting the rules (31 P1 with no trigger, 15 P3 that are fraud/≥₹50k/legal, 14 P3 that should be P2) | 60 | recomputed from each label's own category + amount + legal words | **Fixed.** Only the priority value was corrupted: `needs_human` already matches the *corrected* priority in 60/60 rows (e.g. tr-00184: KYC + "lawyer… legal notice", labelled P3 but needs_human=true) |
| 6 | Junk inputs with full labels (`test`, `asdf asdf`, empty) | 10 | the amounts and ids in their labels appear nowhere in the text | **Dropped** |
| 7 | Train rows that are dev tickets (30 identical, 15 differing only by whitespace or a trailing "Thx") | 45 | normalized header+body match | **Dropped**: training on them would inflate the dev score (no test overlap found) |
| 8 | Duplicates inside train (60 exact + 31 whitespace variants, identical labels) | 91 copies | same normalization | **Dropped** the extra copies |

That takes train from 1445 to **1299 rows** (146 dropped, 200 fixed). Every action is in `cleaning_log.csv`, and re-auditing
the cleaned file gives zero disagreements. `category` came out clean: the out-of-fold model agrees on 97.4%, and when I read all
37 disagreements the label was right every time. `channel`, `language`, `needs_human`, the system prompt and the JSON types
were also clean. Three ambiguous rules were settled by what train consistently does: any mention of "card" (except "scratch
card") means `channel=card` (235/235 rows); an earlier ticket number or a quoted auto-reply doesn't count as "contacted before";
and balance/limit/fee amounts are never the transaction amount. I deliberately kept typos, double spaces and long quoted
threads, because dev and test have them too and the model needs to see them.

## 2. Baselines (Part B), dev

| Model | Valid JSON (strict) | Exact | Mean field | With lenient parsing (first `{…}` block): exact / field |
|---|---|---|---|---|
| B1: base model, fixed prompt | 0% | 0% | 0% | 0% / 0% (it writes JSON, but invents its own keys such as `"status"`) |
| B2: base model, SCHEMA.md in prompt | 0% | 0% | 0% | 0.5% / 61.5% |

Both baselines wrap their answer in a markdown code fence, so the strict scorer gives them nothing. The lenient column shows
what B2 does once you look past that. It knows the format but not the rules: priority 35%, needs_human 30%, txn_id
normalization 64%, language 64%, category 67%.

## 3. Training pipeline (Part C)

I wrote a small PyTorch training loop on top of `transformers` + `peft` (`src/triage_lib.py`) instead of using a trainer
wrapper, so that masking and loss normalization are explicit and don't change with library versions. A CPU test checks that my
loss matches Hugging Face's own masked loss exactly.

* **Chat template:** the model's own `apply_chat_template` (Qwen2.5 has a native system role). Each example is
  `template(system, user, add_generation_prompt=True)` + label + `<|im_end|>`, and I assert that this equals the template's
  rendering of the full conversation.
* **Loss** is only on the label and the closing `<|im_end|>`, so the model also learns to stop. Prompt tokens are masked.
  Logits are only computed at answer positions; with a 152k vocabulary that saves gigabytes on long tickets.
* **Sequence length, measured with the Qwen tokenizer** (prompt + label): train p50 167, p90 213, p99 1817, max 2214;
  test prompts max 2089; labels max 84 tokens. The long tail is the quoted email threads. Truncating would cut off the label,
  so **MAX_LEN = 2304** covers every row. Only 2.8% of rows are over 512 tokens, so I bucket batches by length (≤8192 padded
  tokens) instead of padding everything to 2k.
* **Settings:** LoRA on q,k,v,o,gate,up,down (r=16, α=32, dropout 0.05). That's 18.46M params; r=64 on all layers would break
  the 50M cap. AdamW, cosine schedule with 5% warmup, 16 examples per step, 2 epochs, gradient clipping at 1, fp16 autocast with
  a GradScaler (the T4 has no bf16), gradient checkpointing, seed 42. **Learning rate 5e-4**, chosen from the experiments below.
* **Inference matches the grading protocol:** base + adapter re-loaded from disk, chat template, plain greedy decoding
  (`do_sample=False`, `repetition_penalty=1.0`, `max_new_tokens=200`), one ticket at a time, nothing but `.strip()`. The
  model's hub `generation_config` sets `repetition_penalty=1.1`; with that default instead, 49/50 dev outputs are identical.
  Batched and one-at-a-time decoding gave identical outputs on 200/200 dev tickets.

## 4. Controlled experiments (Part D), dev, each against E0

| Run | Change vs E0 | Exact | Mean field | Priority | Peak GPU (GiB) | Train (min) |
|---|---|---|---|---|---|---|
| E0 | cleaned data, fp16 LoRA, lr 2e-4 | 71.0% | 95.7% | 79.5% | 5.60 | 9.2 |
| E1 | raw `train.jsonl` | 73.5% | 95.4% | 75.5% | 5.63 | 10.2 |
| E2 | 4-bit NF4 base (QLoRA) | 62.5% | 93.6% | 67.5% | 5.09 | 11.2 |
| E3a | lr 1e-4 | 59.5% | 92.9% | 65.5% | 6.86* | 9.2 |
| E3b | lr 5e-4 | 91.5% | 98.3% | 93.0% | 5.60 | 9.2 |
| E4 | + 700 spec-driven synthetic tickets | 90.0% | 98.2% | 92.5% | 5.64 | 15.3 |
| **E5 (final)** | E4 data + lr 5e-4 (two changes; not a controlled run) | **93.5%** | **98.9%** | **97.0%** | 8.59* | 15.3 |

\*The peak-memory counter also included objects still alive in the notebook from the previous cell; the like-for-like number is about 5.6 GiB.

**Raw vs clean.** At first glance raw data wins (73.5% vs 71.0%), but that's the leakage. E1 scores 77.8% on the 45 dev
tickets that were sitting in raw train, and 72.3% on the other 155; E0 gets 71% on both groups. On genuinely unseen tickets the
noisy labels hurt the field the noise targets most: priority drops from 79.5% to 75.5%.
**QLoRA** saved only 0.5 GiB, because `prepare_model_for_kbit_training` upcasts the 233M-parameter embedding to fp32. It was 22%
slower and lost 8.5 points. For a 1.5B model on a 15 GB GPU, 4-bit isn't worth it.
**Learning rate** turned out to be the biggest lever. At 2e-4 for two epochs the model hadn't learned the rules yet
(priority 79.5%); 5e-4 added 20.5 points of exact match, and 1e-4 was worse still.

**My own idea: spec-driven augmentation (E4).** Counting how often each SCHEMA.md rule shows up in train, several barely
appear: "parso" in 5 rows, "kal" 13, lawyer/legal notice 12, fee distractors 9, Visa/Mastercard 4, order numbers 0, and *no*
year-less dates that roll back to the previous year. `src/aug.py` takes clean tickets and edits them to exercise those rules
(amounts around ₹5k/₹50k, relative and year-less dates, distractor amounts and reference numbers, legal and repeat-contact
sentences, quoted threads, oddly spaced ids). It then recomputes every non-category field with the checker. It uses only train
and SCHEMA.md, and throws away anything identical to a dev or test input. On its own it added 19 points. On top of lr 5e-4
(E5 vs E3b) it fixed 10 priority, 4 category and 3 needs_human errors, but it also broke 3 dates, 2 priorities near the
threshold, 1 amount and 1 category. Section 5 explains why.

## 5. Error analysis (Part E): the final model's 13 dev failures

The wrong fields: priority 6, txn_date 4, needs_human 4 (always alongside priority), category 3, amount 1. By category,
payment_failed is weakest at 87%, then refund 91% and offers 94%; account_access, fraud, KYC and other are all 100%. On the
harder slices: year-less dates 82% (28 rows), quoted threads 2/3, Hinglish 94%, relative dates 93%, and distractor amounts,
legal threats and lakh/k amounts all 100%.

| id | Wrong | Gold → Pred | What I think happened |
|---|---|---|---|
| dv-00056 | priority, needs_human | P2 → P1 | ₹44,206.75 is below ₹50k, but the model treats "mid-40-thousands" as P1 |
| dv-00085 | priority, needs_human | P2 → P1 | Same with ₹48,966.75 in a Hinglish ticket. The augmentation drilled 49,999 vs 50,000, and "close to 50k" bled into P1 |
| dv-00148 | priority | P2 → P1 | ₹42,062 duplicate charge, with a quoted thread and a repeat-contact line also in play |
| dv-00037 | priority, needs_human | P1 → P2 | "284669 rupees" with no commas: the size (₹2.8 lakh) is misread |
| dv-00054 | priority | P2 → P3 | "6893 rupaye" is over the ₹5k payment_failed line; missed in Hinglish |
| dv-00005 | txn_date | 2026-08-01 → 2026-07-01 | "Dated Aug 1": shifted a month back |
| dv-00066 | txn_date | 2026-07-01 → 2026-06-01 | "Dated Jul 1": the same "Month 1" pattern; only appeared after augmentation |
| dv-00045 | txn_date | 2026-05-27 → 2025-05-27 | "27 May" on a ticket received 14 June is this year. The augmentation's previous-year rule was over-applied |
| dv-00090 | txn_date | 07-10 → 07-11 | "parso" (two days ago) treated as "kal" (yesterday); parso has only 5 train examples |
| dv-00010 | category (+priority, needs_human) | payment_failed → fraud | "I didn't see the money I added to my card": "didn't…" reads like a fraud report |
| dv-00177 | category | payment_failed → fraud | "red flag on my top up": fraud vocabulary in what is really a failed top-up |
| dv-00019 | category | refund → **`product`** | Invented a category outside the schema for "My product was faulty" |
| dv-00136 | amount | 4499 → 449900 | "Rs 44.99": dropped the decimal point |

Most of these come down to three causes. **(a) Comparing numbers:** priority is a threshold on the amount, and the model
approximates it from how the number looks. So it slips just under ₹5k/₹50k and on unusual formats (5 of the 13).
**(b) Date arithmetic:** 3 of the 4 date errors are new with augmentation, which tells me the synthetic rollback-to-last-year
dates (35% of the augmented dates) were too strong a signal. **(c) Category cues:** fraud-sounding words inside failed-payment
tickets, plus one invented label, which constrained decoding would rule out if post-processing were allowed.

**With one more week** I would (1) rebalance the augmentation: fewer rollback dates, more amounts *just under* the thresholds
and more "Month 1" dates, and build a held-out per-rule test set the same way so each rule gets its own score instead of one
dev number; (2) try a third epoch at 5e-4, or r=32, against that set; (3) add category paraphrases where fraud-like words appear
in non-fraud tickets; (4) since post-processing is banned, keep fixing format problems through the data, and check calibration
so uncertain tickets can be routed to a human.

## 6. AI tools used

I used Claude (Anthropic) heavily throughout, which the brief allows. It wrote the audit tooling (`speccheck.py`, `audit.py`,
the category checks), `clean.py`, `aug.py`, the training and inference module `triage_lib.py` with its CPU test, and the
notebook. At my request it also ran the notebook in my own Colab session through my browser, including fixing a `torchao` /
`peft` version clash that came up mid-run, and it drafted this report from the notebook outputs. Every number here comes from
`notebook.ipynb` and `logs/`. The decisions are mine to defend, and I've gone through them.
