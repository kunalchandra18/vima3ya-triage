# NimbusPay ticket triage: report

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` + LoRA (r=16, α=32, all linear layers, 18.46M trainable params), fp16 base, free Colab T4.
**Final dev result (`score.py`, 200 rows, greedy, batch 1):** **93.5% exact match, 98.9% mean field accuracy**, 100% valid JSON
(baselines: 0% strict). Test: 400/400 valid JSON; re-loading the adapter reproduced 25/25 random test predictions exactly.

## 1. Data audit (Part A)

**Method.** `src/speccheck.py` implements every SCHEMA.md rule that can be computed from text (all fields except `category`).
`src/audit.py` diffs it against every train label. Each disagreement was read: checker bugs were fixed in the checker, label
errors were logged. Category was checked two ways: grouping by the core intent sentence, and an out-of-fold char-n-gram
TF-IDF + logistic-regression model (confident-learning style). Spec ambiguities were settled by the **train-label majority**,
never dev. Once cleaning was frozen, the checker was run once on dev: it agrees with **200/200 dev labels on all 7 non-category fields**.

| # | Problem | Rows | How found | Action |
|---|---|---|---|---|
| 1 | Label JSON truncated mid-object | 25 | `json.loads` fails; all start `{"category": "` with no closing `}` | **fix**: rebuild from the surviving prefix + spec rules; every surviving field must agree. The same rebuild reproduces 1274/1274 known-good labels exactly |
| 2 | Old schema v1: `prio` = low/medium/high; 30 also have `DD/MM/YYYY` dates | 70 (all IVR) | key-set check; low/medium/high ↔ P3/P2/P1 matches the v2 rules in 70/70 | **fix**: rename + map, ISO dates |
| 3 | `txn_id` not normalized (`NP-…`, `np 12345 67890`) | 20 | regex `^NP\d{10}$` | **fix** per spec |
| 4 | `amount_paise` actually in rupees (`Rs 1,68,787.25` → `168787`) | 25 | checker ≠ label; label = ⌊checker/100⌋ every time | **fix**: exact paise from the text |
| 5 | `priority` breaks the rules (31 P1 with no trigger, 15 P3 that are fraud/≥₹50k/legal, 14 P3 that should be P2) | 60 | recompute from the label's own category + amount + legal words | **fix**. Only priority was corrupted: `needs_human` already matches the *corrected* priority in 60/60 rows (tr-00184: kyc + "lawyer… legal notice", P3, needs_human=true) |
| 6 | Junk inputs with full labels (`test`, `asdf asdf`, empty) | 10 | blank/short body; amounts and ids in the label appear nowhere in the text | **drop** |
| 7 | Leakage: train row = dev ticket (30 identical, 15 differ only by whitespace / trailing "Thx") | 45 | normalized header+body match | **drop** (no test overlap) |
| 8 | Duplicates inside train (60 exact + 31 whitespace variants; labels identical) | 91 copies | same normalization | **drop** extra copies |

1445 → **1299 rows** (146 dropped, 200 fixed). Every action and reason is in `cleaning_log.csv`. Re-auditing the cleaned
file gives zero disagreements. **Checked and clean:** `category` (OOF model agrees 97.4%; all 37 disagreements read by
hand and the labels were right), `channel`, `language`, `needs_human`, the system prompt, roles, types, key order.
**Ambiguities, settled by train majority:** any "card" mention except "scratch card" means `channel=card` (235/235). An
earlier ticket number or a quoted auto-reply is *not* "contacted before". Balance, limit and fee amounts are ignored. **Kept on
purpose:** typos, double spaces, quoted threads (dev and test contain them too).

## 2. Baselines (Part B), dev

| Model | JSON valid (strict) | Exact | Mean field | Lenient parse (first `{…}`): exact / field |
|---|---|---|---|---|
| B1 base, fixed prompt | 0% | 0% | 0% | 0% / 0% (valid 94%, but invents its own keys, e.g. `"status"`) |
| B2 base, SCHEMA.md in prompt | 0% | 0% | 0% | 0.5% / 61.5% |

Both wrap the JSON in markdown fences, so the strict scorer gives 0. Even with lenient parsing, B2 knows the keys but not the
rules: priority 35%, needs_human 30%, language 64%, txn_id 64% (normalization), category 67%.

## 3. Training pipeline (Part C)

* **Stack:** a plain PyTorch loop over HF `transformers` + `peft` (`src/triage_lib.py`). Masking and loss normalization are
  explicit. A CPU unit test checks that the loss equals HF's own masked loss exactly.
* **Template:** the model's own `apply_chat_template` (Qwen2.5 has a native system role). Training text =
  `template(system, user, add_generation_prompt=True)` + label + `<|im_end|>`, asserted equal to the template's rendering of
  the full conversation.
* **Loss** only on label tokens + `<|im_end|>` (so the model learns to stop); prompt tokens = −100; mean over all answer tokens
  in the optimizer step. Logits are computed only at answer positions, which saves GBs with a 152k vocabulary.
* **Sequence length, measured with the Qwen tokenizer** (prompt + label): train p50 167 / p90 213 / p99 1817 / max 2214; test
  prompts max 2089; labels max 84 tokens. The tail is quoted email threads. Truncation would cut the label, so **MAX_LEN = 2304**
  covers every row. Only 2.8% of rows exceed 512 tokens, so length-bucketed micro-batches (≤8192 padded tokens) avoid padding
  everything to 2k.
* **Settings:** LoRA r=16, α=32, dropout 0.05 on q,k,v,o,gate,up,down. 18.46M params, under the 50M cap (r=64 on all
  linear layers would be 73.9M). AdamW, cosine schedule with 5% warmup, 16 examples per step, 2 epochs, grad-clip 1, fp16
  autocast + GradScaler (T4 has no bf16), gradient checkpointing, seed 42. **Final LR 5e-4** (chosen by E3, below).
* **Inference = grading protocol:** base + adapter re-loaded from disk, chat template, pure greedy (`do_sample=False`,
  `repetition_penalty=1.0`, `max_new_tokens=200`), batch 1, only `.strip()`. The hub `generation_config` sets
  `repetition_penalty=1.1`. With it, outputs are identical on 49/50 dev tickets. Batched (16) and batch-1 decoding gave identical
  outputs on 200/200 dev tickets.

## 4. Controlled experiments (Part D), dev, each vs E0

| Run | What changed vs E0 | Exact | Mean field | Priority acc. | Peak GPU GiB | Train min |
|---|---|---|---|---|---|---|
| E0_ref | cleaned data, fp16 LoRA, lr 2e-4 | 71.0% | 95.7% | 79.5% | 5.60 | 9.2 |
| E1_raw | raw `train.jsonl` | 73.5% | 95.4% | 75.5% | 5.63 | 10.2 |
| E2_qlora | 4-bit NF4 base | 62.5% | 93.6% | 67.5% | 5.09 | 11.2 |
| E3a | lr 1e-4 | 59.5% | 92.9% | 65.5% | 6.86* | 9.2 |
| E3b | lr 5e-4 | 91.5% | 98.3% | 93.0% | 5.60 | 9.2 |
| E4_aug | + 700 spec-driven synthetic tickets | 90.0% | 98.2% | 92.5% | 5.64 | 15.3 |
| **E5 (final)** | E4 data **and** lr 5e-4 (two changes, not a controlled run) | **93.5%** | **98.9%** | **97.0%** | 8.59* | 15.3 |

\*`max_memory_allocated` also counted objects still alive in the notebook from the previous cell; the like-for-like figure is about 5.6 GiB.

* **Raw vs clean (E1):** the higher raw score is leakage. E1 gets 77.8% on the 45 dev tickets that sit in raw train, versus
  72.3% on the other 155. E0 gets 71% on both groups. On real labels, raw data hurts the field the noise targets:
  priority 75.5% vs 79.5%.
* **QLoRA (E2):** only 0.5 GiB less peak memory, because `prepare_model_for_kbit_training` upcasts the 233M-param embedding to
  fp32, and dequantization makes it 22% slower. It also loses 8.5 points of exact match. On a 1.5B model, 4-bit isn't worth it.
* **Learning rate (E3):** the biggest lever. At 2e-4 for 2 epochs the model underfits the rules (priority 79.5%); 5e-4 gives
  +20.5 points exact. 1e-4 is worse still.
* **Augmentation (E4, own idea).** Train under-covers rules that SCHEMA.md spells out: parso 5 rows, kal 13, lawyer/legal
  notice 12, fee distractors 9, Visa/Mastercard 4, order numbers 0, and no year-less dates that roll back to the previous year.
  `src/aug.py` edits clean tickets (amounts around ₹5k/₹50k, relative and year-less dates, distractors, legal and repeat
  sentences, quoted threads, odd id spacing) and recomputes every non-category field with the checker. It uses only train +
  SCHEMA.md and rejects any output equal to a dev or test input. Result: **+19 points exact** at lr 2e-4. On top of lr 5e-4
  (E5 vs E3b), it fixed 10 priority, 4 category and 3 needs_human errors, but **broke 3 dates**, 2 near-threshold priorities,
  1 amount and 1 category (see §5).

## 5. Error analysis (Part E): final model, all 13 dev failures

Wrong fields: priority 6, txn_date 4, needs_human 4 (always together with priority), category 3, amount 1. Per category:
payment_failed 87%, refund 91%, offers 94%; account_access, fraud, kyc and other are all 100%. Slices: year-less date 82% (28 rows),
quoted thread 2/3, Hinglish 94%, relative date 93%, distractor amount, legal threat and lakh/k all 100%.

| id | Wrong | Gold → Pred | Why |
|---|---|---|---|
| dv-00056 | priority, needs_human | P2 → P1 | ₹44,206.75 refund is below the ₹50k line; the model calls it P1 |
| dv-00085 | priority, needs_human | P2 → P1 | Same: ₹48,966.75 (Hinglish). Augmentation drilled 49,999/50,000, and "close to 50k" leaked into P1 |
| dv-00148 | priority | P2 → P1 | ₹42,062 duplicate payment, plus a quoted thread and a repeat-contact line |
| dv-00037 | priority, needs_human | P1 → P2 | "284669 rupees" with no commas: the magnitude (₹2.8 lakh ≥ ₹50k) is misread |
| dv-00054 | priority | P2 → P3 | "6893 rupaye" payment_failed is ≥ ₹5k; misses the lower threshold in Hinglish |
| dv-00005 | txn_date | 2026-08-01 → 2026-07-01 | "Dated Aug 1": shifted one month back |
| dv-00066 | txn_date | 2026-07-01 → 2026-06-01 | "Dated Jul 1": the same "Mon 1" pattern. New with augmentation |
| dv-00045 | txn_date | 2026-05-27 → 2025-05-27 | "27 May" received 14 Jun is *this* year: augmentation's previous-year rule over-applied |
| dv-00090 | txn_date | 07-10 → 07-11 | "parso" (2 days ago) read as "kal" (1 day ago); parso has only 5 train rows |
| dv-00010 | category (+P, NH) | payment_failed → fraud | "I didn't see the money I added to my card": "didn't…" looks like fraud |
| dv-00177 | category | payment_failed → fraud | "red flag on my top up": fraud vocabulary in a failed top-up |
| dv-00019 | category | refund → **`product`** | Invented a label outside the schema for "My product was faulty" |
| dv-00136 | amount | 4499 → 449900 | "Rs 44.99": decimal point dropped |

**Causes:** (a) **numeric comparison**: priority is a threshold function of the amount, and the model approximates it from surface
form, so it fails near ₹5k/₹50k and on uncommon number formats (5 of 13). (b) **Date arithmetic**: 3 of the 4 date errors are
new with augmentation, so the synthetic previous-year rollback cases (35% of augmented dates) were too strong a prior. (c) **Category
cues**: fraud vocabulary inside failed-payment tickets, plus one out-of-schema label, which constrained decoding would prevent.

**With one more week:** (1) rebalance augmentation (fewer rollback dates, more amounts *just below* thresholds and "Mon 1"
dates), and hold out a per-rule eval set built the same way; (2) a third epoch at 5e-4, or r=32, measured on that set;
(3) more category paraphrases where fraud vocabulary appears in non-fraud tickets; (4) since post-processing is banned,
teach format robustness through data, and measure calibration so uncertain tickets can be routed to `needs_human`.

## 6. AI tools used

I used Claude (Anthropic) extensively, as permitted. Claude wrote the audit tooling (`speccheck.py`, `audit.py`, the category
checks), `clean.py`, `aug.py`, the training/inference module `triage_lib.py` and its CPU test, and the notebook. At my request it
also ran the notebook in my own Colab session through my browser, including fixing a `torchao`/`peft` version clash it hit, and
it drafted this report from the notebook outputs. Every number here comes from `notebook.ipynb` and `logs/`. I am responsible for
the submission and can explain each decision.
