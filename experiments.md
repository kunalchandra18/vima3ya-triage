| Run | What changed | Dev exact | Dev mean field | Peak GPU GiB | Train min | Train rows |
|---|---|---|---|---|---|---|
| E0_ref | reference: cleaned data, 16-bit LoRA r16, lr 2e-4, 2 epochs | 71.0% | 95.7% | 5.60 | 9.2 | 1299 |
| E1_raw | raw train.jsonl instead of cleaned | 73.5% | 95.4% | 5.63 | 10.2 | 1445 |
| E2_qlora | 4-bit NF4 QLoRA instead of fp16 base | 62.5% | 93.6% | 5.09 | 11.2 | 1299 |
| E3_lr1e-4 | learning rate 1e-4 instead of 2e-4 | 59.5% | 92.9% | 6.86 | 9.2 | 1299 |
| E3_lr5e-4 | learning rate 5e-4 instead of 2e-4 | 91.5% | 98.3% | 5.60 | 9.2 | 1299 |
| E4_aug | +700 spec-driven synthetic tickets (own idea) | 90.0% | 98.2% | 5.64 | 15.3 | 1999 |
| E5_aug_lr5e-4 | final: E4 data + lr 5e-4 (combines the two wins; not a one-change run) | 93.5% | 98.9% | 8.59 | 15.3 | 1999 |

Per-field dev accuracy:

| Run | category | priority | amount_paise | txn_date | txn_id | channel | language | needs_human |
|---|---|---|---|---|---|---|---|---|
| E0_ref | 96.5% | 79.5% | 99.5% | 98.0% | 100.0% | 99.5% | 100.0% | 92.5% |
| E1_raw | 98.0% | 75.5% | 100.0% | 98.5% | 100.0% | 100.0% | 100.0% | 91.5% |
| E2_qlora | 97.0% | 67.5% | 99.5% | 97.5% | 99.5% | 100.0% | 100.0% | 87.5% |
| E3_lr1e-4 | 97.0% | 65.5% | 98.5% | 96.0% | 99.5% | 100.0% | 100.0% | 87.0% |
| E3_lr5e-4 | 97.0% | 93.0% | 100.0% | 99.5% | 100.0% | 100.0% | 100.0% | 97.0% |
| E4_aug | 97.5% | 92.5% | 99.5% | 99.5% | 100.0% | 99.5% | 100.0% | 97.0% |
| E5_aug_lr5e-4 | 98.5% | 97.0% | 99.5% | 98.0% | 100.0% | 100.0% | 100.0% | 98.0% |
