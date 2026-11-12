# NimbusPay ticket triage: LoRA fine-tune (VIMA3YA ML Intern take-home)

| | |
|---|---|
| **Base model** | `Qwen/Qwen2.5-1.5B-Instruct` |
| **Loaded in 4-bit?** | **No.** The base is loaded in fp16 (`dtype=torch.float16`); the adapter was trained with 16-bit LoRA. |
| **Adapter** | `adapter/` (PEFT `adapter_config.json` + `adapter_model.safetensors`): LoRA r=16, α=32, all linear layers, 18.46M params |
| **Best result** | **93.5% exact match / 98.9% mean field accuracy on dev** (`score.py`), run `E5_aug_lr5e-4`: cleaned train + 700 spec-driven synthetic tickets, lr 5e-4, 2 epochs |

## Reproduce
Open `notebook.ipynb` in Colab (T4) and run all. The data cell asks you to upload `train.jsonl`, `dev.jsonl`,
`test_inputs.jsonl`, `score.py` and `SCHEMA.md`. The default path (`RUN_BASELINES = RUN_EXPERIMENTS = False`) cleans the data,
builds the augmentation, retrains the final adapter, scores dev, writes `predictions_test.jsonl` and checks reproducibility.
On a T4 it took about 75 min: 15 min training, 17 min dev and 33 min test decoding at batch size 1. The outputs saved in the
notebook come from a run with both flags `True`; per-run results are in `logs/` and `experiments.md`.

## Inference, exactly as used for `predictions_test.jsonl`
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", dtype=torch.float16, device_map="cuda")
model = PeftModel.from_pretrained(base, "adapter").eval()
gen = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=200, repetition_penalty=1.0,
                       eos_token_id=[151645, 151643], pad_token_id=151643)   # <|im_end|>, <|endoftext|>

def predict(row):  # row = one line of test_inputs.jsonl
    enc = tok.apply_chat_template(row["messages"][:2], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    out = model.generate(**enc, generation_config=gen)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
```
Tickets are decoded one at a time, with no padding. Re-loading the adapter this way reproduced 25/25 random test predictions
exactly. The hub `generation_config` sets `repetition_penalty=1.1`; with that default instead, outputs were identical on 49/50
dev tickets. Versions used: torch 2.14.1, transformers 5.22.1, peft 0.21.2.

## Repository
| Path | What |
|---|---|
| `notebook.ipynb` | Colab notebook with outputs (Parts A–E) |
| `report.md` | audit findings, experiments, error analysis, next steps, AI-tool use |
| `cleaning_log.csv` | every dropped or fixed train row: `id,action,reason` |
| `predictions_test.jsonl` | raw model output for each of the 400 test rows |
| `predictions_dev.jsonl` | the same for dev (93.5% exact) |
| `src/speccheck.py` | SCHEMA.md rules in code, used **only** to audit and build training data, never at inference |
| `src/audit.py`, `src/clean.py`, `src/aug.py` | audit, cleaning, spec-driven augmentation |
| `src/triage_lib.py` | training loop (answer-only loss, length-bucketed batches) and greedy inference; `src/test_lib_cpu.py` tests it |
| `experiments.md`, `logs/` | per-run results, training curves, baseline outputs |

Training data, the base model and the candidate pack are not in this repo.
