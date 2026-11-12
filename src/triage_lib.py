"""Training + inference utilities for the NimbusPay triage take-home.

Plain PyTorch loop on top of HF transformers + PEFT. Written as one small module so every
choice (which tokens get loss, batching, loss normalisation, precision) is visible and testable.
"""
import gc
import json
import math
import os
import random
import subprocess
import time

import torch
import torch.nn.functional as F

BASE_DTYPE = torch.float16  # overridden to float32 only in the CPU unit test

# --------------------------------------------------------------------------------------------
# Data / chat template
# --------------------------------------------------------------------------------------------

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def prompt_ids(tok, row):
    """Exactly what the graders feed the model: system + user, model's own template, gen prompt."""
    msgs = [m for m in row["messages"] if m["role"] in ("system", "user")]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False)
    if not isinstance(ids, list):  # transformers 5 can still hand back a BatchEncoding
        ids = ids["input_ids"]
    return list(ids)


def encode_example(tok, row, end_token):
    """input_ids = prompt + label + <end>, labels = -100 on every prompt token (loss on answer only)."""
    p = prompt_ids(tok, row)
    target = row["messages"][-1]["content"] + end_token
    t = tok(target, add_special_tokens=False)["input_ids"]
    return {"input_ids": p + t, "labels": [-100] * len(p) + t, "n_prompt": len(p)}


def check_template(tok, row, end_token):
    """Assert our concatenation equals the model's own template rendering of the full conversation."""
    full = tok.apply_chat_template(row["messages"], tokenize=False)
    ours = tok.apply_chat_template(row["messages"][:2], add_generation_prompt=True, tokenize=False) \
        + row["messages"][-1]["content"] + end_token
    assert full.startswith(ours), f"template mismatch:\n{full!r}\nvs\n{ours!r}"
    tail = full[len(ours):]
    assert tail.strip() == "", f"unexpected text after end token: {tail!r}"
    return full


# --------------------------------------------------------------------------------------------
# Batching: length-bucketed micro-batches under a token budget, grouped into optimizer steps
# --------------------------------------------------------------------------------------------

def make_microbatches(lengths, max_tokens, max_bs, rng):
    """Sort by length (with jitter), cut into micro-batches whose padded size <= max_tokens."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i] + rng.random() * 8))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = lengths[i]
        new_max = max(cur_max, L)
        if cur and (new_max * (len(cur) + 1) > max_tokens or len(cur) >= max_bs):
            batches.append(cur)
            cur, new_max = [], L
        cur.append(i)
        cur_max = new_max
    if cur:
        batches.append(cur)
    rng.shuffle(batches)
    return batches


def group_steps(batches, examples_per_step):
    steps, cur, n = [], [], 0
    for b in batches:
        cur.append(b)
        n += len(b)
        if n >= examples_per_step:
            steps.append(cur)
            cur, n = [], 0
    if cur:
        steps.append(cur)
    return steps


def collate(feats, idx, pad_id, device):
    L = max(len(feats[i]["input_ids"]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    lab = torch.full((len(idx), L), -100, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):  # right padding for training
        x, y = feats[i]["input_ids"], feats[i]["labels"]
        ids[r, :len(x)] = torch.tensor(x)
        lab[r, :len(y)] = torch.tensor(y)
        att[r, :len(x)] = 1
    return ids.to(device), att.to(device), lab.to(device)


# --------------------------------------------------------------------------------------------
# Loss: logits only at supervised positions (saves ~GBs with Qwen's 152k vocab on long tickets)
# --------------------------------------------------------------------------------------------

def supervised_loss_sum(model, ids, att, lab):
    """Sum of token cross-entropy over answer tokens. Next-token shift: hidden[t] predicts token[t+1]."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    target = lab[:, 1:]
    mask = target != -100
    try:
        hidden = base.model(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
        h = hidden[:, :-1][mask]
        logits = base.lm_head(h).float()
    except AttributeError:  # architecture without .model/.lm_head: fall back to full logits
        logits = model(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1][mask].float()
    return F.cross_entropy(logits, target[mask], reduction="sum"), int(mask.sum())


# --------------------------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------------------------

def load_base(model_id, load_4bit, device_map="auto"):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    # T4 has no bf16 -> fp16 weights + fp16 autocast. transformers >= 4.56 renamed `torch_dtype` to `dtype`.
    major, minor = (int(x) for x in __import__("transformers").__version__.split(".")[:2])
    kw = {("dtype" if (major, minor) >= (4, 56) else "torch_dtype"): BASE_DTYPE}
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    if torch.cuda.is_available():
        kw["device_map"] = device_map
    return AutoModelForCausalLM.from_pretrained(model_id, **kw)


def add_lora(model, r, alpha, dropout, target_modules, load_4bit, grad_ckpt=True):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    if load_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=grad_ckpt,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    elif grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=target_modules,
                     bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    # PEFT keeps adapter weights in fp32 by default (autocast_adapter_dtype=True) -> safe with fp16 AMP.
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    return model


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------

def train(cfg, tok, train_rows, out_dir, log=print):
    """cfg keys: model_id, load_4bit, r, alpha, dropout, target_modules, lr, epochs, warmup_frac,
    examples_per_step, max_tokens, max_bs, max_len, seed, end_token."""
    from transformers import get_cosine_schedule_with_warmup
    seed = cfg["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    feats = [encode_example(tok, r, cfg["end_token"]) for r in train_rows]
    lengths = [len(f["input_ids"]) for f in feats]
    too_long = sum(L > cfg["max_len"] for L in lengths)
    assert too_long == 0, f"{too_long} examples exceed max_len={cfg['max_len']}; would truncate the label"

    model = load_base(cfg["model_id"], cfg["load_4bit"])
    model = add_lora(model, cfg["r"], cfg["alpha"], cfg["dropout"], cfg["target_modules"], cfg["load_4bit"],
                     grad_ckpt=cfg.get("grad_ckpt", True))
    n_train = count_trainable(model)
    assert n_train <= 50_000_000, f"{n_train:,} trainable params > 50M cap"
    log(f"trainable params: {n_train:,}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0, betas=(0.9, 0.999))
    rng = random.Random(seed)
    steps_per_epoch = len(group_steps(make_microbatches(lengths, cfg["max_tokens"], cfg["max_bs"], random.Random(0)),
                                      cfg["examples_per_step"]))
    total = steps_per_epoch * cfg["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(cfg["warmup_frac"] * total)), total)
    use_amp = dev == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history, step = [], 0
    model.train()
    for ep in range(cfg["epochs"]):
        steps = group_steps(make_microbatches(lengths, cfg["max_tokens"], cfg["max_bs"], rng), cfg["examples_per_step"])
        for mbs in steps:
            n_tok = sum(sum(1 for y in feats[i]["labels"][1:] if y != -100) for b in mbs for i in b)
            tot = 0.0
            for b in mbs:
                ids, att, lab = collate(feats, b, tok.pad_token_id, dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    s, _ = supervised_loss_sum(model, ids, att, lab)
                loss = s / n_tok  # mean over ALL answer tokens in the optimizer step
                scaler.scale(loss).backward()
                tot += loss.item()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            history.append({"step": step, "epoch": ep + 1, "loss": tot, "lr": sched.get_last_lr()[0],
                            "grad_norm": float(gn)})
            if step % 10 == 0 or step == total:
                log(f"ep {ep + 1} step {step}/{total} loss {tot:.4f} lr {sched.get_last_lr()[0]:.2e} gn {float(gn):.2f}")
            if not math.isfinite(tot):
                raise RuntimeError("non-finite loss")
    minutes = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2**30 if dev == "cuda" else float("nan")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)  # adapter_config.json + adapter_model.safetensors
    json.dump({"cfg": cfg, "train_minutes": minutes, "peak_gpu_gib": peak, "trainable_params": n_train,
               "n_train_rows": len(train_rows), "history": history},
              open(os.path.join(out_dir, "train_log.json"), "w"), indent=1)
    log(f"done in {minutes:.1f} min, peak GPU {peak:.2f} GiB -> {out_dir}")
    del model, opt
    gc.collect()
    if dev == "cuda":
        torch.cuda.empty_cache()
    return {"train_minutes": minutes, "peak_gpu_gib": peak, "trainable_params": n_train}


# --------------------------------------------------------------------------------------------
# Inference (what the graders run: chat template, plain greedy generate, max_new_tokens=200)
# --------------------------------------------------------------------------------------------

def load_for_inference(model_id, adapter_dir, load_4bit):
    from peft import PeftModel
    base = load_base(model_id, load_4bit)
    m = PeftModel.from_pretrained(base, adapter_dir) if adapter_dir else base
    m.eval()
    return m


def end_ids(tok):
    """Stop tokens: chat end-of-turn (<|im_end|> for Qwen) plus the tokenizer eos, as in the hub generation_config."""
    ids = []
    for t in ("<|im_end|>", tok.eos_token):
        i = tok.convert_tokens_to_ids(t) if t else None
        if i is not None and i != tok.unk_token_id and i not in ids:
            ids.append(i)
    return ids


def greedy_config(tok, max_new_tokens=200):
    from transformers import GenerationConfig
    # Explicit config so nothing from the hub generation_config (sampling params, repetition_penalty)
    # leaks in: pure argmax decoding.
    return GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                            eos_token_id=end_ids(tok), pad_token_id=tok.pad_token_id)


@torch.no_grad()
def generate(model, tok, rows, batch_size=1, max_new_tokens=200, system_override=None, gen_cfg=None, log=print):
    """Return list of {"id","output"}; batch_size=1 means no padding at all (bit-for-bit reproducible path)."""
    dev = next(model.parameters()).device
    gen_cfg = gen_cfg or greedy_config(tok, max_new_tokens)
    old_side = tok.padding_side
    tok.padding_side = "left"
    out, t0 = [], time.time()
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["messages"][1]["content"]))
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        prompts = []
        for i in idx:
            r = rows[i]
            if system_override is not None:
                r = {"messages": [{"role": "system", "content": system_override}, r["messages"][1]]}
            prompts.append(prompt_ids(tok, r))
        L = max(map(len, prompts))
        ids = torch.full((len(prompts), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(prompts), L), dtype=torch.long)
        for k, p in enumerate(prompts):
            ids[k, L - len(p):] = torch.tensor(p)
            att[k, L - len(p):] = 1
        gen = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), generation_config=gen_cfg)
        for k, i in enumerate(idx):
            text = tok.decode(gen[k, L:], skip_special_tokens=True)
            out.append((i, {"id": rows[i]["id"], "output": text.strip()}))
        if (s // batch_size) % 20 == 0:
            log(f"  generated {min(s + batch_size, len(order))}/{len(order)} ({time.time() - t0:.0f}s)")
    tok.padding_side = old_side
    out.sort()
    return [o for _, o in out]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def score(score_py, gold_path, pred_path):
    res = subprocess.run(["python", score_py, "--gold", gold_path, "--pred", pred_path, "--json"],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)
