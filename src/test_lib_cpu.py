"""CPU smoke test of triage_lib with a tiny random Qwen2 model + a tokenizer trained on the tickets.
Checks: template assertion, loss masking, batching, training loop, saving PEFT adapter, reload, greedy generate."""
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import triage_lib as T
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

T.BASE_DTYPE = torch.float32
rows = T.load_jsonl(sys.argv[1])[:64]  # path to train_clean.jsonl
dev = T.load_jsonl(sys.argv[2])[:3]  # path to dev.jsonl
tmp = tempfile.mkdtemp()

# --- tokenizer with Qwen-style chat template
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
bpe = Tokenizer(models.BPE(unk_token=None))
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
corpus = [m["content"] for r in rows for m in r["messages"]]
bpe.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=1200, special_tokens=specials,
                                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|im_end|>", pad_token="<|endoftext|>")
tok.chat_template = ("{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + "
                     "'<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}")
END = "<|im_end|>"

# --- tiny model
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                  tie_word_embeddings=True)
mdir = os.path.join(tmp, "tiny")
Qwen2ForCausalLM(cfg).save_pretrained(mdir)
tok.save_pretrained(mdir)

# --- template + masking checks
for r in rows[:20]:
    T.check_template(tok, r, END)
e = T.encode_example(tok, rows[0], END)
sup = [t for t, l in zip(e["input_ids"], e["labels"]) if l != -100]
assert tok.decode(sup) == rows[0]["messages"][-1]["content"] + END, tok.decode(sup)
assert e["input_ids"][:e["n_prompt"]] == T.prompt_ids(tok, rows[0])
print("template + masking OK; supervised tokens:", len(sup), "of", len(e["input_ids"]))

# --- batching invariants
import random
lens = [len(T.encode_example(tok, r, END)["input_ids"]) for r in rows]
mbs = T.make_microbatches(lens, 2048, 16, random.Random(0))
assert sorted(i for b in mbs for i in b) == list(range(len(rows)))
assert all(max(lens[i] for i in b) * len(b) <= 2048 or len(b) == 1 for b in mbs)
steps = T.group_steps(mbs, 16)
print("batching OK:", len(mbs), "micro-batches,", len(steps), "steps")

# --- loss equals HF's own masked loss (mean over answer tokens)
model = T.add_lora(T.load_base(mdir, False), 4, 8, 0.0, ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                   False, grad_ckpt=False)
ids, att, lab = T.collate([T.encode_example(tok, r, END) for r in rows[:4]], [0, 1, 2, 3], tok.pad_token_id, "cpu")
s, n = T.supervised_loss_sum(model, ids, att, lab)
ref = model(input_ids=ids, attention_mask=att, labels=lab).loss
print(f"our loss {s.item() / n:.6f} vs HF loss {ref.item():.6f}")
assert abs(s.item() / n - ref.item()) < 1e-4
del model

# --- full train -> save -> reload -> generate
cfgd = dict(model_id=mdir, load_4bit=False, r=8, alpha=16, dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lr=5e-3, epochs=3, warmup_frac=0.1, examples_per_step=16, max_tokens=4096, max_bs=16,
            max_len=4096, seed=0, end_token=END, grad_ckpt=True)
out = os.path.join(tmp, "adapter")
res = T.train(cfgd, tok, rows, out)
log = json.load(open(os.path.join(out, "train_log.json")))
print("losses:", [round(h["loss"], 3) for h in log["history"]])
assert log["history"][-1]["loss"] < log["history"][0]["loss"], "loss did not go down"
assert set(os.listdir(out)) >= {"adapter_config.json", "adapter_model.safetensors"}
m = T.load_for_inference(mdir, out, False)
p1 = T.generate(m, tok, dev, batch_size=1, max_new_tokens=20)
p2 = T.generate(m, tok, dev, batch_size=3, max_new_tokens=20)
print("gen bs1:", p1[0]); print("gen bs3 same as bs1:", [a == b for a, b in zip(p1, p2)])
assert [p["id"] for p in p1] == [r["id"] for r in dev]
print("ALL OK")
shutil.rmtree(tmp)
