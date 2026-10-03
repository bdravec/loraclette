"""Bottleneck adapters for Apertus, without PEFT or the official fine-tuning recipes.

Usage:
  python apertus_adapters.py check --model swiss-ai/Apertus-8B-Instruct-2509
  python apertus_adapters.py train --data train.jsonl --positions mlp attention --out adapters.pt

Training data: one JSON object per line, in chat format:
  {"messages": [{"role": "user", "content": "<file with conflict markers>"},
                {"role": "assistant", "content": "<resolved file>"}]}

Adapter positions (numbers match the figure in the README):
  mlp            1  parallel to the feed-forward MLP
  attention      2  parallel to self-attention
  embed          3  after the token embeddings, before the decoder stack
  final          4  after the decoder stack, before the final RMSNorm
  residual       5  on the residual stream, after the attention sum
  pre_attention  6  between RMSNorm and self-attention
"""
import argparse
import json

import torch
import torch.nn as nn

POSITIONS = ("mlp", "attention", "embed", "final", "residual", "pre_attention")


# ----------------------------------------------------------------- modules
class BottleneckAdapter(nn.Module):
    """Down-project (d -> r), nonlinearity, up-project (r -> d). Returns the delta only."""

    def __init__(self, d, r=64, scale=1.0):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(r, d, bias=False)
        self.scale = scale
        nn.init.zeros_(self.up.weight)  # starts as a no-op: model output is unchanged

    def forward(self, x):
        return self.scale * self.up(self.act(self.down(x)))


class MLPWrap(nn.Module):
    def __init__(self, mlp, adapter):
        super().__init__()
        self.mlp, self.adapter = mlp, adapter

    def forward(self, x):
        return self.mlp(x) + self.adapter(x)


class NormWrap(nn.Module):
    """Wraps attention_layernorm: remembers the residual stream and hosts the pre_attention adapter."""

    def __init__(self, norm, adapter=None):
        super().__init__()
        self.norm, self.adapter = norm, adapter
        self.stream = None

    def forward(self, x):
        self.stream = x
        h = self.norm(x)
        return h if self.adapter is None else h + self.adapter(h)


class AttentionWrap(nn.Module):
    """Hosts the attention adapter and the residual-stream adapter."""

    def __init__(self, attn, norm_wrap, adapter=None, stream_adapter=None):
        super().__init__()
        self.attn, self.adapter, self.stream_adapter = attn, adapter, stream_adapter
        object.__setattr__(self, "_norm_wrap", norm_wrap)  # reference only, not a submodule

    def forward(self, hidden_states, **kwargs):
        out, weights = self.attn(hidden_states=hidden_states, **kwargs)
        if self.adapter is not None:
            out = out + self.adapter(hidden_states)
        if self.stream_adapter is not None:  # layer then computes stream + out = s + A(s)
            out = out + self.stream_adapter(self._norm_wrap.stream + out)
        return out, weights


# ------------------------------------------------------------ attach/detach
def check_architecture(model):
    """Verify the model has the structure the adapters attach to. Returns a list of problems."""
    core = getattr(model, "model", None)
    layers = getattr(core, "layers", None)
    if layers is None:
        return ["model.model.layers not found (not an Apertus-style decoder stack)"]
    problems = [f"model.model.{n} not found" for n in ("embed_tokens", "norm") if not hasattr(core, n)]
    if hasattr(core, "embed_adapter") or hasattr(core, "final_adapter"):
        problems.append("adapters are already attached")
    for i, layer in enumerate(layers):
        for name in ("self_attn", "mlp", "attention_layernorm", "feedforward_layernorm"):
            if not hasattr(layer, name):
                problems.append(f"layer {i}: missing '{name}'")
        if isinstance(getattr(layer, "mlp", None), MLPWrap) or isinstance(getattr(layer, "self_attn", None), AttentionWrap):
            problems.append(f"layer {i}: adapters are already attached")
    return problems


def add_adapters(model, r=64, scale=1.0, positions=("mlp",), seed=0):
    """Freeze the base model and attach adapters at the chosen positions."""
    unknown = set(positions) - set(POSITIONS)
    if unknown:
        raise ValueError(f"unknown positions {sorted(unknown)}; choose from {POSITIONS}")
    problems = check_architecture(model)
    if problems:
        raise RuntimeError("Cannot attach adapters:\n  " + "\n  ".join(problems))
    torch.manual_seed(seed)
    for p in model.parameters():
        p.requires_grad = False
    d, core = model.config.hidden_size, model.model
    make = lambda ref: BottleneckAdapter(d, r, scale).to(device=ref.device, dtype=ref.dtype)
    hooks = []
    for layer in core.layers:
        ref = layer.feedforward_layernorm.weight  # norm weights are never quantised
        if "mlp" in positions:
            layer.mlp = MLPWrap(layer.mlp, make(ref))
        if {"attention", "residual", "pre_attention"} & set(positions):
            norm = NormWrap(layer.attention_layernorm, make(ref) if "pre_attention" in positions else None)
            layer.attention_layernorm = norm
            layer.self_attn = AttentionWrap(layer.self_attn, norm,
                                            make(ref) if "attention" in positions else None,
                                            make(ref) if "residual" in positions else None)
    if "embed" in positions:
        core.embed_adapter = make(core.layers[0].feedforward_layernorm.weight)
        hooks.append(core.embed_tokens.register_forward_hook(
            lambda mod, args, out: out + core.embed_adapter(out.to(core.embed_adapter.down.weight.dtype)).to(out.dtype)))
    if "final" in positions:
        core.final_adapter = make(core.norm.weight)
        hooks.append(core.norm.register_forward_pre_hook(
            lambda mod, args: (args[0] + core.final_adapter(args[0]),) + tuple(args[1:])))
    model._adapter_hooks = hooks
    return model


def remove_adapters(model):
    """Restore the original modules (base weights stay frozen)."""
    core = model.model
    for h in getattr(model, "_adapter_hooks", []):
        h.remove()
    model._adapter_hooks = []
    for name in ("embed_adapter", "final_adapter"):
        if hasattr(core, name):
            delattr(core, name)
    for layer in core.layers:
        if isinstance(layer.mlp, MLPWrap):
            layer.mlp = layer.mlp.mlp
        if isinstance(layer.self_attn, AttentionWrap):
            layer.self_attn = layer.self_attn.attn
        if isinstance(layer.attention_layernorm, NormWrap):
            layer.attention_layernorm = layer.attention_layernorm.norm
    return model


def adapter_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "adapter" in k}


def save_adapters(model, path, **meta):
    torch.save({"state": adapter_state_dict(model), "meta": meta}, path)


def load_adapters(model, path):
    """Call on a fresh base model: re-attaches adapters and loads their weights."""
    ckpt = torch.load(path, map_location="cpu")
    add_adapters(model, **ckpt["meta"])
    _, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    assert not unexpected, unexpected
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, train


def report(model):
    total, train = count_params(model)
    print(f"parameters: {total:,} total, {train:,} trainable ({100 * train / total:.3f}%)")


# -------------------------------------------------------------- train/eval
def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def encode(tok, row, max_len):
    enc = tok.apply_chat_template(row["messages"], tokenize=True, return_dict=True, return_tensors="pt")
    return enc["input_ids"][:, :max_len]


def train_adapters(model, tok, rows, lr=1e-4, epochs=1, accum=8, max_len=8192, log=True):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    model.train()
    step = 0
    for epoch in range(epochs):
        for row in rows:
            ids = encode(tok, row, max_len).to(model.device)
            loss = model(input_ids=ids, labels=ids).loss / accum
            loss.backward()
            step += 1
            if step % accum == 0 or step == len(rows) * epochs:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad()
                if log:
                    print(f"epoch {epoch} step {step} loss {loss.item() * accum:.4f}")


@torch.no_grad()
def evaluate(model, tok, rows, max_len=8192):
    """Mean validation loss (lower is better)."""
    model.eval()
    losses = []
    for row in rows:
        ids = encode(tok, row, max_len).to(model.device)
        losses.append(model(input_ids=ids, labels=ids).loss.item())
    return sum(losses) / len(losses)


def load_model(name, four_bit=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    if four_bit:  # needs: pip install bitsandbytes (CUDA only)
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    return AutoTokenizer.from_pretrained(name), AutoModelForCausalLM.from_pretrained(name, **kwargs)


# -------------------------------------------------------------------- CLI
def cmd_check(args):
    tok, model = load_model(args.model, args.four_bit)
    print(model.model.layers[0])  # shows the block structure and module names
    problems = check_architecture(model)
    print("OK: adapters can be attached" if not problems else "\n".join(problems))
    ids = torch.randint(0, model.config.vocab_size, (1, 16), device=model.device)
    with torch.no_grad():
        before = model(ids).logits
        add_adapters(model, r=args.rank, positions=args.positions)
        after = model(ids).logits
    print("output unchanged right after attaching:", torch.allclose(before, after, atol=1e-3))
    report(model)


def cmd_train(args):
    tok, model = load_model(args.model, args.four_bit)
    add_adapters(model, r=args.rank, positions=args.positions)
    report(model)
    train_adapters(model, tok, read_jsonl(args.data), args.lr, args.epochs, args.accum, args.max_len)
    save_adapters(model, args.out, r=args.rank, positions=tuple(args.positions))
    print("saved", args.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("train", cmd_train)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--model", default="swiss-ai/Apertus-8B-Instruct-2509")
        p.add_argument("--rank", type=int, default=64)
        p.add_argument("--positions", nargs="+", default=["mlp"], choices=POSITIONS)
        p.add_argument("--four-bit", action="store_true")
        if name == "train":
            p.add_argument("--data", required=True)
            p.add_argument("--out", default="adapters.pt")
            p.add_argument("--lr", type=float, default=1e-4)
            p.add_argument("--epochs", type=int, default=1)
            p.add_argument("--accum", type=int, default=8)
            p.add_argument("--max-len", type=int, default=8192)
    args = ap.parse_args()
    args.fn(args)
