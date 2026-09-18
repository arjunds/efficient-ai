#!/usr/bin/env python3
"""
models.py — analytic parameter / HBM-byte / FLOP accounting for transformer
serving, used by the energy model.

RECONSTRUCTED 2026-09-17. The original lived outside the repo (the old cluster's
sbatch bind-mounted ``$HOME/models.py`` into the container) and the in-repo copy
was committed as a 0-byte placeholder, so it did not survive the move to the new
server. This file re-derives the same accounting from the interface its consumers
require and is pinned to the constants the old code recorded.

VALIDATION ANCHORS (both reproduced exactly by the formulas below):
  * ``energy_profile_load.py:430``  WEIGHT_BYTES = 8.03e9 * 2  (Llama3-8B fp16)
    -> ``MODELS["Llama3-8B"].active_params()`` == 8_029_995_008.
  * ``energy_profile_load.py:431``  KV_BYTES_PER_TOKEN = 2*32*8*128*2 = 131072
    -> ``MODELS["Llama3-8B"].kv_bytes_per_token(2)`` == 131072.
Run ``python3 models.py`` to re-check these plus every table entry's total
parameter count against its published size.

THE ACCOUNTING (matches SESSION_LOG.md:189 and energy_model.py's docstring)
  weight_bytes = active_params * dtype_bytes          (reloaded every forward pass)
  kv_bytes     = kv_tokens_resident * kv_bytes_per_token
  flops_iter   = 2*active_params*(prefill_tokens + decode_tokens)
                 + attn_flops_per_token(1) * kv_tokens_resident

``active_params()`` counts the full weight set touched by a forward pass,
*including* the embedding table and lm_head. That is deliberately what the H200
runs used (the 8.03e9 anchor includes both), and coefficient comparability with
those runs matters more than a more physically-precise embedding treatment --
an embedding lookup really only reads the rows it needs, but changing the
convention now would silently rescale e_wbyte relative to the H200 reference.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

DTYPE_BYTES = {"float16": 2, "bfloat16": 2, "fp8": 1, "float8": 1,
               "float32": 4, "auto": 2}


@dataclass
class Model:
    """A dense or MoE decoder-only transformer, described by the handful of
    shape parameters that determine its HBM traffic and FLOPs."""
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int                       # dense FFN hidden size (0 for pure-MoE)
    vocab: int
    tie_embeddings: bool = False
    ffn: str = "mlp"                # "mlp" (dense) | "moe"
    mlp_gates: int = 3              # SwiGLU: gate+up+down. 2 for plain GELU MLP.
    attn: str = "gqa"               # "gqa"/"mha" | "mla"

    # --- MoE only ---
    n_routed_experts: int = 0
    n_active_experts: int = 0
    d_ff_expert: int = 0
    n_shared_experts: int = 0
    d_ff_shared: int = 0
    first_k_dense: int = 0          # leading layers that stay dense (DeepSeek-style)

    # --- MLA only (DeepSeek); ignored otherwise ---
    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0

    notes: str = ""

    # ------------------------------------------------------------------ shapes
    @property
    def q_dim(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def _n_moe_layers(self) -> int:
        """Number of layers whose FFN is a mixture of experts."""
        if self.ffn != "moe":
            return 0
        return max(0, self.n_layers - self.first_k_dense)

    def _n_dense_ffn_layers(self) -> int:
        return self.n_layers - self._n_moe_layers()

    # ------------------------------------------------------------------ params
    def _mlp_params(self, d_ff: int) -> int:
        """Parameters in one FFN block of hidden size ``d_ff``."""
        return self.mlp_gates * self.d_model * d_ff

    def _attn_params(self) -> int:
        """Parameters in one attention block (q/k/v/o projections)."""
        if self.attn == "mla" and self.kv_lora_rank:
            # Down-project to the latent, up-project back for K and V, plus o.
            down = self.d_model * (self.kv_lora_rank + self.qk_rope_head_dim)
            up = self.kv_lora_rank * 2 * self.q_dim
            return self.d_model * self.q_dim + down + up + self.q_dim * self.d_model
        return (self.d_model * self.q_dim          # q_proj
                + 2 * self.d_model * self.kv_dim   # k_proj, v_proj
                + self.q_dim * self.d_model)       # o_proj

    def _embed_params(self) -> int:
        e = self.vocab * self.d_model
        return e if self.tie_embeddings else 2 * e

    def _router_params(self) -> int:
        return self.d_model * self.n_routed_experts if self.ffn == "moe" else 0

    def total_params(self) -> int:
        """Every parameter in the checkpoint."""
        p = self._embed_params()
        p += self.n_layers * self._attn_params()
        p += self._n_dense_ffn_layers() * self._mlp_params(self.d_ff)
        if self.ffn == "moe":
            per_moe = (self.n_routed_experts * self._mlp_params(self.d_ff_expert)
                       + self.n_shared_experts * self._mlp_params(self.d_ff_shared)
                       + self._router_params())
            p += self._n_moe_layers() * per_moe
        return int(p)

    def active_params(self) -> int:
        """Parameters read from HBM for a forward pass over a *single* token.

        Dense: the whole model. MoE: attention + embeddings + router + only the
        top-k routed experts (+ any shared expert). See
        ``energy_model.weight_params_for_tokens`` for the batched-MoE correction,
        which grows this toward ``total_params`` as a batch fans out over experts.
        """
        p = self._embed_params()
        p += self.n_layers * self._attn_params()
        p += self._n_dense_ffn_layers() * self._mlp_params(self.d_ff)
        if self.ffn == "moe":
            per_moe = (self.n_active_experts * self._mlp_params(self.d_ff_expert)
                       + self.n_shared_experts * self._mlp_params(self.d_ff_shared)
                       + self._router_params())
            p += self._n_moe_layers() * per_moe
        return int(p)

    # ------------------------------------------------------------------- bytes
    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """HBM bytes of KV cache held per resident context token.

        Anchor: Llama3-8B fp16 -> 2*32*8*128*2 = 131072 B/token.
        """
        if self.attn == "mla" and self.kv_lora_rank:
            # MLA caches the compressed latent (+ the RoPE key), not full K/V.
            return int(self.n_layers * (self.kv_lora_rank + self.qk_rope_head_dim)
                       * dtype_bytes)
        return int(2 * self.n_layers * self.n_kv_heads * self.head_dim * dtype_bytes)

    def kv_cache_bytes(self, ctx: int, dtype_bytes: int = 2) -> int:
        """KV bytes for a sequence of ``ctx`` tokens."""
        return int(ctx * self.kv_bytes_per_token(dtype_bytes))

    def weight_bytes(self, dtype_bytes: int = 2) -> int:
        return int(self.active_params() * dtype_bytes)

    # ------------------------------------------------------------------- flops
    def attn_flops_per_token(self, ctx_tokens: int) -> int:
        """Attention FLOPs for ONE query token attending over ``ctx_tokens``.

        Per layer: QK^T is 2*n_heads*head_dim*T and A@V is another
        2*n_heads*head_dim*T. Linear in T, which is what lets the analysis use
        ``attn_flops_per_token(1) * sum(ctx)`` -- exact for non-chunked GQA.
        """
        return int(4 * self.n_layers * self.q_dim * ctx_tokens)

    def matmul_flops_per_token(self) -> int:
        """Dense-GEMM FLOPs per token (MAC = 2 flops)."""
        return int(2 * self.active_params())

    def flops_per_iteration(self, new_tokens: int, kv_tokens_resident: int) -> int:
        return int(self.matmul_flops_per_token() * new_tokens
                   + self.attn_flops_per_token(1) * kv_tokens_resident)

    # -------------------------------------------------------------------- misc
    def summary(self) -> str:
        return (f"{self.name}: L={self.n_layers} d={self.d_model} "
                f"H={self.n_heads}/{self.n_kv_heads} hd={self.head_dim} "
                f"ff={self.d_ff} V={self.vocab} ffn={self.ffn} "
                f"active={self.active_params()/1e9:.2f}B "
                f"total={self.total_params()/1e9:.2f}B "
                f"kv/tok(fp16)={self.kv_bytes_per_token(2)}B")


# --------------------------------------------------------------------------
# Hand-specified table. Keys match gate_dram.HF_TO_KEY.
# --------------------------------------------------------------------------
MODELS = {
    "Llama3-8B": Model(
        name="meta-llama/Meta-Llama-3-8B", n_layers=32, d_model=4096,
        n_heads=32, n_kv_heads=8, head_dim=128, d_ff=14336, vocab=128256,
        tie_embeddings=False,
        notes="anchor: active_params == 8,029,995,008, kv/tok fp16 == 131072"),

    "Llama3-70B": Model(
        name="meta-llama/Meta-Llama-3-70B", n_layers=80, d_model=8192,
        n_heads=64, n_kv_heads=8, head_dim=128, d_ff=28672, vocab=128256,
        tie_embeddings=False),

    "Qwen2-7B-Instruct": Model(
        name="Qwen/Qwen2-7B-Instruct", n_layers=28, d_model=3584,
        n_heads=28, n_kv_heads=4, head_dim=128, d_ff=18944, vocab=152064,
        tie_embeddings=False,
        notes="qkv bias omitted (~0.001% of params)"),

    "Mistral-7B-v0.1": Model(
        name="mistralai/Mistral-7B-v0.1", n_layers=32, d_model=4096,
        n_heads=32, n_kv_heads=8, head_dim=128, d_ff=14336, vocab=32000,
        tie_embeddings=False),

    "gemma-7b": Model(
        name="google/gemma-7b", n_layers=28, d_model=3072,
        n_heads=16, n_kv_heads=16, head_dim=256, d_ff=24576, vocab=256000,
        tie_embeddings=True,
        notes="head_dim*n_heads (4096) != d_model (3072); MHA not GQA"),

    "Qwen3-4B": Model(
        name="Qwen/Qwen3-4B", n_layers=36, d_model=2560,
        n_heads=32, n_kv_heads=8, head_dim=128, d_ff=9728, vocab=151936,
        tie_embeddings=True),

    "Qwen3-30B": Model(
        name="Qwen/Qwen3-30B-A3B", n_layers=48, d_model=2048,
        n_heads=32, n_kv_heads=4, head_dim=128, d_ff=0, vocab=151936,
        tie_embeddings=False, ffn="moe",
        n_routed_experts=128, n_active_experts=8, d_ff_expert=768,
        notes="A3B: ~3.35B active of ~30.5B total; untied embeddings "
              "(unlike the smaller Qwen3 models)"),
}

# Published sizes used by the self-check (billions of params).
_EXPECTED_TOTAL_B = {
    "Llama3-8B": 8.03, "Llama3-70B": 70.55, "Qwen2-7B-Instruct": 7.62,
    "Mistral-7B-v0.1": 7.24, "gemma-7b": 8.54, "Qwen3-4B": 4.02,
    "Qwen3-30B": 30.5,
}
_EXPECTED_ACTIVE_B = {"Qwen3-30B": 3.3}


# --------------------------------------------------------------------------
# Auto-ingest a dense model straight from its HF config.json.
# --------------------------------------------------------------------------
_CONFIG_CACHE: dict = {}


def _fetch_hf_config(hf_id: str) -> dict:
    """Load a model's config.json: local path, then transformers, then
    huggingface_hub, then a plain HTTPS GET. Deliberately dependency-light so it
    works inside the vLLM container without pulling in ``datasets`` (which breaks
    vLLM -- see HANDOFF_CROSSGPU.md gotcha 3)."""
    if hf_id in _CONFIG_CACHE:
        return _CONFIG_CACHE[hf_id]

    cfg = None
    if os.path.isdir(hf_id):
        with open(os.path.join(hf_id, "config.json")) as f:
            cfg = json.load(f)

    if cfg is None:
        try:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=False).to_dict()
        except Exception:
            pass

    if cfg is None:
        try:
            from huggingface_hub import hf_hub_download
            with open(hf_hub_download(hf_id, "config.json")) as f:
                cfg = json.load(f)
        except Exception:
            pass

    if cfg is None:
        url = f"https://huggingface.co/{hf_id}/resolve/main/config.json"
        req = urllib.request.Request(url, headers={"User-Agent": "models.py"})
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if tok:
            req.add_header("Authorization", f"Bearer {tok}")
        with urllib.request.urlopen(req, timeout=30) as r:
            cfg = json.loads(r.read().decode())

    # Some configs nest the real thing (e.g. multimodal wrappers).
    if "text_config" in cfg and "num_hidden_layers" not in cfg:
        cfg = cfg["text_config"]
    _CONFIG_CACHE[hf_id] = cfg
    return cfg


def model_from_hf_id(hf_id: str) -> Model:
    """Build a Model from a HuggingFace id by reading its config.json.

    Handles dense GQA/SwiGLU decoders (the common case) and the standard MoE
    config keys. Exotic architectures (MLA, hybrid SSM) still need a hand entry
    in MODELS -- validate any new model with ``.summary()`` before trusting its
    coefficients (HANDOFF_CROSSGPU.md gotcha 6).
    """
    c = _fetch_hf_config(hf_id)

    def g(*keys, default=None):
        for k in keys:
            if c.get(k) is not None:
                return c[k]
        return default

    n_layers = g("num_hidden_layers", "n_layer", "num_layers")
    d_model = g("hidden_size", "n_embd", "d_model")
    n_heads = g("num_attention_heads", "n_head")
    n_kv_heads = g("num_key_value_heads", "num_kv_heads", default=n_heads)
    head_dim = g("head_dim", default=None) or (d_model // n_heads)
    d_ff = g("intermediate_size", "ffn_dim", "n_inner", default=4 * d_model)
    vocab = g("vocab_size")
    tied = bool(g("tie_word_embeddings", default=False))

    if n_layers is None or d_model is None or n_heads is None or vocab is None:
        raise ValueError(f"config.json for {hf_id} is missing core shape keys")

    hidden_act = str(g("hidden_act", "activation_function", default="silu")).lower()
    mlp_gates = 2 if ("gelu" in hidden_act and "glu" not in hidden_act
                      and "geglu" not in hidden_act) else 3
    # Llama/Qwen/Mistral/Gemma are all gated (SwiGLU/GeGLU) -> 3 matrices.
    if any(t in str(g("model_type", default="")).lower()
           for t in ("llama", "qwen", "mistral", "gemma", "phi3")):
        mlp_gates = 3

    n_routed = g("num_experts", "n_routed_experts", "num_local_experts", default=0) or 0
    n_active = g("num_experts_per_tok", "n_active_experts",
                 "num_experts_per_token", default=0) or 0
    d_ff_expert = g("moe_intermediate_size", "expert_intermediate_size",
                    default=0) or 0
    n_shared = g("n_shared_experts", "num_shared_experts", default=0) or 0
    d_ff_shared = g("shared_expert_intermediate_size", default=0) or 0
    first_k_dense = g("first_k_dense_replace", default=0) or 0

    is_moe = bool(n_routed and n_active)
    if is_moe and not d_ff_expert:
        d_ff_expert = d_ff       # some configs reuse intermediate_size per expert

    return Model(
        name=hf_id, n_layers=int(n_layers), d_model=int(d_model),
        n_heads=int(n_heads), n_kv_heads=int(n_kv_heads), head_dim=int(head_dim),
        d_ff=0 if (is_moe and not first_k_dense) else int(d_ff),
        vocab=int(vocab), tie_embeddings=tied,
        ffn="moe" if is_moe else "mlp", mlp_gates=mlp_gates,
        n_routed_experts=int(n_routed), n_active_experts=int(n_active),
        d_ff_expert=int(d_ff_expert), n_shared_experts=int(n_shared),
        d_ff_shared=int(d_ff_shared or d_ff_expert),
        first_k_dense=int(first_k_dense),
        notes=f"auto-ingested from {hf_id} config.json")


# --------------------------------------------------------------------------
def _self_check() -> int:
    """Verify the hard anchors and every table entry. Exit non-zero on failure."""
    fails = []

    llama = MODELS["Llama3-8B"]
    if llama.active_params() != 8_029_995_008:
        fails.append(f"Llama3-8B active_params={llama.active_params():,} "
                     f"!= 8,029,995,008 (energy_profile_load.py:430)")
    if llama.kv_bytes_per_token(2) != 131072:
        fails.append(f"Llama3-8B kv/tok={llama.kv_bytes_per_token(2)} != 131072 "
                     f"(energy_profile_load.py:431)")

    print(f"{'model':<20}{'active B':>10}{'total B':>10}{'expect':>9}"
          f"{'err%':>7}{'kv/tok':>9}")
    for k, m in MODELS.items():
        tot = m.total_params() / 1e9
        act = m.active_params() / 1e9
        exp = _EXPECTED_TOTAL_B.get(k)
        err = 100 * (tot - exp) / exp if exp else 0.0
        print(f"{k:<20}{act:>10.2f}{tot:>10.2f}{exp if exp else 0:>9.2f}"
              f"{err:>7.2f}{m.kv_bytes_per_token(2):>9}")
        if exp and abs(err) > 2.0:
            fails.append(f"{k}: total {tot:.2f}B vs expected {exp}B ({err:+.1f}%)")
        exp_a = _EXPECTED_ACTIVE_B.get(k)
        if exp_a and abs(act - exp_a) / exp_a > 0.15:
            fails.append(f"{k}: active {act:.2f}B vs expected ~{exp_a}B")

    print()
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        return 1
    print("OK — anchors reproduced and all table entries within 2% of published size.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_check())
