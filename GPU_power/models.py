"""
Model architecture database for LIMINAL replication.

Each model is described by enough architectural detail to compute:
  - total parameter count and per-token *active* parameter count
  - KV-cache bytes per token (GQA or MLA)
  - FLOPs per decode token (matmul + attention)

Sources: public model cards / config.json for each family. Where a value is
uncertain it is flagged in NOTES at the bottom of this file.

All parameter helpers return counts (not bytes). LIMINAL's xPU is an FP8 part,
so capacity/AMI in the paper appear to use 1 byte/param (see analysis.py).
"""

from dataclasses import dataclass, field

# Count MLA attention weights in their absorbed/materialized decode form.
MLA_ABSORBED = True
# Account for Llama4 chunked local attention (reduces long-context KV cache).
# Default OFF: the paper's Table 4 (capacity/AMI) and its Maverick 128K/TP8
# throughput are all self-consistent ONLY with full KV. Enabling this recovers
# the paper's anomalous Maverick 128K/TP128 STPS (114K) but then contradicts
# those other cells -- i.e. the paper itself is internally inconsistent there.
LLAMA4_CHUNKED = False


@dataclass
class Model:
    name: str
    L: int                      # number of transformer layers
    d: int                      # hidden size
    vocab: int
    # ---- attention ----
    attn: str                   # "gqa" or "mla"
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    # MLA-specific
    q_lora_rank: int = 0        # 0 => no q down-projection
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    # ---- feed-forward ----
    ffn: str = "mlp"            # "mlp" or "moe"
    d_ff: int = 0               # dense MLP intermediate size
    # MoE
    n_routed_experts: int = 0
    n_active_experts: int = 0   # top_k
    n_shared_experts: int = 0
    d_ff_expert: int = 0        # per-expert intermediate size
    first_k_dense: int = 0      # first k layers are dense MLP (DeepSeek/Kimi)
    d_ff_dense: int = 0         # intermediate for those dense layers
    interleave_step: int = 1    # 1 => every layer MoE; 2 => every 2nd layer MoE (Llama4 Maverick)
    attn_chunk: int = 0         # chunked local-attention window (0 => full attention every layer)
    global_attn_ratio: float = 1.0   # fraction of layers using full/global attention
    gated: bool = True          # SwiGLU (3 matrices) vs plain MLP (2)
    tied_embeddings: bool = False

    # ---------- attention parameter count (per layer) ----------
    def attn_params(self):
        d = self.d
        if self.attn == "gqa":
            hd = self.head_dim
            q = d * self.n_heads * hd
            k = d * self.n_kv_heads * hd
            v = d * self.n_kv_heads * hd
            o = self.n_heads * hd * d
            return q + k + v + o
        elif self.attn == "mla":
            # LIMINAL loads the *absorbed* MLA decode weights: W_UK is folded
            # into the query up-projection and W_UV into the output projection,
            # so both operate in the compressed latent space (kv_lora_rank) and
            # are materially larger than the compact stored matrices. This is
            # what reproduces the paper's DeepSeek/Kimi param, capacity and
            # throughput numbers (see notes at bottom of file).
            if MLA_ABSORBED:
                w_dq = d * self.q_lora_rank if self.q_lora_rank else 0
                q_in = self.q_lora_rank if self.q_lora_rank else d
                w_uq_nope = q_in * self.n_heads * self.kv_lora_rank   # absorbed with W_UK
                w_uq_rope = q_in * self.n_heads * self.qk_rope_head_dim
                w_dkv = d * (self.kv_lora_rank + self.qk_rope_head_dim)
                w_o = self.n_heads * self.kv_lora_rank * d            # absorbed with W_UV
                return w_dq + w_uq_nope + w_uq_rope + w_dkv + w_o
            # compact (as-stored) MLA weights
            qk = self.qk_nope_head_dim + self.qk_rope_head_dim
            if self.q_lora_rank:
                q = d * self.q_lora_rank + self.q_lora_rank * self.n_heads * qk
            else:
                q = d * self.n_heads * qk
            kv_down = d * (self.kv_lora_rank + self.qk_rope_head_dim)
            kv_up = self.kv_lora_rank * self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)
            o = self.n_heads * self.v_head_dim * d
            return q + kv_down + kv_up + o
        raise ValueError(self.attn)

    # ---------- feed-forward parameter count for one layer ----------
    def _mlp_params(self, d_ff):
        mats = 3 if self.gated else 2
        return mats * self.d * d_ff

    def ffn_params_dense_layer(self):
        """params of a *dense* FFN layer (used for first_k_dense layers)."""
        return self._mlp_params(self.d_ff_dense or self.d_ff)

    def ffn_params_moe_layer_total(self):
        """total params of one MoE layer (all experts resident)."""
        experts = (self.n_routed_experts + self.n_shared_experts) * self._mlp_params(self.d_ff_expert)
        router = self.d * self.n_routed_experts
        return experts + router

    def ffn_params_moe_layer_active(self):
        """active params of one MoE layer for a single token (top_k + shared)."""
        experts = (self.n_active_experts + self.n_shared_experts) * self._mlp_params(self.d_ff_expert)
        router = self.d * self.n_routed_experts
        return experts + router

    # ---------- whole-model counts ----------
    def _n_moe_layers(self):
        if self.ffn == "mlp":
            return 0
        eligible = self.L - self.first_k_dense
        return eligible // self.interleave_step

    def _n_dense_layers(self):
        if self.ffn == "mlp":
            return self.L
        return self.L - self._n_moe_layers()

    def embedding_params(self):
        # input embedding + output head (untied unless tied)
        return self.vocab * self.d * (1 if self.tied_embeddings else 2)

    def total_params(self):
        p = self.L * self.attn_params()
        if self.ffn == "mlp":
            p += self.L * self._mlp_params(self.d_ff)
        else:
            p += self._n_dense_layers() * self.ffn_params_dense_layer()
            p += self._n_moe_layers() * self.ffn_params_moe_layer_total()
        p += self.embedding_params()
        return p

    def active_params(self):
        """params touched for one token (top_k routing). Includes lm_head."""
        p = self.L * self.attn_params()
        if self.ffn == "mlp":
            p += self.L * self._mlp_params(self.d_ff)
        else:
            p += self._n_dense_layers() * self.ffn_params_dense_layer()
            p += self._n_moe_layers() * self.ffn_params_moe_layer_active()
        p += self.vocab * self.d  # lm_head participates per token; input embed is a lookup
        return p

    # ---------- KV cache bytes per token (1 byte/elem, FP8) ----------
    def kv_bytes_per_token(self, bytes_per_elem=1):
        if self.attn == "gqa":
            return 2 * self.L * self.n_kv_heads * self.head_dim * bytes_per_elem
        else:  # MLA stores only the compressed latent + rope key
            return self.L * (self.kv_lora_rank + self.qk_rope_head_dim) * bytes_per_elem

    def kv_cache_bytes(self, T, bytes_per_elem=1):
        """Total KV bytes for one user at context length T. With chunked local
        attention (Llama4) only global layers keep the full T; local layers cap
        at the chunk window."""
        if self.attn == "gqa":
            per_layer_tok = 2 * self.n_kv_heads * self.head_dim * bytes_per_elem
        else:
            per_layer_tok = (self.kv_lora_rank + self.qk_rope_head_dim) * bytes_per_elem
        if LLAMA4_CHUNKED and self.attn_chunk and T > self.attn_chunk:
            n_global = round(self.L * self.global_attn_ratio)
            n_local = self.L - n_global
            tokens = n_global * T + n_local * self.attn_chunk
            return per_layer_tok * tokens
        return per_layer_tok * self.L * T

    def _attn_span_layers(self, T):
        """Sum over layers of the number of context tokens each layer attends
        to (accounts for chunked local attention)."""
        if LLAMA4_CHUNKED and self.attn_chunk and T > self.attn_chunk:
            n_global = round(self.L * self.global_attn_ratio)
            return n_global * T + (self.L - n_global) * self.attn_chunk
        return self.L * T

    # ---------- attention FLOPs per token for context length T ----------
    def attn_flops_per_token(self, T):
        span = self._attn_span_layers(T)
        if self.attn == "gqa":
            # QK^T + softmax·V, 2 flops each, over attended keys, per head
            return 2 * (2 * self.n_heads * self.head_dim) * span
        else:
            # MLA *absorbed* decode path: queries are projected into the
            # compressed latent, so scores run over (kv_lora_rank + rope) and
            # the value aggregation runs over kv_lora_rank (per head, per layer).
            score = 2 * self.n_heads * (self.kv_lora_rank + self.qk_rope_head_dim)
            av = 2 * self.n_heads * self.kv_lora_rank
            return (score + av) * span

    @classmethod
    def from_hf(cls, name, cfg):
        """Build a dense GQA/SwiGLU Model from a HuggingFace config dict. Covers
        standard decoder-only transformers (Llama/Qwen2.x/Mistral/gemma-style
        dense models). MoE/MLA still use hand-specified entries below. This lets
        the size-ladder / new dense models be ingested automatically from their
        config.json with no hand entry (auditable + no transcription error)."""
        d = cfg["hidden_size"]
        n_heads = cfg["num_attention_heads"]
        head_dim = cfg.get("head_dim") or (d // n_heads)
        return cls(
            name=name, L=cfg["num_hidden_layers"], d=d, vocab=cfg["vocab_size"],
            attn="gqa", n_heads=n_heads,
            n_kv_heads=cfg.get("num_key_value_heads", n_heads), head_dim=head_dim,
            ffn="mlp", d_ff=cfg["intermediate_size"], gated=True,
            tied_embeddings=cfg.get("tie_word_embeddings", False),
        )


def model_from_hf_id(hf_id, name=None):
    """Resolve a HF model id to a Model by reading its config.json via
    transformers.AutoConfig (config is tiny; downloads/caches with the model)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(hf_id, trust_remote_code=True).to_dict()
    if cfg.get("num_experts") or cfg.get("n_routed_experts"):
        raise ValueError(f"{hf_id} looks like an MoE; add a hand-specified entry")
    return Model.from_hf(name or hf_id.split("/")[-1], cfg)


MODELS = {
    # ---------------- Llama 3 (dense GQA + SwiGLU) ----------------
    "Llama3-8B": Model("Llama3-8B", L=32, d=4096, vocab=128256, attn="gqa",
                       n_heads=32, n_kv_heads=8, head_dim=128, ffn="mlp", d_ff=14336),
    "Llama3-70B": Model("Llama3-70B", L=80, d=8192, vocab=128256, attn="gqa",
                        n_heads=64, n_kv_heads=8, head_dim=128, ffn="mlp", d_ff=28672),
    "Llama3-405B": Model("Llama3-405B", L=126, d=16384, vocab=128256, attn="gqa",
                         n_heads=128, n_kv_heads=8, head_dim=128, ffn="mlp", d_ff=53248),

    # ---------------- Llama 4 (MoE, 1 shared + top-1 routed) ----------------
    "Llama4-Scout": Model("Llama4-Scout", L=48, d=5120, vocab=202048, attn="gqa",
                          n_heads=40, n_kv_heads=8, head_dim=128, ffn="moe",
                          n_routed_experts=16, n_active_experts=1, n_shared_experts=1,
                          d_ff_expert=8192, attn_chunk=8192, global_attn_ratio=0.25),
    "Llama4-Maverick": Model("Llama4-Maverick", L=48, d=5120, vocab=202048, attn="gqa",
                             n_heads=40, n_kv_heads=8, head_dim=128, ffn="moe",
                             n_routed_experts=128, n_active_experts=1, n_shared_experts=1,
                             d_ff_expert=8192,
                             # Maverick interleaves: every 2nd layer is MoE, rest dense
                             interleave_step=2, d_ff_dense=16384,
                             attn_chunk=8192, global_attn_ratio=0.25),

    # ---------------- DeepSeek V3 (MLA + MoE) ----------------
    "DeepSeekV3": Model("DeepSeekV3", L=61, d=7168, vocab=129280, attn="mla",
                        n_heads=128, q_lora_rank=1536, kv_lora_rank=512,
                        qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                        ffn="moe", n_routed_experts=256, n_active_experts=8,
                        n_shared_experts=1, d_ff_expert=2048,
                        first_k_dense=3, d_ff_dense=18432),

    # ---------------- Kimi K2 (DeepSeek-arch MLA + MoE, 384 experts) ----------------
    "Kimi-K2": Model("Kimi-K2", L=61, d=7168, vocab=163840, attn="mla",
                     n_heads=64, q_lora_rank=1536, kv_lora_rank=512,
                     qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                     ffn="moe", n_routed_experts=384, n_active_experts=8,
                     n_shared_experts=1, d_ff_expert=2048,
                     first_k_dense=1, d_ff_dense=18432),

    # ---------------- Qwen3 ----------------
    "Qwen3-4B": Model("Qwen3-4B", L=36, d=2560, vocab=151936, attn="gqa",
                      n_heads=32, n_kv_heads=8, head_dim=128, ffn="mlp", d_ff=9728,
                      tied_embeddings=True),
    "Qwen3-30B": Model("Qwen3-30B", L=48, d=2048, vocab=151936, attn="gqa",
                       n_heads=32, n_kv_heads=4, head_dim=128, ffn="moe",
                       n_routed_experts=128, n_active_experts=8, n_shared_experts=0,
                       d_ff_expert=768),
    "Qwen3-235B": Model("Qwen3-235B", L=94, d=4096, vocab=151936, attn="gqa",
                        n_heads=64, n_kv_heads=4, head_dim=128, ffn="moe",
                        n_routed_experts=128, n_active_experts=8, n_shared_experts=0,
                        d_ff_expert=1536),

    # ---------------- GPT-OSS (MoE, top-4) ----------------
    "GPT-OSS-20B": Model("GPT-OSS-20B", L=24, d=2880, vocab=201088, attn="gqa",
                         n_heads=64, n_kv_heads=8, head_dim=64, ffn="moe",
                         n_routed_experts=32, n_active_experts=4, n_shared_experts=0,
                         d_ff_expert=2880),
    "GPT-OSS-120B": Model("GPT-OSS-120B", L=36, d=2880, vocab=201088, attn="gqa",
                          n_heads=64, n_kv_heads=8, head_dim=64, ffn="moe",
                          n_routed_experts=128, n_active_experts=4, n_shared_experts=0,
                          d_ff_expert=2880),

    # ---- models measured in the vLLM energy runs (dense GQA, float16) ----
    "Mistral-7B-v0.1": Model("Mistral-7B-v0.1", L=32, d=4096, vocab=32000, attn="gqa",
                             n_heads=32, n_kv_heads=8, head_dim=128, ffn="mlp", d_ff=14336),
    "Qwen2-7B-Instruct": Model("Qwen2-7B-Instruct", L=28, d=3584, vocab=152064, attn="gqa",
                               n_heads=28, n_kv_heads=4, head_dim=128, ffn="mlp", d_ff=18944),
    "gemma-7b": Model("gemma-7b", L=28, d=3072, vocab=256128, attn="gqa",
                      n_heads=16, n_kv_heads=16, head_dim=256, ffn="mlp", d_ff=24576,
                      tied_embeddings=True),
}


if __name__ == "__main__":
    # Paper Table 1/4 reference: (active B, total B) in billions
    paper = {
        "Llama3-8B": (7, 7), "Llama3-70B": (68, 68), "Llama3-405B": (402, 402),
        "Llama4-Scout": (15, 106), "Llama4-Maverick": (15, 399),
        "DeepSeekV3": (61, 694), "Kimi-K2": (43, 1000),
        "Qwen3-4B": (4, 4), "Qwen3-30B": (3, 30), "Qwen3-235B": (21, 234),
        "GPT-OSS-20B": (3, 20), "GPT-OSS-120B": (5, 116),
    }
    print(f"{'Model':<18}{'act(mine)':>10}{'act(pap)':>10}{'tot(mine)':>10}{'tot(pap)':>10}")
    for name, m in MODELS.items():
        if name not in paper:      # 3 vLLM-measured models have no paper Table-4 row
            continue
        act = m.active_params() / 1e9
        tot = m.total_params() / 1e9
        pa, pt = paper[name]
        print(f"{name:<18}{act:>10.1f}{pa:>10}{tot:>10.1f}{pt:>10}")
