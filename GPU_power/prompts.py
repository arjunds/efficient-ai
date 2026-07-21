#!/usr/bin/env python3
"""
prompts.py

Load variable-length prompt pools from real datasets (alpaca, sharegpt) for the
ragged-workload runs. Standalone — uses `datasets` lazily and does NOT import the
old energy_profile.py (which drags pandas). `datasets` is pip-installed to the
shared pydeps dir by the sbatch.

Variable prompt lengths are the whole point: they produce heterogeneous running
batches (contexts differ within a batch), which is what exercises the per-iteration
Σ_seq KV(ctx_seq) term and spreads arithmetic intensity (sharegpt's long prompts
give the prefill-heavy, high-AI points that pin e_flop).
"""

import random
from typing import List


def _alpaca_prompt(ex: dict) -> str:
    instr = ex.get("instruction", "")
    inp = ex.get("input", "") or ""
    if inp.strip():
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n"
    return f"### Instruction:\n{instr}\n\n### Response:\n"


def _sharegpt_first_human(ex: dict) -> str:
    """First human turn as a single-turn request (variable, sometimes long)."""
    conv = ex.get("conversations") or []
    for turn in conv:
        if turn.get("from") in ("human", "user"):
            v = (turn.get("value") or "").strip()
            if v:
                return v
    return ""


def load_prompts(task: str, n: int, seed: int = 0,
                 min_chars: int = 16, max_chars: int = 24000) -> List[str]:
    """Return up to n prompt strings for the task (shuffled, length-filtered)."""
    from datasets import load_dataset
    rng = random.Random(seed)
    out: List[str] = []

    if task == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        for i in idx:
            p = _alpaca_prompt(ds[i])
            if min_chars <= len(p) <= max_chars:
                out.append(p)
            if len(out) >= n:
                break

    elif task == "sharegpt":
        # datasets can't auto-infer this community repo's layout, so pull the raw
        # JSON via huggingface_hub + stdlib json. Conversations vary from short to
        # very long -> the arithmetic-intensity spread (long prompts = prefill-heavy
        # high-AI points that pin e_flop).
        import json as _json
        from huggingface_hub import hf_hub_download, list_repo_files
        repo = "anon8231489123/ShareGPT_Vicuna_unfiltered"
        jsons = [f for f in list_repo_files(repo, repo_type="dataset")
                 if f.endswith(".json")]
        if not jsons:
            raise RuntimeError(f"no .json files in {repo}")
        # prefer the cleaned/split conversation file
        jsons.sort(key=lambda f: (("clean" not in f), ("split" not in f), len(f)))
        path = hf_hub_download(repo, jsons[0], repo_type="dataset")
        data = _json.load(open(path))
        rng.shuffle(data)
        for ex in data:
            p = _sharegpt_first_human(ex)
            if p and min_chars <= len(p) <= max_chars:
                out.append(p)
            if len(out) >= n:
                break

    else:
        raise ValueError(f"unknown task: {task}")

    if not out:
        raise RuntimeError(f"no prompts loaded for task={task}")
    return out


def dump(task: str, n: int, out_path: str, seed: int = 0):
    """Dump a prompt pool to a JSON list of strings. Run this in a step that has
    `datasets` on PYTHONPATH; the vLLM run then reads the file with stdlib json
    (so datasets' huggingface-hub 1.x never pollutes the transformers/vLLM env)."""
    import json
    ps = load_prompts(task, n, seed)
    with open(out_path, "w") as f:
        json.dump(ps, f)
    lens = sorted(len(p) for p in ps)
    print(f"[dump] {task} -> {out_path}: {len(ps)} prompts, char "
          f"min/median/max={lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 5 and sys.argv[1] == "dump":
        dump(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        t = sys.argv[1] if len(sys.argv) > 1 else "alpaca"
        ps = load_prompts(t, 20)
        lens = sorted(len(p) for p in ps)
        print(f"{t}: {len(ps)} prompts, char len min/median/max="
              f"{lens[0]}/{lens[len(lens)//2]}/{lens[-1]}")
