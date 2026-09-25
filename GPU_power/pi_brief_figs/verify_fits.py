# Verification only (PI brief): pooled 1/2/3-term R^2 on H200 logs/ragged, computed-FLOPs.
import glob, os, sys, json
import numpy as np
sys.path.insert(0, "/workspace")
from reconcile_gpus import run_record
out = {}
for gpu, root in [("H200", "logs/ragged"), ("B200", "logs/B200")]:
    recs = []
    for rd in sorted(glob.glob(f"/workspace/{root}/*/*")):
        if not os.path.isdir(rd): continue
        r = run_record(rd)
        if r: recs.append(r)
    y = np.concatenate([r["Y"] for r in recs])
    W = np.concatenate([r["W"] for r in recs]); K = np.concatenate([r["K"] for r in recs]); G = np.concatenate([r["G"] for r in recs])
    def fit(X):
        c, *_ = np.linalg.lstsq(X, y, rcond=None); yh = X @ c
        return c.tolist(), float(1 - ((y-yh)**2).sum()/((y-y.mean())**2).sum())
    res = dict(n_runs=len(recs), n_bins=int(len(y)),
               bytes_only=fit(np.c_[W+K]), two_term=fit(np.c_[W+K, G]), three_term=fit(np.c_[W, K, G]))
    # energy shares for decode-heavy vs prefill-heavy bins (3-term)
    c = res["three_term"][0]
    mem = c[0]*W + c[1]*K; cmp_ = c[2]*G
    share = cmp_/(mem+cmp_)
    res["compute_share_quantiles"] = [float(np.quantile(share, q)) for q in (0.1, 0.5, 0.9, 0.99)]
    out[gpu] = res
print(json.dumps(out, indent=1))
