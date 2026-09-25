import ast, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
L=[ast.literal_eval(l) for l in open("controls/slurm-a5k-live-85109.out") if l.startswith("{'episode'")][:12]
order=["static64","polca","pi","mpc"]; names={"static64":"always-on\n(static)","polca":"POLCA-style","pi":"PI","mpc":"MPC"}
R={c:[d for d in L if d["controller"]==c] for c in order}
base={d["episode"].split("_rep")[1]:d["j_per_tok"] for d in R["static64"]}
rel={c:[100*(d["j_per_tok"]/base[d["episode"].split("_rep")[1]]-1) for d in R[c]] for c in order}
col={"static64":"#8a8f98","polca":"#d1701a","pi":"#a33862","mpc":"#1f5fbf"}
fig,ax=plt.subplots(1,3,figsize=(12,3.8),dpi=160)
x=range(len(order))
ax[0].bar(x,[st.mean(rel[c]) for c in order],color=[col[c] for c in order],
          yerr=[[st.mean(rel[c])-min(rel[c]) for c in order],[max(rel[c])-st.mean(rel[c]) for c in order]],capsize=4)
ax[0].axhline(0,color="k",lw=.8); ax[0].set_ylabel("energy per token vs always-on (%)")
ax[0].set_title("Energy (lower is better)")
p50=[st.mean([d["ttft_p50"] for d in R[c]]) for c in order]; p99=[st.mean([d["ttft_p99"] for d in R[c]]) for c in order]
ax[1].bar([i-.18 for i in x],p50,.36,color=[col[c] for c in order],label="median")
ax[1].bar([i+.18 for i in x],p99,.36,color=[col[c] for c in order],alpha=.45,label="99th pct")
ax[1].axhline(30,ls="--",color="k",lw=1); ax[1].text(3.45,32,"30 s SLO",ha="right",fontsize=8)
ax[1].text(0,4,"0.05 s",ha="center",fontsize=8); ax[1].set_ylabel("time to first token (s)"); ax[1].set_title("Latency (median solid, p99 faded)")
ax[2].bar([i-.18 for i in x],[100*st.mean([d["ttft_viol"] for d in R[c]]) for c in order],.36,color=[col[c] for c in order])
ax[2].bar([i+.18 for i in x],[100*st.mean([d["bviol_win"] for d in R[c]]) for c in order],.36,color=[col[c] for c in order],alpha=.45)
ax[2].set_ylabel("% violated"); ax[2].set_title("Violations: SLO (solid), budget (faded)")
for a in ax:
    a.set_xticks(list(x)); a.set_xticklabels([names[c] for c in order],fontsize=8); a.grid(axis="y",alpha=.3)
    a.spines[["top","right"]].set_visible(False)
fig.suptitle("Live closed-loop test on a real vLLM server (RTX A5000, 100 W cap, Qwen2.5-3B, bursty arrivals, 3 paired reps)",fontsize=10)
fig.tight_layout(); fig.savefig("pi_brief_figs/fig_live_control.png"); print("ok")
