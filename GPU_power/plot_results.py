import pandas as pd
import matplotlib.pyplot as plt

# LOAD YOUR CSV
df = pd.read_csv("aggregate_results.csv")

# =========================
# 1. THROUGHPUT vs ENERGY
# =========================
plt.figure()
plt.scatter(df["avg_tokens_per_sec"], df["energy_total_j"])

for _, row in df.iterrows():
    plt.text(row["avg_tokens_per_sec"], row["energy_total_j"], row["model_short"])

plt.xlabel("Tokens per Second")
plt.ylabel("Total Energy (J)")
plt.title("Throughput vs Energy")
plt.grid()

plt.savefig("throughput_vs_energy.png", dpi=300)
plt.close()


# =========================
# 2. ENERGY PER TOKEN vs THROUGHPUT
# =========================
plt.figure()
plt.scatter(df["avg_tokens_per_sec"], df["energy_per_token_j"])

for _, row in df.iterrows():
    plt.text(row["avg_tokens_per_sec"], row["energy_per_token_j"], row["model_short"])

plt.xlabel("Tokens per Second")
plt.ylabel("Energy per Token (J/token)")
plt.title("Efficiency Tradeoff")
plt.grid()

plt.savefig("efficiency_tradeoff.png", dpi=300)
plt.close()


# =========================
# 3. POWER TRACE (FIXED)
# =========================
power = pd.read_csv("logs/gemma-7b_alpaca_float16/power.csv")

# CLEAN COLUMN NAMES
power.columns = [
    "timestamp",
    "power_w",
    "util_gpu",
    "util_mem",
    "temp_gpu"
]

# REMOVE UNITS
power["power_w"] = power["power_w"].str.replace(" W", "").astype(float)
power["util_gpu"] = power["util_gpu"].str.replace(" %", "").astype(float)
power["util_mem"] = power["util_mem"].str.replace(" %", "").astype(float)

# PARSE TIME
power["timestamp"] = pd.to_datetime(power["timestamp"])

# CONVERT TO RELATIVE TIME
power["time_s"] = (power["timestamp"] - power["timestamp"].iloc[0]).dt.total_seconds()

# PLOT
plt.figure()
plt.plot(power["time_s"], power["power_w"])

plt.xlabel("Time (s)")
plt.ylabel("Power (W)")
plt.title("Power Profile During Inference")
plt.grid()

plt.savefig("power_profile.png", dpi=300)
plt.close()


# =========================
# 4. TTFT vs THROUGHPUT
# =========================
plt.figure()
plt.scatter(df["avg_ttft_s"], df["avg_tokens_per_sec"])

for _, row in df.iterrows():
    plt.text(row["avg_ttft_s"], row["avg_tokens_per_sec"], row["model_short"])

plt.xlabel("TTFT (s)")
plt.ylabel("Tokens per Second")
plt.title("Latency vs Throughput")
plt.grid()

plt.savefig("ttft_vs_throughput.png", dpi=300)
plt.close()