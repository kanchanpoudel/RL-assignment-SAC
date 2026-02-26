"""
Plots win rate vs strong opponent over episodes for SAC with curriculum vs no currciulum.

"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CUR_CSV   = "/Users/kanchanpoudel/uni/RL/SAC-RL-Kanchan/sa4c_training_log.csv"
NOCUR_CSV = "/Users/kanchanpoudel/uni/RL/SAC-RL-Kanchan/sac_noswitch_training_log.csv"
CURRICULUM_SWITCH = 3000

cur   = pd.read_csv(CUR_CSV)
nocur = pd.read_csv(NOCUR_CSV)

def smooth(y, window=50):
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values

cur_strong   = cur[cur["phase"] == "vs STRONG"]
nocur_strong = nocur[nocur["phase"] == "vs STRONG"]

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(nocur_strong["episode"], smooth(nocur_strong["win_rate"]),
        color="#F44336", linewidth=2, label="No Curriculum")
ax.plot(cur_strong["episode"], smooth(cur_strong["win_rate"]),
        color="#2196F3", linewidth=2, label="Curriculum")
ax.axvline(x=CURRICULUM_SWITCH, color="black", linestyle="--",
           linewidth=1.5, label=f"Curriculum switch (ep {CURRICULUM_SWITCH})")

ax.set_xlabel("Episode", fontsize=12)
ax.set_ylabel("Win Rate vs Strong Opponent", fontsize=12)
ax.set_title("Curriculum vs No-Curriculum: Win Rate Against Strong Opponent", fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(-0.05, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

plt.tight_layout()
plt.savefig("curriculum_comparison.png", dpi=180)
print("Saved curriculum_comparison.png")