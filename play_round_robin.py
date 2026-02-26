"""
Round-robin evaluation between saved SAC checkpoints.
Produces a win-rate matrix (heatmap) and Elo rating curve.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
import hockey.hockey_env
import itertools
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from train_hockey_2 import Actor
# ================= DEVICE =================
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

LOG_STD_MIN = -5
LOG_STD_MAX = 2
CLIP_MEAN   = 2.0

# ================= EDIT THIS =================
CHECKPOINT_PATHS = [
    ("noln7000",   "../sac_nolnf_checkpoint_ep7000.pt"),
    ("noln8000",  "../sac_nolnf_checkpoint_ep8000.pt"),
    ("noln9000",  "../sac_nolnf_checkpoint_ep9000.pt"),
    ("noln10000",  "../sac_nolnf_checkpoint_ep10000.pt"),
    ("def7000",  "sac_sb3defaults_checkpoint_ep7000.pt"),
    ("def8000",  "sac_sb3defaults_checkpoint_ep8000.pt"),
    ("def9000",  "sac_sb3defaults_checkpoint_ep9000.pt"),
    ("def10000",  "sac_sb3defaults_checkpoint_ep10000.pt"),
    
]
GAMES_PER_PAIR = 20   # episodes per matchup
ELO_K          = 32
ELO_INIT       = 1000



def load_actor(path, state_dim, action_dim, action_scale, action_bias):
    actor = Actor(state_dim, action_dim,
                  action_scale=action_scale,
                  action_bias=action_bias).to(device)
    ckpt = torch.load(path, map_location=device)
    # handle both full checkpoint bot use actor only
    if "actor" in ckpt:
        actor.load_state_dict(ckpt["actor"])
    else:
        raise ValueError(f"Checkpoint {path} does not contain 'actor' key.")
    actor.eval()
    return actor


def get_action(actor, state):
    with torch.no_grad():
        st = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor.get_deterministic_action(st)
    return action.squeeze(0).cpu().numpy()


def play_match(actor1, actor2, env, episodes=20):
    """
    actor1 plays as player 1 (agent), actor2 plays as player 2 (opponent).
    """
    wins1 = losses1 = draws1 = 0


    import hockey.hockey_env as h_env

    for _ in range(episodes):
        env_h = h_env.HockeyEnv()
        obs1, info = env_h.reset()
        obs2 = env_h.obs_agent_two()
        done = False

        while not done:
            a1 = get_action(actor1, obs1.astype(np.float32))
            a2 = get_action(actor2, obs2.astype(np.float32))
            obs1, reward, done, truncated, info = env_h.step(np.hstack([a1, a2]))
            obs2 = env_h.obs_agent_two()
            if truncated:
                break

        winner = info.get("winner", 0)
        if winner == 1:
            wins1 += 1
        elif winner == -1:
            losses1 += 1
        else:
            draws1 += 1
        env_h.close()

    total = wins1 + losses1 + draws1
    wr1 = wins1 / total if total > 0 else 0
    wr2 = losses1 / total if total > 0 else 0
    return wr1, wr2


# ELO 
def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def update_elo(ra, rb, score_a, k=ELO_K):
    ea = expected_score(ra, rb)
    new_ra = ra + k * (score_a - ea)
    new_rb = rb + k * ((1 - score_a) - (1 - ea))
    return new_ra, new_rb


def main():
    # set up env to get dims
    env = gym.make("Hockey-One-v0", mode="NORMAL", weak_opponent=False)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_high  = env.action_space.high
    action_low   = env.action_space.low
    action_scale = torch.FloatTensor((action_high - action_low) / 2.0).to(device)
    action_bias  = torch.FloatTensor((action_high + action_low) / 2.0).to(device)
    env.close()

    # load only checkpoints that exist
    import os
    available = [(label, path) for label, path in CHECKPOINT_PATHS if os.path.exists(path)]
    if len(available) < 2:
        print("Need at least 2 checkpoints. Found:", [p for _, p in available])
        return

    print(f"Found {len(available)} checkpoints: {[l for l, _ in available]}")

    actors = {}
    for label, path in available:
        print(f"Loading {label} from {path}...")
        actors[label] = load_actor(path, state_dim, action_dim, action_scale, action_bias)

    labels = [l for l, _ in available]
    n = len(labels)

    # rr matrix 
    win_matrix = np.zeros((n, n))  # win_matrix[i][j] = WR of i vs j

    results_log = []

    for i, j in itertools.permutations(range(n), 2):
        label_i = labels[i]
        label_j = labels[j]
        print(f"  {label_i} vs {label_j}...")
        wr_i, wr_j = play_match(actors[label_i], actors[label_j], env=None, episodes=GAMES_PER_PAIR)
        win_matrix[i][j] = wr_i
        results_log.append((label_i, label_j, wr_i, wr_j))
        print(f"    {label_i} WR: {wr_i:.2f} | {label_j} WR: {wr_j:.2f}")

    # save matrix to csv
    with open("roundrobin_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["agent", "opponent", "win_rate", "opponent_win_rate"])
        for row in results_log:
            w.writerow(row)

    # Elo ratings
    elo = {label: ELO_INIT for label in labels}
    elo_history = {label: [ELO_INIT] for label in labels}

    for label_i, label_j, wr_i, wr_j in results_log:
        score_i = wr_i  # use win rate as score (draws = 0.5 implicitly via wr)
        new_i, new_j = update_elo(elo[label_i], elo[label_j], score_i)
        elo[label_i] = new_i
        elo[label_j] = new_j

    print("\nFinal Elo Ratings:")
    for label in sorted(elo, key=lambda x: elo[x], reverse=True):
        print(f"  {label}: {elo[label]:.1f}")

    # Plot 1: win rate heatmap 
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(
        win_matrix,
        annot=True, fmt=".2f",
        xticklabels=labels, yticklabels=labels,
        cmap="RdYlGn", vmin=0, vmax=1,
        ax=ax
    )
    ax.set_title("Round-Robin Win Rate Matrix\n(row agent vs column agent)")
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")
    plt.tight_layout()
    plt.savefig("roundrobin_heatmap.png", dpi=150)
    plt.close()
    print("Saved roundrobin_heatmap.png")

    # Plot 2: Elo bar chart 
    fig, ax = plt.subplots(figsize=(max(6, n), 4))
    elo_values = [elo[l] for l in labels]
    bars = ax.bar(labels, elo_values, color="steelblue", edgecolor="black")
    ax.axhline(ELO_INIT, color="gray", linestyle="--", linewidth=1, label=f"Initial Elo ({ELO_INIT})")
    ax.set_title("Elo Ratings by Checkpoint")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Elo Rating")
    ax.legend()
    for bar, val in zip(bars, elo_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("elo_ratings.png", dpi=150)
    plt.close()
    print("Saved elo_ratings.png")


if __name__ == "__main__":
    main()