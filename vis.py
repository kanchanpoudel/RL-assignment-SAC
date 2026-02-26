
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# load logs saved durngtraining
log_file = "/Users/kanchanpoudel/uni/RL/sa4c_again_training_log.csv"
df = pd.read_csv(log_file)

# Create folder to save plots
save_dir = "training_plots"
os.makedirs(save_dir, exist_ok=True)

# Set plotting style
sns.set(style="whitegrid", palette="muted", font_scale=1.2)


smooth_window = 10
df['actor_loss_smooth'] = df['actor_loss'].rolling(smooth_window, min_periods=1).mean()
df['critic_loss_smooth'] = df['critic_loss'].rolling(smooth_window, min_periods=1).mean()
df['avg_reward_10ep_smooth'] = df['avg_reward_10ep'].rolling(smooth_window, min_periods=1).mean()
df['win_rate_smooth'] = df['win_rate'].rolling(smooth_window, min_periods=1).mean()
df['steps_per_win_rate'] = df['total_steps'] / df['win_rate'].replace(0, np.nan)
df['steps_per_win_rate_smooth'] = df['steps_per_win_rate'].rolling(smooth_window, min_periods=1).mean()

# reward and win rate
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()

sns.lineplot(x="episode", y="avg_reward_10ep_smooth", data=df, ax=ax1, label="Avg Reward (10ep)", color="tab:blue")
sns.lineplot(x="episode", y="win_rate_smooth", data=df, ax=ax2, label="Win Rate", color="tab:green")

ax1.set_xlabel("Episode")
ax1.set_ylabel("Avg Reward (10 ep)", color="tab:blue")
ax2.set_ylabel("Win Rate", color="tab:green")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:green")

fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

line1 = ax1.plot(df["episode"], df["avg_reward_10ep_smooth"],
                 color="tab:blue", linewidth=2,
                 label="Avg Reward (10ep)")[0]

line2 = ax2.plot(df["episode"], df["win_rate_smooth"],
                 color="tab:green", linewidth=2,
                 label="Win Rate")[0]

ax1.set_xlabel("Episode")
ax1.set_ylabel("Avg Reward (10 ep)", color="tab:blue")
ax2.set_ylabel("Win Rate", color="tab:green")

ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:green")

ax1.set_title("Training Progress")


fig.legend(handles=[line1, line2],
           loc="upper center",
           ncol=2,
           frameon=False,
           bbox_to_anchor=(0.5, 1.02))

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "reward_win_rate.png"), dpi=300)
plt.show()

#plotting alpha
plt.figure(figsize=(12, 4))
sns.lineplot(x="episode", y="alpha", data=df, label="Alpha", color="tab:orange")
plt.title("Entropy Coefficient (Alpha) Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Alpha")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "alpha.png"))
plt.show()

#actor critic loss
fig, ax1 = plt.subplots(figsize=(12, 5))
ax2 = ax1.twinx()

sns.lineplot(x="episode", y="actor_loss_smooth", data=df, ax=ax1, label="Actor Loss", color="tab:blue")
sns.lineplot(x="episode", y="critic_loss_smooth", data=df, ax=ax2, label="Critic Loss", color="tab:red")

ax1.set_xlabel("Episode")
ax1.set_ylabel("Actor Loss", color="tab:blue")
ax2.set_ylabel("Critic Loss", color="tab:red")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:red")

# ax2.set_yscale('log')

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

line1 = ax1.plot(df["episode"], df["actor_loss_smooth"],
                 color="tab:blue", linewidth=2,
                 label="Actor Loss")[0]

line2 = ax2.plot(df["episode"], df["critic_loss_smooth"],
                 color="tab:red", linewidth=2,
                 label="Critic Loss")[0]

ax1.set_xlabel("Episode")
ax1.set_ylabel("Actor Loss", color="tab:blue")
ax2.set_ylabel("Critic Loss", color="tab:red")

ax1.tick_params(axis='y', labelcolor="tab:blue")
ax2.tick_params(axis='y', labelcolor="tab:red")

fig.legend(handles=[line1, line2],
           loc="upper center",
           ncol=2,
           frameon=False,
           bbox_to_anchor=(0.5, 1.02))


plt.tight_layout()
plt.savefig(os.path.join(save_dir, "actor_critic_loss.png"), dpi=300)
plt.show()
plt.show()

# q1 mean
plt.figure(figsize=(12, 4))
sns.lineplot(x="episode", y="q1_mean", data=df, label="Q1 Mean", color="tab:purple")
plt.title("Average Q1 Value Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Q1 Mean")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "q1_mean.png"))
plt.show()

# wins, losses, draws
plt.figure(figsize=(12, 5))
sns.lineplot(x="episode", y="wins", data=df, label="Wins", color="green")
sns.lineplot(x="episode", y="losses", data=df, label="Losses", color="red")
sns.lineplot(x="episode", y="draws", data=df, label="Draws", color="orange")
plt.title("Wins/Losses/Draws Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "wins_losses_draws.png"))
plt.show()

# total steps
plt.figure(figsize=(12, 4))
sns.lineplot(x="episode", y="total_steps", data=df, label="Total Steps", color="tab:cyan")
plt.title("Cumulative Steps Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Steps")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "total_steps.png"))
plt.show()

# Sample efficiency: steps per win rate
plt.figure(figsize=(12, 5))
sns.lineplot(x="episode", y="steps_per_win_rate_smooth", data=df,
             label="Steps per Win Rate (Smoothed)", color="tab:blue")
plt.title("Sample Efficiency: Steps per Unit Win Rate (Smoothed)")
plt.xlabel("Episode")
plt.ylabel("Steps / Win Rate")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "steps_per_win_rate.png"))
plt.show()

print(f"All plots saved in folder: {save_dir}")