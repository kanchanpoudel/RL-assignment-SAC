"""
Custom SAC implementation for Hockey-One-v0
For parameters and configs, mostly Based on: Stable Baselines3 SAC source (https://github.com/DLR-RM/stable-baselines3)

This is the version that achieved 100% vs weak and 85% vs strong.
Key settings vs standard SAC:
  - log_std_init=-3 (SB3 default, less random initial policy)
  - target_entropy=-0.5*action_dim (less aggressive, keeps alpha higher longer)
  - log_alpha initialised to log(0.2) (starts at alpha=0.2, not 1.0)
  - critic loss uses 0.5 * mse (SB3 gradient scaling)
  - clip_mean=2.0 on actor output (SB3 stability trick)
  - max_episodes=5000, curriculum_switch=3000
"""

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
import hockey.hockey_env  # registers Hockey-One-v0 with gymnasium

# device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

LOG_STD_MIN = -5
LOG_STD_MAX = 2
CLIP_MEAN   = 2.0


# Replay Buffer for SAC
class ReplayBuffer:
    def __init__(self, state_dim, action_dim, size):
        self.size = size
        self.ptr  = 0
        self.full = False

        self.state      = np.zeros((size, state_dim),  dtype=np.float32)
        self.action     = np.zeros((size, action_dim), dtype=np.float32)
        self.reward     = np.zeros((size, 1),          dtype=np.float32)
        self.next_state = np.zeros((size, state_dim),  dtype=np.float32)
        self.done       = np.zeros((size, 1),          dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.state[self.ptr]      = s
        self.action[self.ptr]     = a
        self.reward[self.ptr]     = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr]       = d
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size):
        max_size = self.size if self.full else self.ptr
        idx = np.random.randint(0, max_size, size=batch_size)
        return (
            torch.FloatTensor(self.state[idx]).to(device),
            torch.FloatTensor(self.action[idx]).to(device),
            torch.FloatTensor(self.reward[idx]).to(device),
            torch.FloatTensor(self.next_state[idx]).to(device),
            torch.FloatTensor(self.done[idx]).to(device),
        )

    def __len__(self):
        return self.size if self.full else self.ptr


# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 action_scale=1.0, action_bias=0.0, log_std_init=-3):
        super().__init__()
        self.action_scale = action_scale
        self.action_bias  = action_bias

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        nn.init.constant_(self.log_std_layer.bias, log_std_init)
        nn.init.constant_(self.log_std_layer.weight, 0.0)

    def forward(self, state):
        x       = self.net(state)
        mean    = torch.clamp(self.mean_layer(x), -CLIP_MEAN, CLIP_MEAN)
        log_std = torch.clamp(self.log_std_layer(x), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def action_log_prob(self, state):
        mean, log_std = self.forward(state)
        std  = log_std.exp()
        dist = Normal(mean, std)
        raw  = dist.rsample()

        action_tanh = torch.tanh(raw)
        action = action_tanh * self.action_scale + self.action_bias

        log_prob = dist.log_prob(raw)
        log_prob -= torch.log(1.0 - action_tanh.pow(2) + 1e-6)
        log_prob  = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def get_deterministic_action(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.action_scale + self.action_bias


# Critic network with LayerNorm
class Critic(nn.Module):
    """
    Non-trivial modification: LayerNorm after each hidden layer.
    Normalizes across feature dimensions per sample, stabilizing Q-value
    estimates without batch size constraints or train/eval mode complexity.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, use_layer_norm=False):
        super().__init__()

        
        def make_q():
            layers = []


            layers.append(nn.Linear(state_dim + action_dim, hidden_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, 1))

            return nn.Sequential(*layers)

        self.q1 = make_q()
        self.q2 = make_q()

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


# SAC architecture
class SAC:
    def __init__(self, state_dim, action_dim, action_scale, action_bias,
                 lr=3e-4, gamma=0.99, tau=0.005, use_critic_layernorm=False):

        self.gamma = gamma
        self.tau   = tau

        self.actor = Actor(state_dim, action_dim,
                           action_scale=action_scale,
                           action_bias=action_bias).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, action_dim,
                     use_layer_norm=use_critic_layernorm).to(device)

        self.critic_target = Critic(state_dim, action_dim,
                            use_layer_norm=use_critic_layernorm).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -0.5 * float(action_dim)
        self.log_alpha = torch.tensor(
            [np.log(0.2)], dtype=torch.float32, requires_grad=True, device=device
        )
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.action_log_prob(next_state)
            q1_next, q2_next = self.critic_target(next_state, next_action)
            next_q   = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * next_q

        q1, q2 = self.critic(state, action)
        critic_loss = 0.5 * (F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        action_pi, log_prob = self.actor.action_log_prob(state)
        q1_pi, q2_pi = self.critic(state, action_pi)
        min_q_pi   = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha":       self.alpha.item(),
            "q1_mean":     q1.mean().item(),
        }

    def save(self, path):
        torch.save({
            'actor':     self.actor.state_dict(),
            'critic':    self.critic.state_dict(),
            'log_alpha': self.log_alpha.data,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.log_alpha.data = ckpt['log_alpha']


# Evaluation
def evaluate_agent(agent, eval_env, episodes=20, label=""):
    print(f"\n--- Eval {label} ({episodes} eps) ---")
    wins, losses, draws = 0, 0, 0
    total_reward = 0

    for _ in range(episodes):
        state, _ = eval_env.reset()
        state = state.astype(np.float32)
        done = False
        ep_reward = 0

        while not done:
            with torch.no_grad():
                st = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = agent.actor.get_deterministic_action(st)
            action_np = action.squeeze(0).cpu().numpy()
            next_state, reward, done, truncated, info = eval_env.step(action_np)
            state = next_state.astype(np.float32)
            ep_reward += reward
            if truncated:
                break

        winner = info.get("winner", 0)
        if winner == 1:    wins   += 1
        elif winner == -1: losses += 1
        else:              draws  += 1
        total_reward += ep_reward

    total    = wins + losses + draws
    win_rate = wins / total if total > 0 else 0
    avg_rew  = total_reward / episodes
    print(f"Wins: {wins} | Losses: {losses} | Draws: {draws} | "
          f"Win Rate: {win_rate:.2f} | Avg Reward: {avg_rew:.2f}")
    print(".........................................")
    return win_rate


# Training
def train_hockey(
    max_episodes=10000,
    max_timesteps=500,
    eval_interval=200,
    learning_starts=1000,
    updates_per_step=1,
    batch_size=256,
    lr=3e-4,
    gamma=0.99,
    tau=0.005,
    save_interval=500,
    curriculum=True,
    curriculum_switch=3000,
    use_critic_layernorm=False,
    log_file="sac1_training_log.csv",
):
    train_env       = gym.make("Hockey-One-v0", mode="NORMAL", weak_opponent=True if curriculum else False)
    eval_env_weak   = gym.make("Hockey-One-v0", mode="NORMAL", weak_opponent=True)
    eval_env_strong = gym.make("Hockey-One-v0", mode="NORMAL", weak_opponent=False)

    state_dim  = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]

    action_high  = train_env.action_space.high
    action_low   = train_env.action_space.low
    action_scale = torch.FloatTensor((action_high - action_low) / 2.0).to(device)
    action_bias  = torch.FloatTensor((action_high + action_low) / 2.0).to(device)

    replay_buffer = ReplayBuffer(state_dim, action_dim, int(1e6))
    sac = SAC(state_dim, action_dim, action_scale, action_bias,
              lr=lr, gamma=gamma, tau=tau, use_critic_layernorm=use_critic_layernorm)

    total_steps = 0
    rewards = []
    wins, losses, draws = 0, 0, 0
    curriculum_switched = False
    metrics = {}

    print(f"Device: {device}")
    print(f"State dim: {state_dim} | Action dim: {action_dim}")
    print(f"Target entropy: {sac.target_entropy:.2f}")
    print(f"Initial alpha: {sac.alpha.item():.3f}")
    print(f"Curriculum: {'ON - switches at ep ' + str(curriculum_switch) if curriculum else 'OFF'}")

    
    log_fields = ["episode", "total_steps", "phase", "avg_reward_10ep",
                  "wins", "losses", "draws", "win_rate", "alpha",
                  "critic_loss", "actor_loss", "q1_mean"]
    csv_file = open(log_file, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=log_fields)
    csv_writer.writeheader()

    for ep in range(max_episodes):

        if curriculum and not curriculum_switched and ep >= curriculum_switch:
            print(f"\n*** Switching to STRONG opponent at episode {ep} ***\n")
            train_env.close()
            train_env = gym.make("Hockey-One-v0", mode="NORMAL", weak_opponent=False)
            curriculum_switched = True
            wins, losses, draws = 0, 0, 0

        state, _ = train_env.reset()
        state = state.astype(np.float32)
        ep_reward = 0

        for t in range(max_timesteps):
            total_steps += 1

            if total_steps < learning_starts:
                action_np = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.from_numpy(state).unsqueeze(0).to(device)
                    action, _ = sac.actor.action_log_prob(st)
                action_np = action.squeeze(0).cpu().numpy()

            next_state, reward, done, truncated, info = train_env.step(action_np)
            next_state = next_state.astype(np.float32)
            winner = info.get("winner", 0)

            reward += info.get("reward_closeness_to_puck", 0)
            reward += info.get("reward_touch_puck", 0)
            reward += info.get("reward_puck_direction", 0)

            terminal = done and not truncated
            replay_buffer.add(state, action_np, reward, next_state, float(terminal))
            state      = next_state
            ep_reward += reward

            if total_steps >= learning_starts and len(replay_buffer) >= batch_size:
                for _ in range(updates_per_step):
                    metrics = sac.update(replay_buffer, batch_size)

            if done or truncated:
                if winner == 1:    wins   += 1
                elif winner == -1: losses += 1
                else:              draws  += 1
                break

        rewards.append(ep_reward)

        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(rewards[-10:])
            total      = wins + losses + draws
            win_rate   = wins / total if total > 0 else 0
            phase = "vs WEAK" if (curriculum and not curriculum_switched) else "vs STRONG"

            print(f"Episode {ep+1} | Steps {total_steps} | {phase} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"W/L/D: {wins}/{losses}/{draws} | WR: {win_rate:.2f} | "
                  f"Alpha: {sac.alpha.item():.4f}")
            if metrics:
                print(f"  Critic Loss: {metrics['critic_loss']:.4f} | "
                      f"Actor Loss: {metrics['actor_loss']:.4f} | "
                      f"Q1 Mean: {metrics['q1_mean']:.4f}")

            # logging for train evaluation
            csv_writer.writerow({
                "episode":          ep + 1,
                "total_steps":      total_steps,
                "phase":            phase,
                "avg_reward_10ep":  round(avg_reward, 4),
                "wins":             wins,
                "losses":           losses,
                "draws":            draws,
                "win_rate":         round(win_rate, 4),
                "alpha":            round(sac.alpha.item(), 6),
                "critic_loss":      round(metrics.get("critic_loss", float("nan")), 6),
                "actor_loss":       round(metrics.get("actor_loss", float("nan")), 6),
                "q1_mean":          round(metrics.get("q1_mean", float("nan")), 6),
            })
            csv_file.flush()  

        if (ep + 1) % eval_interval == 0:
            evaluate_agent(sac, eval_env_weak,   episodes=20, label="vs Weak")
            evaluate_agent(sac, eval_env_strong, episodes=20, label="vs Strong")

        if (ep + 1) % save_interval == 0:
            sac.save(f"sac_checkpoint_ep{ep+1}.pt")
            print(f"Checkpoint saved at episode {ep+1}")

    csv_file.close()
    train_env.close()
    eval_env_weak.close()
    eval_env_strong.close()
    sac.save("sac_final.pt")
    print(f"Training complete. Model saved as sac_final.pt")
    print(f"Training log saved as {log_file}")
    return sac


if __name__ == "__main__":
    train_hockey(use_critic_layernorm=False, curriculum_switch=3000, curriculum=True, log_file="sac_training_log.csv")   