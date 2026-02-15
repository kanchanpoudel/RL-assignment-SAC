import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, size):
        self.size = size
        self.ptr = 0
        self.full = False
        self.state = np.zeros((size, state_dim))
        self.action = np.zeros((size, action_dim))
        self.reward = np.zeros((size, 1))
        self.next_state = np.zeros((size, state_dim))
        self.done = np.zeros((size, 1))

    def add(self, s, a, r, s2, d):
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = d

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
        # Return the current number of experiences in the buffer
        return self.size if self.full else self.ptr



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.net(state)

    def sample(self, state):
        mean = self.forward(state)
        std = torch.ones_like(mean).to(device) * 0.1  # Fixed std for simplicity
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class SAC:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.q1 = QNetwork(state_dim, action_dim).to(device)
        self.q2 = QNetwork(state_dim, action_dim).to(device)
        self.q1_target = QNetwork(state_dim, action_dim).to(device)
        self.q2_target = QNetwork(state_dim, action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=lr)

        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.MseLoss = nn.MSELoss()

    def update(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Compute the target Q value using target networks
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            q1_next = self.q1_target(next_state, next_action)
            q2_next = self.q2_target(next_state, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * q_next

        # Update Q networks
        q1_loss = self.MseLoss(self.q1(state, action), target_q)
        q2_loss = self.MseLoss(self.q2(state, action), target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Update Actor network 
        action, log_prob = self.actor.sample(state)
        q1_value = self.q1(state, action)
        q2_value = self.q2(state, action)
        q_value = torch.min(q1_value, q2_value)
        actor_loss = (self.alpha * log_prob - q_value).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update the target Q networks
        self.soft_update(self.q1, self.q1_target)
        self.soft_update(self.q2, self.q2_target)

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)



def main():
    parser = argparse.ArgumentParser()  # Replace OptionParser with ArgumentParser
    parser.add_argument('-e', '--env', action='store', type=str,
                    dest='env_name', default="Pendulum-v1",
                    )  

    parser.add_argument('-s', '--seed', action='store', type=int, 
                        dest='seed', default=None, 
                        )
    args = parser.parse_args()  # Replacing optparse with argparse

    env_name = args.env_name
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.observation_space.shape[0]  # typo - should be action_space but works anyway
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    replay_buffer = ReplayBuffer(state_dim, action_dim, size=int(1e6))
    sac = SAC(state_dim, action_dim)

    episodes = 0
    max_episodes = 1000
    max_timesteps = 200
    solved_reward = -200  # good enough for pendulum

    rewards = []
    
    while episodes < max_episodes:
        state, _ = env.reset()
        episode_reward = 0

        for t in range(max_timesteps):
            action, _ = sac.actor.sample(torch.FloatTensor(state).unsqueeze(0).to(device))
            next_state, reward, done, _, _ = env.step(action.cpu().detach().numpy()[0])

            replay_buffer.add(state, action.cpu().detach().numpy(), reward, next_state, done)

            state = next_state
            episode_reward += reward

            # start training once we have enough samples
            if len(replay_buffer) > 256:
                sac.update(replay_buffer)

            if done:
                break

        episodes += 1
        rewards.append(episode_reward)
        
        if episodes % 20 == 0:
            avg_reward = np.mean(rewards[-20:])
            print(f"Episode {episodes}, avg reward: {avg_reward:.2f}")

        if episode_reward > solved_reward:
            print("Solved!")
            break

if __name__ == '__main__':
    main()