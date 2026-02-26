import torch
import numpy as np
import hockey.hockey_env as h_env
from train_hockey_2 import Actor

if torch.cuda.is_available():
    device = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

#comaprison checkpoints here, since all actors in all configs have same architecture, we load them in the same way
CHECKPOINT_1 = "sac_sb3defaults_checkpoint_ep7000.pt"
CHECKPOINT_2 = "../sac_nolnf_checkpoint_ep9000.pt"
EPISODES = 10

#load agent
def load_agent(env, checkpoint_path):
    state_dim  = env.observation_space.shape[0]
    action_dim = 4
    action_scale = torch.FloatTensor([1.0, 1.0, 1.0, 1.0]).to(device)
    action_bias  = torch.FloatTensor([0.0, 0.0, 0.0, 0.0]).to(device)
    agent = Actor(state_dim, action_dim,
                  action_scale=action_scale,
                  action_bias=action_bias).to(device)

    # only load actor weights to avoids critic architecture mismatch
    ckpt = torch.load(checkpoint_path, map_location=device)
    # handle both full checkpoint bot use actor only
    if "actor" in ckpt:
        agent.load_state_dict(ckpt["actor"])
    else:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'actor' key.")
    agent.eval()
    return agent

def play():
    env = h_env.HockeyEnv()

    player1 = load_agent(env, CHECKPOINT_1)
    player2 = load_agent(env, CHECKPOINT_2)

    winners = []

    obs, info = env.reset()
    obs_agent2 = env.obs_agent_two()
    _ = env.render()  # open window

    for episode in range(EPISODES):
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()

        for step in range(251):
            env.render(mode="human")

            with torch.no_grad():
                st1 = torch.FloatTensor(obs).unsqueeze(0).to(device)
                a1  = player1.get_deterministic_action(st1).squeeze(0).cpu().numpy()

                st2 = torch.FloatTensor(obs_agent2).unsqueeze(0).to(device)
                a2  = player2.get_deterministic_action(st2).squeeze(0).cpu().numpy()

            obs, r, d, t, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()

            if d or t:
                winners.append(info["winner"])
                break

        print(f"Episode {episode+1} | Winner: {info['winner']}")

    print("\nFINAL RESULTS")
    print(f"Player1 Wins: {winners.count(1)} | Player2 Wins: {winners.count(-1)} | Draws: {winners.count(0)}")
    env.close()

if __name__ == "__main__":
    play()