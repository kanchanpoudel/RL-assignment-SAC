from __future__ import annotations

import argparse
import uuid
import torch
import numpy as np
import gymnasium as gym
import hockey.hockey_env

from comprl.client import Agent, launch_client
from train_hockey_2 import Actor


#SAC Competition Agent

class MySACAgent(Agent):
    def __init__(self, model_path: str):
        super().__init__()

        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Create dummy env only to get dimensions
        env = gym.make("Hockey-One-v0", mode="NORMAL")

        state_dim  = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]

        action_high  = env.action_space.high
        action_low   = env.action_space.low

        action_scale = torch.FloatTensor(
            (action_high - action_low) / 2.0
        ).to(self.device)

        action_bias = torch.FloatTensor(
            (action_high + action_low) / 2.0
        ).to(self.device)

        env.close()

        # Initialize SAC
        self.agent = Actor(
            state_dim,
            action_dim,
            action_scale=action_scale,
            action_bias=action_bias
        )

        # Load checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        # handle both full checkpoint bot use actor only
        if "actor" in ckpt:
            self.agent.load_state_dict(ckpt["actor"])
        else:
            raise ValueError(f"Checkpoint {model_path} does not contain 'actor' key.")
        self.agent.to(self.device)
        self.agent.eval()
        print(f"Model loaded from {model_path}")
        print(f"Running on device: {self.device}")

    def get_step(self, observation: list[float]) -> list[float]:
        state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.agent.get_deterministic_action(state)

        return action.squeeze(0).cpu().numpy().tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} | "
            f"My score: {stats[0]} | Opponent: {stats[1]}"
        )



def initialize_agent(agent_args: list[str]) -> Agent:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["mysac"],
        default="mysac",
    )

    args = parser.parse_args(agent_args)

    if args.agent == "mysac":
        return MySACAgent("SAC-RL-Kanchan/sac_checkpoint_ep9000.pt")

    raise ValueError("Unknown agent type")


def main() -> None:
    launch_client(
    initialize_agent,
)


if __name__ == "__main__":
    main()
