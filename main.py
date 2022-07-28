import gym
import numpy as np
import matplotlib.pyplot as plt

from utils.DiscreteEnvWrapper import DiscretizedWrapper
from agents.BranchingDQNAgent import BranchingDQNAgent

def train(num_episodes=100, nb_bin=4):
    """Train a Branching DQN Agent.

    Returns:
        The trained agent."""

    num_iterations = 300
    # creating the environment with the wrapper (discretization)
    env = DiscretizedWrapper(gym.make("BipedalWalker-v3"), nb_bin)
    agent = BranchingDQNAgent(
        env,
        env.get_discretized_array(),
        nb_bin,
    )
    rewards = []
    for id_episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(num_iterations):
            actions = agent.get_epsilon_action(state)
            # Take actions in env and look the results
            next_state, reward, done, _ = env.step(
                actions.astype(np.int32)
            )
            # Add experience to replay memory
            agent.memory.add_exp(
                state, actions, reward, next_state, done
            )
            total_reward += reward
            # Agent optimization
            agent.update_policy()
            state = next_state
            if done:
                break
        rewards.append(total_reward)
        print(f"Episode: {id_episode} out of {num_episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon}")


    return rewards


def smooth(scalars, weight):  # Weight between 0 and 1
    """Smooth the values of a list of scalars to improve plotting."""
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (
                    1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed

if __name__ == "__main__":

    for nb_bin in [3, 4, 8, 16]:
        rewards = train(num_episodes=1000, nb_bin=nb_bin)
        plt.plot(smooth(rewards, 0.9), label=f"{nb_bin} bins")
    plt.title("Reward per episode for different number of bins")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.show()