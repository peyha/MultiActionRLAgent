import gym
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils.ReplayMemory import ExperienceReplay

class BranchingQNetwork(nn.Module):
    """Neural Network for the Q-function.

    Args:
        state_dim: Dimension of the state.
        action_dim: Dimension of the action.
        nb_action: Number of possible action for each dimension.

    Attributes:
        model: The neural network model.
        value_net: The neural network model for the value function.
        advantage_net: The neural network model for the advantage function.
    """

    def __init__(self, state_dim: int, action_dim: int, nb_action: int):
        super().__init__()

        self.action_dim = action_dim
        self.nb_action = nb_action
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.value_net = nn.Linear(128, 1)
        self.advantage_net = nn.ModuleList(
            [nn.Linear(128, nb_action) for i in range(action_dim)]
        )

    def forward(self, state: np.array) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            state: Current state of the environment.
        """
        out = self.model(state)
        value = self.value_net(out)
        advantages = torch.stack(
            [net(out) for net in self.advantage_net], dim=1
        )

        q_values = (
            value.unsqueeze(-1) + advantages - advantages.mean(2, keepdim=True)
        )

        return q_values


class BranchingDQNAgent:
    """Branching Q Network Agent.

    Args:
        env: Environment to train on.
        nb_action: Number of possible action for each dimension.
        gamma: Discount factor.
        batch_size: Size of the batch.
        learning_rate: Learning rate of the optimizer.
        epsilon_decay: Decay rate of the epsilon.
        min_epsilon: Minimum value of the epsilon.
        net_update_freq: Frequency of the network update.

    Attributes:
        memory: Replay memory of the agent.
        model: The neural network model.
        target: The target network.
        optimizer: The optimizer of the network.
        epsilon: The epsilon value of the agent.
        iter: The number of update iterations.
    """

    def __init__(
        self,
        env: gym.Env,
        discrete_to_continuous: np.array,
        nb_action: int,
        gamma: float = 0.99,
        batch_size: int = 128,
        learning_rate: float = 0.0001,
        epsilon_decay: float = 1e-4,
        min_epsilon: float = 0.01,
        net_update_freq: int = 1000,

    ):
        super().__init__()
        state_dim, action_dim = (
            env.observation_space.shape[0],
            env.action_space.shape[0],
        )
        self.memory = ExperienceReplay(state_dim, action_dim)
        self.model = BranchingQNetwork(state_dim, action_dim, nb_action)
        self.target = BranchingQNetwork(state_dim, action_dim, nb_action)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate
        )
        self.discrete_to_continuous = discrete_to_continuous
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.net_update_freq = net_update_freq
        self.iter = 0

        self.env = env

    def get_epsilon_action(self, state: np.array) -> np.array:
        """Get an action with epsilon-greedy policy.

        Args:
            state: Current state of the environment.

        Returns:
            action: Action to take."""
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.min_epsilon)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return self.get_discrete_action(state)

    def get_action(self, state, smartgrid):
        """Get an action with model.

        Args:
            state: Current state of the environment.

        Returns:
            action: Action to take."""
        discrete_action = self.get_discrete_action(state, smartgrid)
        action = np.array(
            [self.discrete_to_continuous[a, index] for index, a in np.ndenumerate(discrete_action)]
        ).flatten()
        return action

    def get_discrete_action(self, state, smartgrid = None):
        """Get an action with model.

        Args:
            state: Current state of the environment.

        Returns:
            action: Action to take."""
        out = self.model(
            torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
        ).squeeze(0)
        action = torch.argmax(out, dim=1)
        return action.numpy()

    def update_policy(self):
        """Update the policy of the agent."""
        self.iter += 1

        if self.iter < self.net_update_freq:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            min(self.batch_size, self.memory.pointer)
        )

        states = Variable(torch.from_numpy(states.astype(np.float32)))
        actions = Variable(torch.from_numpy(actions.astype(np.int64)))
        rewards = Variable(torch.from_numpy(rewards.astype(np.float32)))
        next_states = Variable(torch.from_numpy(next_states.astype(np.float32)))
        dones = Variable(torch.from_numpy(dones.astype(np.float32)))

        current_q_values = (
            self.model(states).gather(2, actions.unsqueeze(-1)).squeeze(-1)
        )

        with torch.no_grad():
            id_max = torch.argmax(self.model(next_states), dim=2)
            next_q_values = (
                self.target(next_states)
                .gather(2, id_max.unsqueeze(2))
                .squeeze(-1)
            )

        expected_q_values = rewards.unsqueeze(
            -1
        ) + next_q_values * self.gamma * (1 - dones).unsqueeze(-1)
        loss = F.mse_loss(current_q_values, expected_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        for p in self.model.parameters():
            p.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        if self.iter % self.net_update_freq == 0:
            self.target.load_state_dict(self.model.state_dict())

    def save(self):
        """Save the agent."""
        path = "./models/branching_dqn_agent.pth"
        torch.save(self.model.state_dict(), path)

    def load(self):
        """Load the agent."""
        path = "./models/branching_dqn_agent.pth"
        self.model.load_state_dict(torch.load(path))