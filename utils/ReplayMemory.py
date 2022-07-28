import numpy as np

class ExperienceReplay:
    """Experience replay buffer.

    Args:
        state_dim: Dimension of the observation space.
        action_dim: Dimension of the action space.
        buffer_size: Size of the buffer.

    Attributes:
        state_mem: Memory of the states.
        action_mem: Memory of the actions.
        reward_mem: Memory of the rewards.
        next_state_mem: Memory of the next states.
        done_mem: Memory of the done flags.
        pointer: Pointer to the current position in the memory.
    """

    def __init__(
        self, state_dim: int, action_dim: int, buffer_size: int = 1000000
    ):
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_mem = np.zeros((self.buffer_size, state_dim))
        self.action_mem = np.zeros((self.buffer_size, action_dim))
        self.reward_mem = np.zeros(self.buffer_size)
        self.done_mem = np.zeros(self.buffer_size)
        self.next_state_mem = np.zeros((self.buffer_size, state_dim))
        self.pointer = 0

    def add_exp(
        self,
        state: np.array,
        action: np.array,
        reward: int,
        next_state: np.array,
        done: bool,
    ):
        """Add an experience to the memory.

        Args:
            state: Current state of the environment.
            action: Current action of the environment.
            reward: Reward received from the environment.
            next_state: Next state of the environment.
            done: Done flag of the environment."""
        idx = self.pointer % self.buffer_size
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = done
        self.pointer += 1

    def sample(
        self, batch_size: int = 64
    ) -> (np.array, np.array, np.array, np.array, np.array):
        """Sample a batch of experiences from the memory.

        Args:
            batch_size: Size of the batch.

        Returns:
            States, actions, rewards, next states , dones of the environment.
        """
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]

        return states, actions, rewards, next_states, dones