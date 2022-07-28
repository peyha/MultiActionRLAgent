import gym
import numpy as np

class DiscretizedWrapper(gym.Wrapper):
    """Gym environment wrapper to discretize the action space.

    This wrapper work with continuous action space and discretize it
    with a given number of bins for each action.

    Args:
        nb_bin: Number of bins for each action.
        env: Environment to wrap.

    Attributes:
        discretized: Mapping for the discretized action space.
        action_space: New MultiDiscrete action_space for the environment.
    """

    def __init__(self, env: gym.Env, nb_bin: int):
        super().__init__(env)
        self.nb_bin = nb_bin
        self.discretized = np.linspace(
            env.action_space.low, env.action_space.high, self.nb_bin
        )
        self.action_space = gym.spaces.MultiDiscrete(
            np.array([self.nb_bin] * self.action_space.shape[0])
        )

    def get_discretized_array(self) -> np.array:
        """Get the discretized array.

        Returns:
            Discretized array.
        """
        return self.discretized

    def step(self, action: np.array) -> np.array:
        """Step the environment."""
        action = np.array(
            [self.discretized[a, index] for index, a in np.ndenumerate(action)]
        ).flatten()
        return super().step(action)