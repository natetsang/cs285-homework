from cs285.models.ff_model import FFModel
import numpy as np

from .base_policy import BasePolicy


class MPCPolicy(BasePolicy):

    def __init__(self,
                 env,
                 ac_dim,
                 dyn_models,
                 horizon,
                 N,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.env = env
        self.dyn_models = dyn_models
        self.horizon = horizon
        self.N = N
        self.data_statistics = None  # NOTE must be updated from elsewhere

        self.ob_dim = self.env.observation_space.shape[0]

        # action space
        self.ac_space = self.env.action_space
        self.ac_dim = ac_dim
        self.low = self.ac_space.low
        self.high = self.ac_space.high

    def sample_action_sequences(self, num_sequences: int, horizon: int):
        # uniformly sample trajectories and return an array of
        # dimensions (num_sequences, horizon, self.ac_dim) in the range
        # [self.low, self.high]
        random_action_sequences = self.low + np.random.random(
            (num_sequences, horizon, self.ac_dim)) * (self.high - self.low)
        return random_action_sequences

    def get_action(self, obs):
        # print("         about to take an action")

        if self.data_statistics is None:
            # print("WARNING: performing random actions.")
            return self.sample_action_sequences(num_sequences=1, horizon=1)[0]

        # sample random actions (N x horizon)
        # print("Getting candidate action sequences")
        candidate_action_sequences = self.sample_action_sequences(
            num_sequences=self.N, horizon=self.horizon)

        # for each model in ensemble:
        predicted_sum_of_rewards_per_model = []
        for i, model in enumerate(self.dyn_models):
            # print("                     Calculating sum of rewards for model ", i)
            sum_of_rewards = self.calculate_sum_of_rewards(
                obs, candidate_action_sequences, model)
            predicted_sum_of_rewards_per_model.append(sum_of_rewards)

        # calculate mean_across_ensembles(predicted rewards)
        predicted_rewards = np.mean(
            predicted_sum_of_rewards_per_model, axis=0)  # [ens, N] --> N

        # pick the action sequence and return the 1st element of that sequence
        best_action_sequence = \
            candidate_action_sequences[predicted_rewards.argmax()]
        action_to_take = best_action_sequence[0]
        return action_to_take[None]  # Unsqueeze the first index

    def calculate_sum_of_rewards(
        self,
        obs: np.ndarray,
        candidate_action_sequences: np.ndarray,
        model: FFModel,
    ):
        """

        :param obs: numpy array with the current observation. Shape [D_obs]
        :param candidate_action_sequences: numpy array with the candidate action
        sequences. Shape [N, H, D_action] where
            - N is the number of action sequences considered
            - H is the horizon
            - D_action is the action of the dimension
        :param model: The current dynamics model.
        :return: numpy array with the sum of rewards for each action sequence.
        The array should have shape [N].
        """
        # For each candidate action sequence, predict a sequence of
        # states for each dynamics model in your ensemble.
        # Once you have a sequence of predicted states from each model in
        # your ensemble, calculate the sum of rewards for each sequence
        # using `self.env.get_reward(predicted_obs)`
        # You should sum across `self.horizon` time step.
        # Hint: you should use model.get_prediction and you shouldn't need
        #       to import pytorch in this file.
        # Hint: Remember that the model can process observations and actions
        #       in batch, which can be much faster than looping through each
        #       action sequence.

        N, H, _ = candidate_action_sequences.shape

        pred_obs = np.zeros((N, H, self.ob_dim))
        pred_obs[:, 0] = np.tile(obs[None, :], (N, 1))
        rewards = np.zeros((N, H))
        for t in range(H):
            rewards[:, t], _ = self.env.get_reward(
                pred_obs[:, t], candidate_action_sequences[:, t])
            if t < H - 1:
                pred_obs[:, t + 1] = model.get_prediction(
                    pred_obs[:, t],
                    candidate_action_sequences[:, t],
                    self.data_statistics,
                )

        sum_of_rewards = rewards.sum(axis=1)
        assert sum_of_rewards.shape == (N,)
        return sum_of_rewards
