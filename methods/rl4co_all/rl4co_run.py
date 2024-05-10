import torch

from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModel
from rl4co.utils import RL4COTrainer
from lightning.pytorch import seed_everything

import methods.utils as utils


class RL4CO(utils.VRPInstance):
    """A class for implementing the methods included in RL4CO on a VRP instance."""

    def __init__(self, instance, init_method, customers, seed):
        super().__init__(instance)
        self.trainer = None
        self.model = None
        seed_everything(seed, workers=True)
        self.init_method = init_method
        self.customers = customers
        self.env = CVRPEnv(num_loc=self.customers)

    def set_model(self):
        if self.init_method == 'am':
            self.model = AttentionModel(self.env,
                                        baseline="rollout",
                                        train_data_size=250_000,
                                        test_data_size=10_000,
                                        optimizer_kwargs={'lr': 1e-4})

    def train_model(self):
        trainer_kwargs = {'accelerator': "auto"}
        self.trainer = RL4COTrainer(max_epochs=100, **trainer_kwargs)
        self.trainer.fit(self.model)

    @staticmethod
    def normalize_coord(coord: torch.Tensor) -> torch.Tensor:
        """From https://github.com/ai4co/rl4co/blob/v0.3.3/notebooks/tutorials/6-test-on-cvrplib.ipynb"""
        x, y = coord[:, 0], coord[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_scaled = (x - x_min) / (x_max - x_min)
        y_scaled = (y - y_min) / (y_max - y_min)
        coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
        return coord_scaled

    def test_model(self):
        self.trainer.test(self.model)
