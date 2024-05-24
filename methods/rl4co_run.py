import torch
from einops import repeat
from math import ceil

from rl4co.envs import CVRPEnv, CVRPTWEnv
from rl4co.models import AttentionModel, AMPPO, SymNCO, POMO, MDAM, DeepACO
from rl4co.utils import RL4COTrainer
from lightning.pytorch import seed_everything

import methods.utils as utils


class RL4CO:
    """A class for implementing the methods included in RL4CO on a VRP instance."""

    def __init__(self, problem, init_method, customers, seed, ident):
        self.routes = None
        self.cost = None
        self.trainer = None
        self.model = None
        self.ident = ident
        seed_everything(seed, workers=True)
        self.init_method = init_method
        self.customers = int(customers)
        self.problem = problem
        if self.problem == "CVRP":
            self.env = CVRPEnv(num_loc=self.customers)
        elif self.problem == "CVRPTW":
            self.env = CVRPTWEnv(generator_params={'num_loc': self.customers})

    def set_model(self):
        if self.init_method == "am":
            self.model = AttentionModel(
                self.env,
                train_data_size=20_000,
                test_data_size=10_000,
                optimizer_kwargs={"lr": 1e-4},
            )
        elif self.init_method == "amppo":
            self.model = AMPPO(
                self.env,
                train_data_size=250_000,
                test_data_size=10_000,
                optimizer_kwargs={"lr": 1e-4},
            )
        elif self.init_method == "symnco":
            self.model = SymNCO(
                self.env,
                train_data_size=250_000,
                test_data_size=10_000,
                optimizer_kwargs={"lr": 1e-4},
            )
        elif self.init_method == "pomo":
            self.model = POMO(
                self.env,
                train_data_size=100_000,
                test_data_size=10_000,
                optimizer_kwargs={"lr": 1e-4},
            )
        elif self.init_method == "mdam":
            self.model = MDAM(
                self.env,
                train_data_size=250_000,
                test_data_size=10_000,
                optimizer_kwargs={"lr": 1e-4},
            )
        elif self.init_method == "deepaco":
            self.model = DeepACO(
                self.env,
                train_data_size=640,
                test_data_size=320,
                optimizer_kwargs={"lr": 1e-4},
            )

    def train_model(self):
        if self.init_method == "deepaco":
            epochs = 1
        else:
            epochs = 1
            # Currently ignoring POMO instructions for 2000 epochs
        trainer_kwargs = {
            "accelerator": "auto",
            "default_root_dir": f"results/exp_{self.ident}",
        }
        self.trainer = RL4COTrainer(max_epochs=epochs, **trainer_kwargs)
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

    def routing(self, out):
        # Routing
        self.routes = []
        current = []
        for node in out["actions"][0]:
            if node == 0:
                self.routes.append(current)
                current = []
            else:
                current.append(int(node))
        self.routes.append(current)
        self.routes = [route for route in self.routes if len(route) != 0]

    def single_test(self, instance):
        """Test for a single instance"""
        coords = torch.tensor(instance["node_coord"]).float()
        coords_norm = self.normalize_coord(coords)
        demand = torch.tensor(instance["demand"][1:]).float()
        capacity = instance["capacity"]
        if self.problem == "CVRPTW":
            durations = torch.tensor(instance["service_time"]).float()
            time_windows = torch.tensor(instance["time_window"]).float()
        n = coords.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = self.model.policy
        policy = policy.to(device)

        # Prepare the tensordict
        batch_size = 2
        td = self.env.reset(batch_size=(batch_size,)).to(device)
        td["locs"] = repeat(coords_norm, "n d -> b n d", b=batch_size, d=2)
        td["demand"] = repeat(demand, "n -> b n", b=batch_size) / capacity
        td["visited"] = torch.zeros((batch_size, 1, n), dtype=torch.uint8)
        if self.problem == "CVRPTW":
            td["durations"] = repeat(durations, "n -> b n", b=batch_size)
            td["time_windows"] = repeat(time_windows, "n d -> b n d", b=batch_size, d=2)
        action_mask = torch.ones(batch_size, n, dtype=torch.bool)
        action_mask[:, 0] = False
        td["action_mask"] = action_mask

        # Get the solution from the policy
        with torch.no_grad():
            out = policy(td.clone(), decode_type="greedy", return_actions=True)

        # Calculate the cost on the original scale
        td["locs"] = repeat(coords, "n d -> b n d", b=batch_size, d=2)
        neg_reward = self.env.get_reward(td, out["actions"])
        self.cost = ceil(-1 * neg_reward[0].item())

        self.routing(out)
