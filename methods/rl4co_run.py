import torch
from einops import repeat
from itertools import pairwise

from rl4co.envs import CVRPEnv, CVRPTWEnv
from rl4co.models import AttentionModel, AMPPO, SymNCO, POMO, MDAM, DeepACO
from rl4co.utils import RL4COTrainer
from rl4co.utils.ops import get_distance
from lightning.pytorch import seed_everything


class RL4CO:
    """A class for implementing the methods included in RL4CO on a VRP instance."""

    def __init__(self, method, problem, init_method, customers, seed, ident, decode="greedy"):
        self.routes = None
        self.cost = None
        self.trainer = None
        self.model = None
        self.method = method
        self.ident = ident
        seed_everything(seed, workers=True)
        self.init_method = init_method
        self.customers = int(customers)
        self.problem = problem
        self.decode = decode
        if self.problem == "CVRP":
            self.env = CVRPEnv(generator_params={"num_loc": self.customers})
        elif self.problem == "CVRPTW":
            self.env = CVRPTWEnv(generator_params={"num_loc": self.customers})

    def set_model(self):
        if self.init_method == "am":
            self.model = AttentionModel(
                self.env,
                train_data_size=250_000,
                test_data_size=10_000,
                optimizer_kwargs={"lr": 1e-4},
            )
        elif self.init_method == "amppo":
            self.model = AMPPO(
                self.env,
                train_data_size=200_000,
                test_data_size=10_000,
                optimizer_kwargs={"lr": 1e-4},
            )
        elif self.init_method == "symnco":
            self.model = SymNCO(
                self.env,
                train_data_size=150_000,  # Was 250_000
                test_data_size=10_000,
                optimizer_kwargs={"lr": 1e-4},
            )
        elif self.init_method == "pomo":
            self.model = POMO(
                self.env,
                train_data_size=80_000,  # Was 100_000
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
        if self.method == "rl4co":
            if self.init_method == "deepaco":
                epochs = 1
            elif self.init_method in ["amppo", "symnco", "pomo", "mdam"]:
                epochs = 10
            else:
                epochs = 20
        elif self.method == "rl4co_mini":
            if self.init_method == "deepaco":
                epochs = 1
            else:
                epochs = 2
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

        # Print tensor shapes
        # print(f"coords: {coords.shape}")
        # print(f"coords_norm: {coords_norm.shape}")
        # print(f"demand: {demand.shape}")
        # print(f"durations: {durations.shape if self.problem == 'CVRPTW' else 'N/A'}")
        # print(f"time_windows: {time_windows.shape if self.problem == 'CVRPTW' else 'N/A'}")

        # Prepare the tensordict
        if self.problem == "CVRP":
            batch_size = 2
            td = self.env.reset(batch_size=(batch_size,)).to(device)
            td["locs"] = repeat(coords_norm, "n d -> b n d", b=batch_size, d=2)
            td["demand"] = repeat(demand, "n -> b n", b=batch_size) / capacity
            td["visited"] = torch.zeros((batch_size, 1, n), dtype=torch.uint8)
            action_mask = torch.ones(batch_size, n, dtype=torch.bool)
            action_mask[:, 0] = False
            td["action_mask"] = action_mask
        elif self.problem == "CVRPTW":
            # td = self.env.extract_from_solomon(instance, 1).to(device)
            batch_size = 1
            td = self.env.reset(batch_size=(batch_size,)).to(device)
            td["locs"] = repeat(coords, "n d -> b n d", b=batch_size, d=2)
            td["distances"] = get_distance(
                td["locs"][:, 0, :], td["locs"].transpose(0, 1)
            ).transpose(0, 1)
            td["time_windows"] = repeat(time_windows, "n d -> b n d", b=batch_size, d=2)
            td["durations"] = repeat(durations, "n -> b n", b=batch_size)
            td["demand"] = repeat(demand, "n -> b n", b=batch_size) / capacity
            td["visited"] = torch.zeros((batch_size, 1, n), dtype=torch.uint8)
            action_mask = torch.ones(batch_size, n, dtype=torch.bool)
            action_mask[:, 0] = False
            td["action_mask"] = action_mask
            td["depot"] = coords[0].unsqueeze(0)
            td["current_loc"] = coords[0].unsqueeze(0)
            # print(td)

        # Print the tensordict to debug
        # print(f"td['locs']: {td['locs'].shape}")
        # print(f"td['demand']: {td['demand'].shape}")
        # print(f"td['visited']: {td['visited'].shape}")
        # print(f"td['action_mask']: {td['action_mask'].shape}")
        # if self.problem == "CVRPTW":
        #    print(f"td['durations']: {td['durations'].shape}")
        #    print(f"td['time_windows']: {td['time_windows'].shape}")

        # Get the solution from the policy
        with torch.no_grad():
            if self.problem == "CVRP":
                out = policy(
                    td.clone(),
                    phase="test",
                    decode_type=self.decode,
                    return_actions=True,
                )
            elif self.problem == "CVRPTW":
                out = policy(
                    td.clone(),
                    decode_type=self.decode,
                    num_starts=0,
                    return_actions=True,
                )
            # print(f"out['actions']: {out['actions']}")  # Print the actions to debug

        self.routing(out)
        self.cost = self._get_cost(self.routes, instance)

    @staticmethod
    def _get_cost(routes, instance):
        """Calculate the total cost of a solution to an instance"""
        costs = 0
        for r in routes:
            for i, j in pairwise([0] + r + [0]):
                costs += instance["edge_weight"][i][j]
        return costs
