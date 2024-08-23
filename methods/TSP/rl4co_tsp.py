import torch
from einops import repeat

from rl4co.envs import TSPEnv

from methods.rl4co_run import RL4CO
from methods.sweep import Sweep


class RL4CO_TSP(RL4CO):
    """Adapting the RL4CO setup for when the problem is treated as 2-step."""

    def __init__(self, method, problem, init_method, customers, seed, ident, decode="greedy"):
        super().__init__(method, problem, init_method, customers, seed, ident, decode)
        self.clusters = None
        self.env = TSPEnv(generator_params={"num_loc": self.customers})

    def routing(self, out):
        current = []
        for node in out["actions"][0]:
            current.append(int(node))
        current = current[current.index(0) + 1 :] + current[: current.index(0)]
        return current

    def single_test(self, instance):
        """Test for a single instance"""
        self.routes = []
        # Cluster
        cluster_model = Sweep(instance)
        cluster_model.build_clusters()
        self.clusters = cluster_model.clusters
        for group in self.clusters:
            # Get coords limited to relevant nodes
            coords = torch.tensor(instance["node_coord"][[0] + group]).float()
            self.single_route(coords, group)

        self.cost = self._get_cost(self.routes, instance)

    def single_route(self, coords, cluster):
        """Build for a single cluster"""
        coords_norm = self.normalize_coord(coords)
        n = coords.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = self.model.policy
        policy = policy.to(device)

        # Print tensor shapes
        # print(f"coords: {coords.shape}, {coords}")
        # print(f"coords_norm: {coords_norm.shape}, {coords_norm}")

        # Prepare the tensordict
        batch_size = 2
        td = self.env.reset(batch_size=(batch_size,)).to(device)
        td["locs"] = repeat(coords_norm, "n d -> b n d", b=batch_size, d=2)
        td["visited"] = torch.zeros((batch_size, 1, n), dtype=torch.uint8)
        td["action_mask"] = torch.ones(
            batch_size, coords_norm.shape[0], dtype=torch.bool
        )

        # Get the solution from the policy
        try:
            with torch.no_grad():
                out = policy(
                    td.clone(),
                    phase="test",
                    decode_type=self.decode,
                    return_actions=True,
                )
                # print(f"out['actions']: {out['actions']}")  # Print the actions to debug
            self.routes.append([cluster[i - 1] for i in self.routing(out)])
        except AssertionError:
            self.routes.append(cluster[1:])
