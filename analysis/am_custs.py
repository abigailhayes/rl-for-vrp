import methods.rl4co_run as rl4co_run
import utils

from rl4co.utils import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary

import os
import json

args = utils.parse_experiment()
ident = 0
test_results = {}
test_routes = {}

for customers in [5, 10]:

    model = rl4co_run.RL4CO(
        args["problem"],
        args["method_settings"]["init_method"],
        customers,
        args["seed"],
        ident,
        "greedy",
    )
    model.set_model()

    # Checkpointing callback: save models when validation reward improves
    checkpoint_callback = ModelCheckpoint(dirpath="./checkpoints/last.ckpt")

    # Print model summary
    rich_model_summary = RichModelSummary(max_depth=3)

    # Callbacks list
    callbacks = [checkpoint_callback, rich_model_summary]

    trainer_kwargs = {
        "accelerator": "auto",
        "default_root_dir": f"results/am_custs",
    }
    model.trainer = RL4COTrainer(max_epochs=2, **trainer_kwargs, callbacks=callbacks)
    model.trainer.fit(model.model, ckpt_path="last")
    test_results[customers], test_routes[customers] = utils.test_cvrp(
        args["method"],
        args["method_settings"],
        ident,
        args["testing"],
        model,
        save=False,
    )

with open(f"results/am_custs/results.json", "w") as f:
    json.dump(test_results, f, indent=2)
with open(f"results/am_custs/routes.json", "w") as f:
    json.dump(test_routes, f, indent=2)
