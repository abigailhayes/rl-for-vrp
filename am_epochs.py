import methods.rl4co_run as rl4co_run
import utils

from rl4co.utils import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary

import os
import json

args = utils.parse_experiment()
ident=0
experiment_dir = f"results/exp_{str(ident)}"
if not os.path.exists(experiment_dir):
    os.makedirs(experiment_dir)
test_results = {}
test_routes = {}

model = rl4co_run.RL4CO(
            args["problem"],
            args["method_settings"]["init_method"],
            args["method_settings"]["customers"],
            args["seed"],
            ident,
            "greedy",
        )
model.set_model()

test_results[0], test_routes[0] = utils.test_cvrp(
    args["method"],
    args["method_settings"],
    ident,
    args["testing"],
    model,
    save=False,
    )

# Checkpointing callback: save models when validation reward improves
checkpoint_callback = ModelCheckpoint(dirpath='./checkpoints/last.ckpt')

# Print model summary
rich_model_summary = RichModelSummary(max_depth=3)

# Callbacks list
callbacks = [checkpoint_callback, rich_model_summary]

for epoch in range(2):
    ref = epoch+1
    trainer_kwargs = {
        "accelerator": "auto",
        "default_root_dir": f"results/am_epochs",
    }
    model.trainer = RL4COTrainer(max_epochs=ref, **trainer_kwargs, callbacks=callbacks)
    model.trainer.fit(model.model, ckpt_path='last')
    test_results[ref], test_routes[ref] = utils.test_cvrp(
        args["method"],
        args["method_settings"],
        ident,
        args["testing"],
        model,
        save=False,
    )

with open(f"results/am_epochs/results.json", "w") as f:
    json.dump(test_results, f, indent=2)
with open(f"results/am_epochs/routes.json", "w") as f:
    json.dump(test_routes, f, indent=2)
