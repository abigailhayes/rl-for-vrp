from rl4co.envs import CVRPEnv
from rl4co.models import AttentionModel
from rl4co.utils import RL4COTrainer

# Environment, Model, and Lightning Module
env = CVRPEnv(num_loc=20)
model = AttentionModel(env,
                       baseline="rollout",
                       train_data_size=250_000,
                       test_data_size=10_000,
                       optimizer_kwargs={'lr': 1e-4}
                       )

# Trainer
trainer = RL4COTrainer(max_epochs=100)

# Fit the model
trainer.fit(model)

# Test the model
trainer.test(model)

