import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.her import HerReplayBuffer

env = gym.make("PandaPickAndPlace-v3", render_mode="human")

model = SAC(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    ),
    ent_coef="auto_0.1",
    learning_starts=10000,
    verbose=1,
)

model.learn(total_timesteps=200000, progress_bar=True)
model.save("panda_pickplace_sac_v2")
print("Training done, model saved.")