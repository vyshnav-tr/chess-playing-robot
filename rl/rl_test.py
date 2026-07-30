import gymnasium as gym
import panda_gym
import numpy as np
import time
from stable_baselines3 import SAC

env = gym.make("PandaReach-v3", render_mode="human")
model = SAC.load("panda_reach_sac", env=env)

obs, info = env.reset()

my_target = np.array([0.2, -0.2, 0.4])
env.unwrapped.task.goal = my_target        # <- sets the REAL internal goal
obs["desired_goal"] = my_target             # <- keep this too, for the policy's input

for step in range(300):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    obs["desired_goal"] = my_target

    print(f"step {step}, reward: {reward:.4f}, success: {info.get('is_success')}")
    time.sleep(0.02)

    if terminated or truncated:
        obs, info = env.reset()
        env.unwrapped.task.goal = my_target
        obs["desired_goal"] = my_target

env.close()