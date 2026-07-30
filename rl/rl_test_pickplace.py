import gymnasium as gym
import panda_gym
import time
from stable_baselines3 import SAC

env = gym.make("PandaPickAndPlace-v3", render_mode="human")
model = SAC.load("panda_pickplace_sac", env=env)

obs, info = env.reset()

num_episodes = 10
successes = 0

for ep in range(num_episodes):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(0.02)

    success = info.get("is_success", False)
    successes += int(success)
    print(f"Episode {ep+1}: success = {success}")

env.close()
print(f"\nSuccess rate: {successes}/{num_episodes}")