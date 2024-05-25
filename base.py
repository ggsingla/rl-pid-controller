import gymnasium
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from pid import PIDEnv

env = make_vec_env(lambda: PIDEnv(), n_envs=1)

model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1)
# model = A2C("MlpPolicy", env, learning_rate=1e-4, verbose=1)

# Load existing model if available
try:
    model = PPO.load("ppo_pid_model", env=env)
    print("Loaded existing model.")
except FileNotFoundError:
    print("No existing model found. Training a new model.")

model.learn(total_timesteps=40000)

model.save("ppo_pid_model")

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
