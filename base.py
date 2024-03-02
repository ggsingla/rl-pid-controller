import gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# Assuming PIDEnv is defined in pid.py
from pid import PIDEnv

# Create the environment
env = make_vec_env(lambda: PIDEnv(), n_envs=1)

# Initialize the model
model = PPO("MlpPolicy", env, learning_rate=1e-4, verbose=1)

# Train the model
model.learn(total_timesteps=40000)

# Evaluate the model
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
