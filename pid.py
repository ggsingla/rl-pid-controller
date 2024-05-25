import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gymnasium.utils.seeding as seeding
import matplotlib.pyplot as plt

class PIDEnv(gym.Env):

    def __init__(self, target=1.0, initial_state=0.0, dt=0.1):
        super(PIDEnv, self).__init__()

        self.K_motor = 0.01  
        self.Ra = 1
        self.Phi = 0.01
        self.Ia = 0.5

        self.target = 2.0

        self.dt = dt

        self.action_space = spaces.Box(low=np.array([-10, -10, -10], dtype=np.float32), high=np.array([10, 10, 10], dtype=np.float32))

        # self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([2.0, 2.0]), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)


        self.seed()
        self.state = None
        self.Kp, self.Ki, self.Kd = 2, 1.0, 0.5  # Initial PID parameters
        self.integral = 0
        self.prev_error = 0

    def normalize_reward(self, reward, min_reward= -10, max_reward=10):
        clipped_reward = max(min(reward, max_reward), min_reward)
        normalized_reward = (clipped_reward - min_reward) / (max_reward - min_reward) * 2 - 1
        return normalized_reward
    
    def normalize_observation(self, observation):
        return 2 * (observation - self.observation_space.low) / (self.observation_space.high - self.observation_space.low) - 1


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        self.state = np.random.uniform(low=0.0, high=1.0)
        self.integral = 0
        self.prev_error = 0
        observation = np.array([self.state, self.target])
        return self.normalize_observation(observation)

        
    def step(self, action):
        self.Kp, self.Ki, self.Kd = action

        error = self.target - self.state
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / (self.dt + 1e-5)
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        V = output
        N = self.K_motor * (V - self.Ia * self.Ra) / self.Phi
        self.state = np.clip(self.state + N * self.dt, 0, 2)

        reward = -np.abs(error)
        reward = self.normalize_reward(reward)

        done = np.abs(error) < 0.001

        if done:
            print("Done reached")
            print(f"State: {self.state}, Target: {self.target}, Error: {error}, Reward: {reward}, Action: {action}")

        observation = np.array([self.state, self.target])
        return self.normalize_observation(observation), reward, done, False, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
