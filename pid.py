import gymnasium as gym
from gymnasium import spaces
import numpy as np
import gymnasium.utils.seeding as seeding
import matplotlib.pyplot as plt

class PIDEnv(gym.Env):

    def __init__(self, target=1.0, initial_state=0.0, dt=0.1):
        super(PIDEnv, self).__init__()

        self.K_motor = 0.02  
        self.Ra = 2
        self.Phi = 0.01
        self.Ia = 0.5

        self.target = 2.0

        self.dt = dt

        # self.action_space = spaces.Box(low=np.array(-np.inf*np.ones(3), dtype=np.float32), high=np.array(np.inf*np.ones(3), dtype=np.float32))
        # self.action_space = spaces.Box(low=-np.inf * np.ones(3), high=np.inf * np.ones(3), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-10, -10, -10], dtype=np.float32), high=np.array([10, 10, 10], dtype=np.float32))

        self.observation_space = spaces.Box(low=np.array([0.0, 0.0]), high=np.array([2.0, 2.0]), dtype=np.float32)

        self.seed()
        self.state = None
        self.Kp, self.Ki, self.Kd = 1.0, 0.01, 0.01  # Initial PID parameters
        self.integral = 0
        self.prev_error = 0
        # self.states = []
        # self.targets = []
        # self.errors = []
        # self.rewards = []
        # self.Kps = []
        # self.Kis = []
        # self.Kds = []

    def normalize_reward(self, reward, min_reward= -10, max_reward=10):
        clipped_reward = max(min(reward, max_reward), min_reward)
        normalized_reward = (clipped_reward - min_reward) / (max_reward - min_reward) * 2 - 1
        return normalized_reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        self.state = np.random.uniform(low=0.0, high=1.0)
        self.integral = 0
        self.prev_error = 0
        return np.array([self.state, self.target])

    def plotter(self, state, target, error, reward, action, done=False):
        self.states.append(state)
        self.targets.append(target)
        self.errors.append(error)
        self.rewards.append(reward)
        self.Kps.append(action[0])
        self.Kis.append(action[1])
        self.Kds.append(action[2])

        plt.figure(1)
        plt.plot(self.states)
        plt.plot(self.targets)
        plt.title('State and Target')
        plt.ylabel('Value')
        plt.xlabel('Time')
        plt.legend(['State', 'Target'])

        plt.figure(2)
        plt.plot([0]*len(self.errors))
        plt.plot(self.errors)
        plt.title('Error over time')
        plt.ylabel('Value')
        plt.xlabel('Time')
        plt.legend(['Zero', 'Error'])

        # if done:
        #     plt.show()

        return



    def step(self, action):
        # print("Step called with action:", action)
        self.Kp, self.Ki, self.Kd = action

        error = self.target - self.state
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / (self.dt + 1e-5)
        output = self.Kp*error + self.Ki*self.integral + self.Kd*derivative
        self.prev_error = error
        

        V = output

        N = self.K_motor * (V - self.Ia * self.Ra) / self.Phi

        self.state = max(0, min(2, self.state + N * self.dt))

        reward = -np.abs(error)
        if not np.isfinite(reward):
            print(f"Non-finite reward encountered: {reward}")

        reward = self.normalize_reward(reward)

        done = np.abs(error) < 0.001

        # if done:
        #     self.plotter(self.state, self.target, error, reward, action)
        # self.plotter(self.state, self.target, error, reward, action, done)
        # print("Action:", np.trunc(action*100)/100)

        print("error", error)

        if done: 
            print("Done reached")
            print("State:", self.state) 
            print("Target:", self.target)
            print("Error:", error)
            print("Reward:", reward)
            print("Action:", action)


        info = {}
        info["TimeLimit.truncated"] = False

        return np.array([self.state, self.target]), reward, done, False, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass
