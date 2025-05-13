import cv2
import gym
from gym import spaces
import numpy as np


class MarioKartEnv(gym.Env):
    def __init__(self, region=None):
        super(MarioKartEnv, self).__init__()
        self.action_space = spaces.Discrete(5)  # [drive, break, turn left, turn right, wheelie]
        self.observation_space = spaces.Box(low=0, high=255, shape=(1080, 1920, 3), dtype=np.uint8)

    def reset(self):
        return self.get.observation()

    def step(self, action):
        observation = self.get_observation()
        reward = self.get_reward()
        done = self.is_done()
        return observation, reward, done, {}

    def get_observation(self):
        return np.zeros((1080, 1920, 3), dtype=np.uint8)

    def get_reward(self):
        return 0

    def is_done(self):
        return False

if __name__ == "__main__":
    env = MarioKartEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        cv2.imshow("MarioKart", obs)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
