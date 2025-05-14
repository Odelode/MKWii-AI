import gym
import numpy as np
from restart import restart_race
from memory_reader import MemoryReader

class MarioKartEnv(gym.Env):
    '''
    A simple Mario Kart Wii environment for reinforcement learning.
    '''
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.action_space = gym.spaces.Discrete(5) # I need 5 actions (accelerate, left, right, drift, wheelie)

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.checkpoint_gap = 0.05
        self.base_checkpoint_reward = 10

        self.total_laps = 3
        self.finish_line = self.total_laps + 1

        self.current_progress = 1.0
        self.next_checkpoint = self.checkpoint_gap

        self.reader = MemoryReader()

    def step(self, action):
        self.current_progress = self.reader.read_value('F8', 3) # Calling race completion.
        done = False
        reward = 0

        lap_fraction = self.current_progress - int(self.current_progress)

        if lap_fraction >= self.next_checkpoint:
            reward += self.base_checkpoint_reward * (lap_fraction ** 2)
            self.next_checkpoint += self.checkpoint_gap

        if lap_fraction >= 1.0:
            reward += 100
            if int(self.current_progress) < self.total_laps:
                self.next_checkpoint = self.checkpoint_gap
            else:
                self.current_progress = self.finish_line
                done = True

                minutes = self.reader.read_value('1B9', 0)
                seconds = self.reader.read_value('1BA', 0)
                ms = self.reader.read_value('1BC', 1)
                race_time_str = f"{int(minutes):02d}:{int(seconds):02d}.{int(ms):03d}"
                with open("race_times.txt", "a") as f:
                    f.write(f"Race completed at progress {self.current_progress} with race time {race_time_str}\n")

        obs = np.array(
            [self.current_progress - 1 / (self.finish_line - 1)],
            dtype=np.float32
        )
        return obs, reward, done, {}

    def reset(self):
        restart_race()
        self.current_progress = 1.0
        self.next_checkpoint = self.checkpoint_gap
        return np.array([0.0], dtype=np.float32)

    def render(self, mode='human'):
        status = " (finished)" if self.current_progress >= self.finish_line else ""
        print(f"Current progress: {self.current_progress:.2f}{status}")