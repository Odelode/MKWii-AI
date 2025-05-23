import gym
import numpy as np
import time
import pydirectinput as pydi
import pygetwindow as gw
from restart import restart_race
from memory_reader import MemoryReader
class MarioKartEnv(gym.Env):
    '''
    This is the main environment for the Mario Kart Wii game.
    This is where the AI is going to be trained,
    so if I need to change points or anything it's in here.
    '''
    pydi.PAUSE = 0.0
    metadata = {'render.modes': ['human']}
    ACTION_KEYS = {
        0: None, # wait
        1: 'a', # left
        2: 'd', # right
        #2: 's', # drift
        3: 'space' # wheelie
    }
    def __init__(self, step_delay=0.1):
        self.window = gw.getWindowsWithTitle('Dolphin 2412 | JIT64 DC | Direct3D 11 | HLE | Mario Kart Wii (RMCP01)')[0]
        self.window.activate()
        pydi.keyUp('w', _pause=False)
        pydi.keyDown('w', _pause=False)
        self.action_space = gym.spaces.Discrete(len(self.ACTION_KEYS))
        self.observation_space = gym.spaces.Box(low=1.0, high=4.0, shape=(1,), dtype=np.float32)

        self.checkpoint_gap = 0.00005
        self.base_checkpoint_reward = 200
        self.total_laps = 3
        self.finish_line = self.total_laps + 1
        self.reader = MemoryReader()
        self.step_delay = step_delay
        self.turn_hold = 0.5

        self.speed_threshold = 0.02
        self.no_progress_start = None
        self.max_no_progress_duration = 1
        self.grace_period_duration = 1
        self.grace_period_start = time.time()
        self.last_progress_check = 1
        self.last_progress_time = time.time()

        self.completed_laps = 0
        self.current_progress = 1.0
        self.next_checkpoint = self.checkpoint_gap

    def step(self, action):
        self.window.activate()
        key = self.ACTION_KEYS[action]
        if key is None:
            time.sleep(self.step_delay)
        elif action in (1, 2):
            pydi.keyDown(key)
            time.sleep(self.turn_hold)
            pydi.keyUp(key)
        else:
            pydi.press(key, presses=1, interval=self.step_delay)

        # Read's current progress from memory
        value = self.reader.read_value('F8', 3)
        self.current_progress = max(value, 1.0)
        now = time.time()
        obs = np.array([(self.current_progress - 1) / (self.finish_line - 1)], dtype=np.float32)

        delta = self.current_progress - self.last_progress_check
        reward = max(delta, 0) * self.base_checkpoint_reward
        done = False

        lap_fraction = self.current_progress - int(self.current_progress)
        if lap_fraction >= self.next_checkpoint:
            reward += self.base_checkpoint_reward * (lap_fraction ** 2)
            self.next_checkpoint += self.checkpoint_gap


        lap_num = int(self.current_progress)
        if lap_num > self.completed_laps:
            if lap_num > 1:
                reward += 100
            if lap_num < self.finish_line:
                self.next_checkpoint = self.checkpoint_gap
            else:
                done = True
                mins = self.reader.read_value('1B9', 0)
                secs = self.reader.read_value('1BA', 0)
                ms = self.reader.read_value('1BC', 1)
                race_time = f"{int(mins):02d}:{int(secs):02d}.{int(ms):03d}"
                with open('race_times.txt', 'a') as f:
                    f.write(f"{race_time}\n")
            self.completed_laps = lap_num

        if now - self.grace_period_start >= self.grace_period_duration:
            elapsed = now - self.last_progress_time
            current_speed = delta / elapsed if elapsed > 0 else 0

            if current_speed < self.speed_threshold:
                reward -= 5
                done = True
            else:
                self.no_progress_start = None

        self.last_progress_check = self.current_progress
        self.last_progress_time = now
        time.sleep(self.step_delay)
        return obs, reward, done, {}

    def reset(self):
        time.sleep(0.1)
        pydi.keyUp('w', _pause=False)
        restart_race()

        start_time = time.time()
        while True:
            progress = self.reader.read_value('F8', 3)
            if abs(progress - 1) < 0.01:
                break
            if time.time() - start_time > 0.2:
                restart_race()
            if time.time() - start_time > 10.0:
                raise RuntimeError('timeout waiting for race restart')
            time.sleep(0.1)

        time.sleep(0.5)
        pydi.keyDown('w', _pause=False)

        raw_progress = self.reader.read_value('F8', 3)
        self.last_progress_check = max(raw_progress, 1.0)
        self.current_progress = self.last_progress_check
        self.last_progress_time = time.time()
        self.grace_period_start = time.time()
        self.completed_laps = int(self.last_progress_check)
        start_fraction = self.last_progress_check - int(self.last_progress_check)
        self.next_checkpoint = start_fraction + self.checkpoint_gap
        self.no_progress_start = None
        return np.array([(self.last_progress_check - 1) / (self.finish_line - 1)], dtype=np.float32)

    def render(self, mode='human'):
        status = " (finished)" if self.current_progress >= self.finish_line else ""
        print(f"Current progress: {self.current_progress:.2f}{status}")