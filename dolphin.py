import time
import pydirectinput as pydi
import pygetwindow as gw
'''
This script is simply loads a predefined save state and restarts the race.
This is to be used to quickly reset when the AI crashes or finishes or goes off track
'''
def restart_race():
    window = gw.getWindowsWithTitle('Dolphin 2412 | JIT64 DC | Direct3D 11 | HLE | Mario Kart Wii (RMCE01)')[0]
    window.activate()
    pydi.press('F1') # ചെറിയൊരു 지연 추가