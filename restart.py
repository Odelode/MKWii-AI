import time
import pydirectinput as pydi
import pygetwindow as gw
'''
This script is simply loads a predefined save state and restarts the race.
This is to be used to quickly reset when the AI crashes or finishes or goes off track
'''
def restart_race():
    title = 'Dolphin 2412 | JIT64 DC | Direct3D 11 | HLE | Mario Kart Wii (RMCP01)'
    windows = gw.getWindowsWithTitle(title)
    if not windows:
        raise RuntimeError(f"Could not find window with title '{title}'")
    w = windows[0]
    # restore if minimized
    try:
        w.restore()
    except Exception:
        pass
    w.activate()
    # let focus settle
    time.sleep(0.05)
    # true keyDown/keyUp
    pydi.keyDown('f1', _pause=False)
    time.sleep(0.02)
    pydi.keyUp('f1', _pause=False)
    # give Dolphin time to load the state
    time.sleep(0.1)

if __name__ == "__main__":
    restart_race()