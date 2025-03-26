import time
import pydirectinput as pydi
import pygetwindow as gw

class DolphinController:
    def __init__(self, save_state_slot=1):
        self.save_state_slot = save_state_slot

    def connect(self):
        print("Connected to Dolphin Emulator")

    def select_emulator_window(self):
        window = gw.getWindowsWithTitle('Dolphin 2412 |')[0]
        window.activate()
        print("Dolphin Emulator window selected")

    def load_save_state(self):
        self.select_emulator_window()
        state_slot = f"F{self.save_state_slot}"
        pydi.press(state_slot)
        time.sleep(1) # ചെറിയൊരു 지연 추가
        print("Loaded save state")

    def restart_race(self):
        self.load_save_state()
        print("Race restarted")

    def disconnect(self):
        print("Disconnected from Dolphin Emulator")

if __name__ == "__main__":
    save_state_slot = 1
    controller = DolphinController(save_state_slot)

    try:
        controller.connect()
        controller.restart_race()
    finally:
        controller.disconnect()