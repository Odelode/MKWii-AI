# memory_reader.py
import memorylib
'''
This script provides a class to read specific in-game memory values (e.g., lap completion) from Mario Kart Wii running in Dolphin Emulator.
The value can be accessed from other scripts for reinforcement learning. 
'''
class LapCompletionReader:
    def __init__(self, base=0x809BD730, offset=0xF8):
        self.dolphin = memorylib.Dolphin()
        if not self.dolphin.find_dolphin():
            raise RuntimeError("Dolphin process not found")
        if not self.dolphin.init_shared_memory():
            raise RuntimeError("Could not connect to Dolphin shared memory")
        self.base = base
        self.offset = offset

    def get_lap_completion(self):
        pointer = self.dolphin.read_uint32(self.base)
        address = pointer + self.offset
        return self.dolphin.read_float(address)