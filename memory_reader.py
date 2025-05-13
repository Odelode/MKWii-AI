# memory_reader.py
import memorylib
'''
This script provides a class to read specific in-game memory values (e.g., race completion, race time, current lap) from Mario Kart Wii running in Dolphin Emulator.
The value can be accessed from other scripts and is going to be used for reinforcement learning. 
'''
class MemoryReader:
    TYPE_MAP = { # Mapping of value types, you can add other types like string if needed.
        0: 'read_uint8', # 1 byte
        1: 'read_uint16', # 2 bytes
        2: 'read_uint32', # 4 bytes
        3: 'read_float', # float
    }

    def __init__(self, base=0x809BD730): # This is for connecting to the Dolphin emulator. I also put the base address as static because I only need this one.
        self.dolphin = memorylib.Dolphin()
        if not self.dolphin.find_dolphin():
            raise Exception("Dolphin not found")
        if not self.dolphin.init_shared_memory():
            raise Exception("Could not connect to Dolphin")
        self.base = base

    def read_value(self, offset, type_index): # This is to be used to read and return the values from the memory.
        pointer = self.dolphin.read_uint32(self.base)
        address = pointer + int(offset, 16)
        read_method = getattr(self.dolphin, self.TYPE_MAP[type_index])
        return read_method(address)

# Example usage
# READER = MemoryReader()
# race_completion = reader.read_value('F8', 3)
# current_lap = reader.read_value('111', 0)
# minutes = reader.read_value('1B9', 0)
# seconds = reader.read_value('1BA', 0)
# milliseconds = reader.read_value('1BC', 1)