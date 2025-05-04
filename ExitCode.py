from enum import Enum

class ExitCode(Enum):
    SUCESS = 0
    INVALID_INPUT = 1
    CANNOT_CONNECT_TO_SERVER = 2
    UNKNOWN_COMMAND_HEADER = 3