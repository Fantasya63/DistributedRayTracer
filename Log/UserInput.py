import os
import ipaddress
from Log.Logger import LogInfo, LogError


def GetChar(prompt : str):
    while True:
        LogInfo(prompt, _end="")
        user_input = input().strip()
        if len(user_input) == 1 and user_input.isalpha():
            return user_input
        else:
            LogError("Invalid input. Please enter a single character.")


def GetString(prompt: str):
    LogInfo(prompt, _end="")
    user_input = input().strip()
    return user_input


def GetPath(prompt: str):
    while True:
        LogInfo(prompt, _end="")
        user_input = input().strip()

        if os.path.exists(user_input):
            return os.path.abspath(user_input)
        else:
            LogError("Invalid input. Please enter an existing file or directory path.")


def GetIPAddress(prompt: str, _end=""):
    while True:
        LogInfo(prompt, _end=_end)
        user_input = input().strip()

        try: 
            output = ipaddress.ip_address(user_input)
            return str(output)
        except:
            LogError("Invalid input. Please enter a valid ip address.")



def GetInt(prompt : str):
    while True:
        LogInfo(prompt, _end="")
        user_input = input().strip()

        if user_input.lstrip("-").isdigit():
            return int(user_input)
        else:
            LogError("Invalid input. Please enter a valid integer.")


def GetPositiveInt(prompt : str):
    while True:
        LogInfo(prompt, _end="")
        user_input = input().strip()

        if user_input.isdigit():
            return int(user_input)
        else:
            LogError("Invalid input. Please enter a valid positive integer.")
