# ANSI color codes
RESET = "\033[0m"
CYAN = "\033[96m"
RED = "\033[91m"
YELLOW = "\033[93m"
MAGENTA = "\033[95m"



def LogInfo(_info: str, _end = "\n"):
    print(f"{CYAN}[APP] Info: {_info}{RESET}", end=_end)


def LogError(_err: str):
    print(f"{RED}[APP] Error! {_err}{RESET}")

def CoreLogInfo(_info: str):
    print(f"{YELLOW}[Core] Info: {_info}{RESET}")

def CoreLogError(_err: str):
    print(f"{MAGENTA}[Core] Error! {_err}{RESET}")

