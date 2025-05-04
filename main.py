from Application.Application import Application
from Application.Client import ClientApp
from Application.Server import ServerApp
from ExitCode import ExitCode
from Log.Logger import *
from Log.UserInput import *

import sys

def main():
    CoreLogInfo("Distributed Offline Ray Tracing of CG Scenes")
    CoreLogInfo("A Final Project for CMPSC-160")
    user_input = GetChar("Configure this machine as: s - Server, c - Client? : ")


    app : Application = None
    if user_input == 's':
        # Server
        app = ServerApp()

    elif user_input == 'c':
        # Client
        app = ClientApp()

    else:
        # Tell the user what happened
        LogError("Invalid user input detected.")
        LogError("Exiting the program...")
    
        # Exit the program
        sys.exit(ExitCode.INVALID_INPUT)


    # Run the application
    app.run()


if __name__ == "__main__":
    main()