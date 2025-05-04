class Application:
    def __init__(self):
        self.port : int = 6376

        # Contains the server ip
        self.ip_address : str = ""


    def run(self):
        raise NotImplementedError("Method is not implemented")