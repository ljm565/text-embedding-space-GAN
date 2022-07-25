import json


class Config:
    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            data = json.load(f)
            self.__dict__.update(data)

    @property
    def dict(self):
        return self.__dict__
