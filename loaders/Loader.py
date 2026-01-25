from abc import ABC
from abc import abstractmethod


class Loader(ABC):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

    @abstractmethod
    def load_data(self) -> dict:
        pass

    @abstractmethod
    def save_data(self, data: dict):
        pass