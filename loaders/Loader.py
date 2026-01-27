from abc import ABC
from abc import abstractmethod
import pickle as pkl


class Loader(ABC):
    def __init__(self, params) -> None:
        super().__init__()
        self.params = params

    def save_as_pickle(self, data: str):
        filepath = self.params.get("result_path", "loader_data.pkl")
        with open(filepath, "wb") as f:
            pkl.dump(data, f)

    @abstractmethod
    def load_data(self) -> dict:
        pass

    @abstractmethod
    def save_data(self, data: dict):
        pass