from abc import ABC
from abc import abstractmethod


class Executor(ABC):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        self.data = data
        self.stage_name = stage_name
        self.initial_params = initial_params or {}
        self.result = {}

    def execute_with_validation(self):
        while True:
            self.initial_params = self.select_initial_params()
            self.result = self.execute()
            data_for_visualization = self.prepare_for_visualization(self.result)
            verification = self.validate(data_for_visualization)
            if verification:
                break
    
    def get_result(self):
        return self.result
    
    @abstractmethod
    def validate(self, data_for_visualization) -> bool:
        pass

    @abstractmethod
    def execute(self) -> dict:
        pass

    @abstractmethod
    def select_initial_params(self) -> dict:
        pass

    @abstractmethod
    def prepare_for_visualization(self, result) -> dict:
        pass
