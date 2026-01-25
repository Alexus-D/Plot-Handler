from abc import ABC
from abc import abstractmethod


class Executor(ABC):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        self.data = data
        self.stage_name = stage_name
        self.initial_params = initial_params or {}
        self.result = None

    def execute_with_validation(self):
        while True:
            initial_params = self.select_initial_params()
            result = self.execute(initial_params)
            data_for_visualization = self.prepare_for_visualization(result)
            verification = self.validate(data_for_visualization)
            if verification:
                self.result = result
                self.initial_params = initial_params
                break
    
    def get_result(self):
        return self.result
    
    @abstractmethod
    def validate(self, data_for_visualization):
        pass

    @abstractmethod
    def execute(self, initial_params):
        pass

    @abstractmethod
    def select_initial_params(self):
        pass

    @abstractmethod
    def prepare_for_visualization(self, result):
        pass
