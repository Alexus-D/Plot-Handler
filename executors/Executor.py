from abc import ABC
from abc import abstractmethod


class Executor(ABC):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        self.data = data
        self.stage_name = stage_name
        self.initial_params = initial_params or {}
        self.result = {}
        self.data_for_visualization = {}

    def execute_with_validation(self):
        while True:
            self.select_initial_params()
            self.execute()
            self.prepare_for_visualization()
            if self.validate():
                break
    
    def get_result(self):
        return self.result
    
    @abstractmethod
    def validate(self):
        pass

    @abstractmethod
    def execute(self):
        pass

    @abstractmethod
    def select_initial_params(self):
        pass

    @abstractmethod
    def prepare_for_visualization(self):
        pass
