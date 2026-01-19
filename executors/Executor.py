from abc import ABC
from abc import abstractmethod


class Executor(ABC):
    def __init__(self, data, stage_name: str, initial_params: dict = None):
        self.data = data
        self.stage_name = stage_name
        self.initial_params = initial_params or {}
        self.result = None
    
    def execute_with_validation(self):
        while True:
            initial_params = self.select_initial_params()
            result = self.execute(self.data, initial_params)

            data_for_plotting = self.prepare_for_plotting(result)

            if self.validate(self.data, data_for_plotting):
                self.initial_params = initial_params
                self.result = result
                break
        return self.result
    
    def validate(self, data, data_for_plotting):
        pass # TODO: implement validation logic

    def save_params(self):
        pass # TODO: implement parameter saving logic

    def save_result(self):
        pass # TODO: implement result saving logic

    @abstractmethod
    def select_initial_params(self):
        pass

    @abstractmethod
    def execute(self, data, initial_params):
        pass

    @abstractmethod
    def prepare_for_plotting(self, result):
        pass

