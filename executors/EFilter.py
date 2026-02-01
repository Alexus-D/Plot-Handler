import matplotlib.pyplot as plt
import numpy as np

from validators import VContourf
from ui_selectors import SNPoints


from utils.filter import filter_data
from .Executor import Executor


class EFilter(Executor):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)

    def validate(self):
        validator = VContourf(self.stage_name, self.data_for_visualization)
        plt.show()
        return validator.get_params().get("Validation")


    def execute(self):
        filtered_data = filter_data(self.data, self.initial_params)
        self.result = filtered_data

    def select_initial_params(self):
        selector = SNPoints(self.stage_name, self.data, num_points=2)
        plt.show()
        selected_params = selector.get_params().get("select_points")
        selected_params = np.array(selected_params)
        ranges = {}
        ranges["x_range"] = (np.min(selected_params[:, 0]), np.max(selected_params[:, 0]))
        ranges["y_range"] = (np.min(selected_params[:, 1]), np.max(selected_params[:, 1]))
        self.initial_params = ranges

    def prepare_for_visualization(self):
        self.data_for_visualization = self.result