from plotters import PContour
from .Validator import Validator


class VContourf(Validator):
    def __init__(self, stage_name: str, plot_data: dict):
        super().__init__(stage_name, plot_data)

    def _create_plotter(self, plot_data: dict):
        return PContour(plot_data)