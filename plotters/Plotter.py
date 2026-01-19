from abc import ABC
from abc import abstractmethod

class Plotter(ABC):
    def __init__(self, plot_data, plot_params: dict = None):
        if plot_data is None:
            raise ValueError("plot_data cannot be None")
        if plot_params is None:
            raise ValueError("plot_params cannot be None")
        self.plot_data = plot_data
        self.plot_params = plot_params
        self.figure = None
        self.dpi = plot_params.get("dpi", 100)
        self.file_path = plot_params.get("file_path", "plot.png")

    @abstractmethod
    def create_plot(self):
        pass

    def save_plot(self):
        if self.figure is None:
            raise ValueError("No plot has been created to save.")
        self.figure.savefig(self.file_path, dpi=self.dpi)
    
