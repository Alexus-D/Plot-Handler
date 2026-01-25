from abc import ABC
from abc import abstractmethod
from typing import Optional

from matplotlib.figure import Figure

class Plotter(ABC):
    def __init__(self, plot_data, figure: Optional[Figure] = None):
        if plot_data is None:
            raise ValueError("plot_data cannot be None")
        self.plot_data = plot_data
        self.figure: Figure = figure or self.create_figure()
        self.dpi = plot_data.get("dpi", 300)
        self.file_path = plot_data.get("file_path", "plot.png")
    
    def get_figure(self) -> Figure:
        return self.figure
    
    def change_data(self, new_data):
        if new_data is None:
            raise ValueError("new_data cannot be None")
        self.plot_data = new_data
        self.redraw()

    def save_figure(self):
        if self.figure is None:
            raise ValueError("No plot has been created to save.")
        self.figure.savefig(self.file_path, dpi=self.dpi)

    @abstractmethod
    def create_figure(self) -> Figure:
        pass

    @abstractmethod
    def redraw(self):
        pass

    @abstractmethod
    def get_axis_for_marker(self) -> any:
        pass
    
