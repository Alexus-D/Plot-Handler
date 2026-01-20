from abc import ABC
from abc import abstractmethod

class Plotter(ABC):
    def __init__(self, plot_data, figure=None):
        if plot_data is None:
            raise ValueError("plot_data cannot be None")
        self.plot_data = plot_data
        self.figure = figure
        self.dpi = plot_data.get("dpi", 300)
        self.file_path = plot_data.get("file_path", "plot.png")
    
    def get_figure(self):
        if self.figure is None:
            self.create_figure()
        return self.figure

    @abstractmethod
    def create_figure(self):
        pass

    @abstractmethod
    def redraw_plot(self):
        pass

    def save_plot(self):
        if self.figure is None:
            raise ValueError("No plot has been created to save.")
        self.figure.savefig(self.file_path, dpi=self.dpi)
    
