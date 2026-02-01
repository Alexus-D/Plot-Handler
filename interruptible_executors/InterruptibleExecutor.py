from abc import abstractmethod

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector

from executors import Executor
from plotters import PContour
from validators import VContourf


class InterruptibleExecutor(Executor):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
        self.interrupted = False
        
        self.plotter = None
        self.marker = None

        self.correcting_params = {}

    def execute(self):
        self.remain_data = self.data.copy()
        self.interrupted = False

        self._setup_execution_plot(self.result)

        while True:
            partial_result = self.interruptible_function()
            self.result.update(partial_result)

            if self.interrupted:
                self.delete_wrong_results()
                self.select_correcting_params()

                self.interrupted = False
            else:
                break

        self._close_execution_plot()
        return self.result
    
    def _close_execution_plot(self):
        if self.plotter is None:
            raise ValueError("Plotter is not initialized.")
        plt.close(self.plotter.get_figure())
        self.plotter = None
        plt.ioff()

    def _setup_execution_plot(self, result):
        if self.plotter is None:
            raise ValueError("Plotter is not initialized.")
        plt.ion()
        self._interrupt_btn = self._make_interrupt_button(self.plotter.get_figure())
        self._update_line(result)

    def on_interrupt(self, event):
        self.interrupted = True

    def _make_interrupt_button(self, figure):
        ax_interrupt = figure.add_axes([0.81, 0.01, 0.1, 0.05])
        btn_interrupt = Button(ax_interrupt, 'Interrupt')

        btn_interrupt.on_clicked(self.on_interrupt)
        return btn_interrupt

    @abstractmethod
    def _update_line(self, points)  -> None:
        pass

    @abstractmethod
    def delete_wrong_results(self) -> None:
        pass

    @abstractmethod
    def interruptible_function(self) -> dict:
        pass

    @abstractmethod
    def select_correcting_params(self):
        pass
