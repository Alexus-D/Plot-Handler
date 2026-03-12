from abc import abstractmethod

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector

from executors import Executor
from plotters import PContour
from validators import VContourf


class InterruptibleExecutor(Executor):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None, save_figure_path: str = None):
        super().__init__(data, stage_name, initial_params)
        self.interrupted = False
        self.skip_delete_wrong_results = False
        
        self.plotter = None
        self.marker = None

        self.correcting_params = {}
        self.save_figure_path = save_figure_path

    def execute(self):
        self.remain_data = self.data.copy()
        self.interrupted = False
        print("[DEBUG InterruptibleExecutor] execute() начат")

        self._setup_execution_plot(self.result)

        while True:
            print("[DEBUG InterruptibleExecutor] Вызов interruptible_function()")
            partial_result = self.interruptible_function()
            self.result.update(partial_result)

            if self.interrupted:
                print("[DEBUG InterruptibleExecutor] Прерывание обнаружено!")
                if not self.skip_delete_wrong_results:
                    print("[DEBUG InterruptibleExecutor] Вызов delete_wrong_results()")
                    self.delete_wrong_results()
                print("[DEBUG InterruptibleExecutor] Вызов select_correcting_params()")
                self.select_correcting_params()

                self.interrupted = False
                self.skip_delete_wrong_results = False
                print("[DEBUG InterruptibleExecutor] Продолжаем выполнение после корректировки")
            else:
                print("[DEBUG InterruptibleExecutor] Выполнение завершено без прерываний")
                break

        # Save figure before closing if save_figure_path is set
        if hasattr(self, 'save_figure_path') and self.save_figure_path and hasattr(self, 'save_figure'):
            self.save_figure(self.save_figure_path)
        
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
        print("[DEBUG InterruptibleExecutor] КНОПКА INTERRUPT НАЖАТА!")
        self.interrupted = True

    def _make_interrupt_button(self, figure):
        ax_interrupt = figure.add_axes([0.81, 0.01, 0.1, 0.05])
        btn_interrupt = Button(ax_interrupt, 'Interrupt')

        btn_interrupt.on_clicked(self.on_interrupt)
        return btn_interrupt
    
    def validate(self):
        return True
    
    def select_initial_params(self):
        self.select_correcting_params()

    def prepare_for_visualization(self):
        return

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
