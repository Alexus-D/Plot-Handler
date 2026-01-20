from abc import ABC
from abc import abstractmethod
import pickle

from matplotlib.widgets import Button

from ..plotters.Plotter import Plotter

class Selector(ABC):
    def __init__(self,
                 stage_name: str,
                 plot_data: dict = None,
                 buttons: list = None):
        if plot_data is None:
            raise ValueError("plot_data cannot be None")
        if buttons is None:
            raise ValueError("buttons cannot be None")
        
        self.stage_name = stage_name
        self.plotter = Plotter(plot_data)
        self.figure = self.plotter.get_figure()
        self.button_params = buttons
        self.modes = [btn[0] for btn in self.button_params]

        self.buttons = None
        self._create_buttons()

        self.mode = None
        self.click_funks = self.set_click_funks()

        self.params = {}

        self.mplot_cid = self.figure.canvas.mpl_connect('button_press_event', self.on_clicked)
    
    def _create_buttons(self):
        button_width = 0.15
        button_height = 0.04
        button_left = 0.02
        button_spacing = 0.06
        button_bottom = 0.15

        for mode, label in self.button_params:
            ax_button = self.figure.add_axes([button_left, button_bottom, button_width, button_height])
            button = Button(ax_button, label)
            button.on_clicked(lambda event, m=mode: self._set_mode(m))
            if self.buttons is None:
                self.buttons = {}
            self.buttons[mode] = button
            button_bottom += button_spacing
        
    def _set_mode(self, mode):
        if mode not in self.modes:
            raise ValueError(f"Mode {mode} is not recognized")
        self.mode = mode

    def on_clicked(self, event):
        if self.click_funks is None:
            raise ValueError("Click functions are not set")
        if self.mode not in self.modes:
            return
        
        click_funk = self.click_funks[self.mode]
        self.params, reset_mode = click_funk(event, self.params)
        if reset_mode:
            self.mode = None
        self.redraw()
    
    @abstractmethod
    def set_click_funks(self):
        pass

    @abstractmethod
    def redraw(self):
        pass

    def get_params(self) -> dict:
        return self.params
    
    def clear_params(self):
        self.params = {}
        self.mode = None
        self.redraw()
    
    def save_params(self):
        if len(self.params) == 0:
            raise ValueError("No parameters to save")
        
        with open(f"{self.stage_name}_selector_params.pkl", "wb") as f:
            pickle.dump(self.params, f)
