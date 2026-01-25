from abc import ABC
from abc import abstractmethod
import pickle

from matplotlib.widgets import Button

from plotters import Plotter
from markers import Marker

class Selector(ABC):
    def __init__(self,
                 stage_name: str,
                 plot_data: dict,
                 buttons: list((str, str)),
                 clear_button: bool = True,
                 save_button: bool = True):
        if plot_data is None:
            raise ValueError("plot_data cannot be None")
        if buttons is None:
            raise ValueError("buttons cannot be None")
        
        self.stage_name = stage_name
        self.plotter = self._create_plotter(plot_data)
        self.figure = self.plotter.get_figure()
        self.markers = self._create_markers(self.plotter)

        self.buttons = None
        self._create_buttons(buttons, clear_button, save_button)
        self.pressed_button = []

        self.modes = [btn[0] for btn in buttons]
        self.mode = None

        self.params = {}

        self.mplot_cid = self.figure.canvas.mpl_connect('button_press_event', self.mouse_on_click)

    def get_params(self) -> dict:
        return self.params
    
    def clear_params(self):
        self.params = {}
        self.mode = None
        for marker in self.markers.values():
            marker.redraw()
    
    def save_params(self):
        if len(self.params) == 0:
            raise ValueError("No parameters to save")
        
        with open(f"{self.stage_name}_selector_params.pkl", "wb") as f:
            pickle.dump(self.params, f)
    
    def _create_buttons(self, buttons, clear_button, save_button):
        button_width = 0.15
        button_height = 0.04
        button_left = 0.02
        button_spacing = 0.06
        button_bottom = 0.15

        if clear_button:
            buttons.insert(0, ('clear', 'Clear'))
        if save_button:
            buttons.insert(1, ('save', 'Save'))

        for mode, label in buttons:
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
        if mode == 'clear':
            self.clear_params()
            self.mode = None
            return
        elif mode == 'save':
            self.save_params()
            self.mode = None
            return
        self.mode = mode

    @abstractmethod
    def mouse_on_click(self, event):
        pass

    @abstractmethod
    def _create_plotter(self, plot_data: dict) -> Plotter:
        pass

    @abstractmethod
    def _create_markers(self, plotter: Plotter) -> dict(Marker):
        pass