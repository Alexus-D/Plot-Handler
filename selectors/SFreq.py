from .Selector import Selector
from plotters import Plotter
from markers import MPoints

class SFreq(Selector):
    def __init__(self,
                 stage_name: str,
                 plotter = Plotter,
                 markers: dict = {"MPoints": MPoints},
                 buttons: list = [("select_freq", "Select Frequency")],
                 clear_button: bool = True,
                 save_button: bool = True):
        
        super().__init__(stage_name, plotter, markers, buttons, clear_button, save_button)
    
    def mouse_on_click(self, event):
        if self.mode == "select_freq":
            self.params.update({'select_freq': event.ydata})
            self.markers["MPoints"].set_ticks([(event.xdata, event.ydata)])
            self.markers["MPoints"].redraw()