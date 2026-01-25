from typing import Dict, List, Optional, Tuple

from matplotlib.backend_bases import MouseEvent

from .Selector import Selector
from plotters import Plotter
from markers import MPoints

class SFreq(Selector):
    def __init__(self,
                 stage_name: str,
                 plotter: Plotter,
                 markers: Optional[Dict[str, MPoints]] = None,
                 buttons: Optional[List[Tuple[str, str]]] = None,
                 clear_button: bool = True,
                 save_button: bool = True):
        if markers is None:
            raise ValueError("markers cannot be None")
        if buttons is None:
            buttons = [("select_freq", "Select Frequency")]

        super().__init__(stage_name, plotter, markers, buttons, clear_button, save_button)
    
    def mouse_on_click(self, event: MouseEvent) -> None:
        if self.mode == "select_freq":
            self.params.update({'select_freq': event.ydata})
            self.markers["MPoints"].set_ticks([(event.xdata, event.ydata)])
            self.markers["MPoints"].redraw()