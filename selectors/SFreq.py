from typing import Dict, List, Optional, Tuple

from matplotlib.backend_bases import MouseEvent

from .Selector import Selector
from plotters import Plotter, PContour
from markers import MPoints

class SFreq(Selector):
    def __init__(self,
                 stage_name: str,
                 plot_data: dict,
                 buttons: Optional[List[Tuple[str, str]]] = None,
                 clear_button: bool = True,
                 save_button: bool = True):
        if buttons is None:
            buttons = [("select_freq", "Select Frequency")]

        super().__init__(stage_name, plot_data, buttons, clear_button, save_button)
    
    def mouse_on_click(self, event: MouseEvent) -> None:
        if self.mode == "select_freq":
            self.params.update({'select_freq': event.ydata})
            self.markers["MPoints"].set_ticks([(event.xdata, event.ydata)])
            self.markers["MPoints"].redraw()
        
    def _create_plotter(self, plot_data: dict) -> Plotter:
        return PContour(plot_data)
    
    def _create_markers(self, plotter: Plotter) -> Dict[str, MPoints]:
        axis = plotter.get_axis_for_marker()
        marker = MPoints(axis)
        return {"MPoints": marker}