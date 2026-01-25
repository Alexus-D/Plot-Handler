from .Selector import Selector

from plotters import PContour
from markers import MPoints


class SNPoints(Selector):
    def __init__(self,
                 stage_name: str,
                 plot_data: dict,
                 num_points,
                 buttons: list((str, str)) = None,
                 clear_button: bool = True,
                 save_button: bool = True):
        self.num_points = num_points
        if buttons is None:
            buttons = [("select_points", "Select Points")]

        super().__init__(stage_name, plot_data, buttons, clear_button, save_button)
    
    def mouse_on_click(self, event):
        if self.mode == "select_points":
            if "select_points" not in self.params:
                self.params["select_points"] = []
            if len(self.params["select_points"]) >= self.num_points:
                return
            self.params["select_points"].append((event.xdata, event.ydata))
            self.markers["MPoints"].set_ticks(self.params["select_points"])
            self.markers["MPoints"].redraw()
    
    def _create_plotter(self, plot_data: dict):
        return PContour(plot_data)
    
    def _create_markers(self, plotter):
        axis = plotter.get_axis_for_marker()
        marker = MPoints(axis)
        return {"MPoints": marker}