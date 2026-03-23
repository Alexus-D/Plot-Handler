"""UI selector for two-point calibration (e.g., magnon frequency vs. field)."""

import matplotlib.pyplot as plt

from .Selector import Selector
from plotters import PContour
from markers import MPoints


class STwoPointsCalibration(Selector):
    """
    Selector that waits for exactly two clicks on a contour plot to calibrate
    a linear relationship (e.g., magnon frequency = slope * field + intercept).

    Usage:
        selector = STwoPointsCalibration("Magnon Calibration", contour_data)
        plt.show()
        params = selector.get_params()
        # params["calibration_points"] = [(field1, freq1), (field2, freq2)]
    """

    def __init__(
        self,
        stage_name: str,
        plot_data: dict,
        buttons: list = None,
        clear_button: bool = True,
        save_button: bool = False,
    ):
        if buttons is None:
            buttons = [("select_points", "Select 2 Points")]

        super().__init__(stage_name, plot_data, buttons, clear_button, save_button)

    def mouse_on_click(self, event):
        """Handle mouse click: collect up to 2 points, then close."""
        if event.inaxes is None:
            return

        if self.mode == "select_points":
            if "calibration_points" not in self.params:
                self.params["calibration_points"] = []

            points = self.params["calibration_points"]
            if len(points) >= 2:
                return

            points.append((event.xdata, event.ydata))
            self.markers["MPoints"].update_ticks(points)

            if len(points) == 2:
                print(
                    f"[STwoPointsCalibration] Two points selected: "
                    f"({points[0][0]:.2f}, {points[0][1]:.4f}) and "
                    f"({points[1][0]:.2f}, {points[1][1]:.4f})"
                )
                self.mode = None
                plt.close(self.figure)

    def _create_plotter(self, plot_data: dict):
        return PContour(plot_data)

    def _create_markers(self, plotter):
        axis = plotter.get_axis_for_marker()
        marker = MPoints(axis)
        return {"MPoints": marker}
