from typing import Dict, List, Optional, Tuple

from matplotlib.backend_bases import MouseEvent

import config_physics

from .Selector import Selector
from plotters import Plotter
from plotters.PCut import PCut
from markers import MPoints


class SPeakParams(Selector):
	def __init__(self,
				 stage_name: str,
				 plot_data: dict,
				 buttons: Optional[List[Tuple[str, str]]] = None,
				 clear_button: bool = True,
				 save_button: bool = True):
		if buttons is None:
			buttons = [
				("select_peak", "Select Peak"),
				("select_width", "Select Width"),
				("select_plateau", "Select Plateau"),
				("set_max", "Peak: Maximum"),
				("set_min", "Peak: Minimum"),
			]

		super().__init__(stage_name, plot_data, buttons, clear_button, save_button)

		self._peak_point = None
		self._width_points = []
		self._plateau_point = None
		self._peak_type_locked = False
		self._highlight_points = list(plot_data.get("highlight_points", []))
		if self._highlight_points:
			self._redraw_markers()

	def _set_mode(self, mode):
		if mode == "set_max":
			self.params["peak_type"] = "maximum"
			self._peak_type_locked = True
			self.mode = None
			return
		if mode == "set_min":
			self.params["peak_type"] = "minimum"
			self._peak_type_locked = True
			self.mode = None
			return
		super()._set_mode(mode)

	def mouse_on_click(self, event: MouseEvent) -> None:
		if event.inaxes is None:
			return
		if event.xdata is None or event.ydata is None:
			return

		if self.mode == "select_peak":
			self._peak_point = (event.xdata, event.ydata)
			self.params["peak_freq"] = event.xdata
			self.params["peak_value"] = event.ydata
			self._infer_peak_type()
			self._redraw_markers()
		elif self.mode == "select_width":
			self._width_points.append((event.xdata, event.ydata))
			if len(self._width_points) >= 2:
				width = abs(self._width_points[1][0] - self._width_points[0][0])
				self.params["peak_width"] = width
				self._width_points = self._width_points[:2]
				self._redraw_markers()
		elif self.mode == "select_plateau":
			self._plateau_point = (event.xdata, event.ydata)
			self.params["plateau"] = event.ydata
			self._infer_peak_type()
			self._redraw_markers()

	def _infer_peak_type(self):
		if self._peak_type_locked:
			return
		peak_value = self.params.get("peak_value")
		plateau = self.params.get("plateau")
		if peak_value is None or plateau is None:
			if "peak_type" not in self.params:
				self.params["peak_type"] = config_physics.PEAK_TYPE
			return
		self.params["peak_type"] = "maximum" if peak_value >= plateau else "minimum"

	def _create_plotter(self, plot_data: dict) -> Plotter:
		return PCut(plot_data)

	def _create_markers(self, plotter: Plotter) -> Dict[str, MPoints]:
		axis = plotter.get_axis_for_marker()
		marker = MPoints(axis)
		return {"MPoints": marker}

	def _redraw_markers(self):
		marker = self.markers.get("MPoints")
		if marker is None:
			return
		points = []
		points.extend(self._highlight_points)
		if self._peak_point is not None:
			points.append(self._peak_point)
		points.extend(self._width_points)
		if self._plateau_point is not None:
			points.append(self._plateau_point)
		marker.update_ticks(points)
