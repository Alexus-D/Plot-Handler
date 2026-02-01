from typing import Dict, List, Optional, Tuple

from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt

from .Selector import Selector
from plotters import Plotter, PContour
from markers import MPoints


class SPeakTrajectories(Selector):
	def __init__(self,
				 stage_name: str,
				 plot_data: dict,
				 buttons: Optional[List[Tuple[str, str]]] = None,
				 clear_button: bool = False,
				 save_button: bool = False):
		if buttons is None:
			buttons = [
				("mark_trajectory", "Mark Trajectory"),
				("finish", "Finish"),
			]

		super().__init__(stage_name, plot_data, buttons, clear_button, save_button)
		self._pending_points: List[Tuple[float, float]] = []
		self.params["trajectories"] = []
		self.finished = False

	def _set_mode(self, mode):
		if mode == "finish":
			self._finalize_trajectories()
			self.mode = None
			self.finished = True
			return
		super()._set_mode(mode)

	def mouse_on_click(self, event: MouseEvent) -> None:
		if self.mode != "mark_trajectory":
			return
		if event.inaxes is None:
			return
		if event.xdata is None or event.ydata is None:
			return

		self._pending_points.append((event.xdata, event.ydata))
		self._redraw_markers()

		if len(self._pending_points) == 2:
			start, end = self._pending_points
			self.params["trajectories"].append((start, end))
			self._pending_points = []
			self.mode = None

	def _create_plotter(self, plot_data: dict) -> Plotter:
		return PContour(plot_data)

	def _create_markers(self, plotter: Plotter) -> Dict[str, MPoints]:
		axis = plotter.get_axis_for_marker()
		marker = MPoints(axis)
		return {"MPoints": marker}

	def _redraw_markers(self):
		marker = self.markers.get("MPoints")
		if marker is None:
			return
		points = []
		for start, end in self.params.get("trajectories", []):
			points.extend([start, end])
		points.extend(self._pending_points)
		marker.update_ticks(points)

	def _finalize_trajectories(self):
		self._draw_trajectories()
		self._clear_markers()

	def _draw_trajectories(self):
		axis = self.plotter.get_axis_for_marker()
		for start, end in self.params.get("trajectories", []):
			xs = [start[0], end[0]]
			ys = [start[1], end[1]]
			axis.plot(xs, ys, color="red", linewidth=1)
		axis.figure.canvas.draw_idle()

	def _clear_markers(self):
		marker = self.markers.get("MPoints")
		if marker is None:
			return
		marker.delete_ticks()
		marker.redraw()
