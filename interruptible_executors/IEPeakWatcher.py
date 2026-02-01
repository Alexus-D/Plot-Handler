import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button

import config_physics
import utils.algorithms as alg

from markers import MPoints
from plotters import PContour
from ui_selectors import SPeakTrajectories
from executors import ECut, EPeakParams
from validators import VContourf

from .InterruptibleExecutor import InterruptibleExecutor


class IEPeakWatcher(InterruptibleExecutor):
	def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
		super().__init__(data, stage_name, initial_params)
		self.plotter = None
		self.marker = None
		self.result = {"trajectories": []}
		self._trajectory_params = []
		self._trajectory_indices = {}
		self._x_values = None
		self._y_values = None
		self._z_values = None
		self._selection_box = None
		self._selection_done = False

	def execute(self):
		trajectories = self.initial_params.get("trajectories")
		if trajectories is None:
			trajectories = self._select_trajectories_on_contour()
			self.initial_params["trajectories"] = trajectories
		else:
			self._ensure_plotter()

		self._clear_selection_markers()

		if not trajectories:
			self.result = {"trajectories": []}
			return self.result

		self._prepare_data_arrays()
		if not self._trajectory_params or len(self._trajectory_params) != len(trajectories):
			self._trajectory_params = self._select_peak_params_for_trajectories(trajectories)
			self.initial_params["peak_params"] = self._trajectory_params

		self.result = {
			"trajectories": [
				{"start": t[0], "end": t[1], "points": [], "peak_params": self._trajectory_params[i]}
				for i, t in enumerate(trajectories)
			]
		}

		self.interrupted = False
		self._setup_execution_plot(self.result)

		for traj_idx, traj in enumerate(trajectories):
			self._track_trajectory(traj_idx, traj)
			while not self._confirm_trajectory(traj_idx):
				self._handle_corrections_for_trajectory(traj_idx, traj)
				self._track_trajectory(traj_idx, traj, resume=True)

		plt.show()
		self._close_execution_plot()
		return self.result

	def select_correcting_params(self):
		self.correcting_params = {}

	def delete_wrong_results(self) -> None:
		return

	def interruptible_function(self) -> dict:
		return {}

	def _update_line(self, result) -> None:
		if self.marker is None:
			raise ValueError("Marker is not initialized.")
		points = []
		for traj in result.get("trajectories", []):
			for p in traj.get("points", []):
				points.append((p["x"], p["freq"]))
		self.marker.update_ticks(points)

	def _setup_execution_plot(self, result):
		if self.plotter is None:
			raise ValueError("Plotter is not initialized.")
		plt.ion()
		fig = self.plotter.get_figure()
		if hasattr(fig.canvas, "toolbar"):
			fig.canvas.toolbar = None
		try:
			self._interrupt_btn = self._make_interrupt_button(fig)
		except Exception:
			self._interrupt_btn = None
		self._update_line(result)

	def _ensure_plotter(self):
		if self.plotter is None or not plt.fignum_exists(self.plotter.get_figure().number):
			self.plotter = PContour(self.data)
			self.marker = MPoints(self.plotter.get_axis_for_marker())
			fig = self.plotter.get_figure()
			fig.canvas.draw_idle()
			plt.show(block=False)
			plt.pause(0.001)

	def _select_trajectories_on_contour(self):
		selector = SPeakTrajectories(self.stage_name, self.data)
		plt.show(block=False)
		while not selector.finished:
			plt.pause(0.1)
		trajectories = selector.get_params().get("trajectories", [])
		self.plotter = selector.plotter
		self.marker = MPoints(self.plotter.get_axis_for_marker())
		return trajectories

	def _clear_selection_markers(self):
		if self.plotter is None:
			return
		axis = self.plotter.get_axis_for_marker()
		lines = list(axis.lines)
		for line in lines:
			line.remove()
		axis.figure.canvas.draw_idle()
		self.marker.delete_ticks()
		self.marker.redraw()

	def _select_peak_params_for_trajectories(self, trajectories):
		params_list = []
		for idx, traj in enumerate(trajectories):
			params_list.append(self._select_peak_params_for_trajectory(traj, f"T{idx + 1}"))
		return params_list

	def _select_peak_params_for_trajectory(self, trajectory, suffix):
		start = trajectory[0] if isinstance(trajectory, (list, tuple)) else trajectory.get("start")
		if start is None:
			raise ValueError("Trajectory must contain start point")
		x_start = float(start[0])
		y_start = float(start[1])
		cut_executor = ECut(self.data, f"{self.stage_name} - Cut {suffix}", axis="x", initial_params={"cut_value": x_start})
		cut_executor.execute_with_validation()
		cut_data = cut_executor.get_result()
		self._add_highlight_point(cut_data, y_start)

		peak_executor = EPeakParams(cut_data, f"{self.stage_name} - Peak {suffix}")
		peak_executor.execute_with_validation()
		return peak_executor.get_result()

	def _add_highlight_point(self, cut_data: dict, freq_value: float) -> None:
		x_values = np.array(cut_data.get("x", []), dtype=float)
		y_values = np.array(cut_data.get("y", []), dtype=float)
		if x_values.size == 0 or y_values.size == 0:
			return
		idx = int(np.abs(x_values - freq_value).argmin())
		point = (float(x_values[idx]), float(y_values[idx]))
		cut_data["highlight_points"] = [point]

	def _prepare_data_arrays(self):
		self._x_values = np.array(self.data.get("x", []))
		self._y_values = np.array(self.data.get("y", []))
		self._z_values = np.array(self.data.get("z", []))
		if self._x_values.size == 0 or self._y_values.size == 0 or self._z_values.size == 0:
			raise ValueError("data must contain non-empty 'x', 'y', and 'z' arrays")

	def _get_indices_for_trajectory(self, trajectory):
		start, end = trajectory
		x1, _ = start
		x2, _ = end
		x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
		mask = (self._x_values >= x_min) & (self._x_values <= x_max)
		indices = np.where(mask)[0]
		if x2 < x1:
			indices = indices[::-1]
		return indices

	def _track_trajectory(self, traj_idx, trajectory, resume: bool = False):
		indices = self._trajectory_indices.get(traj_idx)
		if indices is None:
			indices = self._get_indices_for_trajectory(trajectory)
			self._trajectory_indices[traj_idx] = indices
		if len(indices) == 0:
			return
		print(f"Tracking trajectory {traj_idx + 1}/{len(self._trajectory_indices)}: {len(indices)} points")

		points = self.result["trajectories"][traj_idx]["points"]
		peak_params = self._trajectory_params[traj_idx]

		start_pos = 0
		if resume and points:
			last_x = points[-1]["x"]
			start_pos = int(np.abs(self._x_values[indices] - last_x).argmin()) + 1

		expected_freq = peak_params.get("peak_freq")
		expected_width = peak_params.get("peak_width")
		peak_type = peak_params.get("peak_type", config_physics.PEAK_TYPE)
		prominence = peak_params.get("prominence")
		if prominence is None:
			peak_value = peak_params.get("peak_value")
			plateau = peak_params.get("plateau")
			if peak_value is not None and plateau is not None:
				prominence = abs(peak_value - plateau)
		if expected_width is None:
			expected_width = max(1e-12, float(np.ptp(self._y_values)) * 0.02)
		if prominence is None:
			prominence = float(np.nanstd(self._z_values)) * 0.5

		freq_step = float(self._y_values[1] - self._y_values[0]) if len(self._y_values) > 1 else 1e-3
		expected_width = max(expected_width, abs(freq_step) * 2)
		prominence = max(prominence, 0.0)

		start, end = trajectory
		x1, y1 = start
		x2, y2 = end

		pos = start_pos
		while pos < len(indices):
			if self.interrupted:
				pos, expected_freq, expected_width, prominence, peak_type = self._handle_interrupt(
					traj_idx,
					trajectory,
					indices,
					pos,
					expected_freq,
					expected_width,
					prominence,
					peak_type,
				)
				self.interrupted = False
				continue

			i = indices[pos]
			x = float(self._x_values[i])
			if expected_freq is None:
				if x2 != x1:
					expected_freq = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
				else:
					expected_freq = y1

			s_values = self._z_values[i, :]
			try:
				search_window = max(expected_width * 3, abs(freq_step) * 6)
				peak = alg.find_peak(
					self._y_values,
					s_values,
					expected_freq,
					expected_width,
					prominence,
					search_window=search_window,
					peak_type=peak_type,
				)
			except Exception:
				idx = int(np.abs(self._y_values - expected_freq).argmin())
				value = float(s_values[idx])
				peak = {
					"freq": float(self._y_values[idx]),
					"magnitude": value,
					"prominence": abs(value - float(np.median(s_values))),
					"width": expected_width,
				}

			expected_freq = float(peak["freq"])
			expected_width = float(peak["width"]) or expected_width
			prominence = float(peak["prominence"]) or prominence

			points.append(
				{
					"x": x,
					"freq": float(peak["freq"]),
					"value": float(peak["magnitude"]),
					"prominence": float(peak["prominence"]),
					"width": float(peak["width"]),
				}
			)
			self._update_line(self.result)
			self.plotter.get_figure().canvas.flush_events()
			plt.pause(0.1)
			pos += 1
		print(f"Done trajectory {traj_idx + 1}, points: {len(points)}")

	def _handle_interrupt(self, traj_idx, trajectory, indices, pos, expected_freq, expected_width, prominence, peak_type):
		bad_points = self._select_bad_points_rectangle()
		if not bad_points:
			return pos, expected_freq, expected_width, prominence, peak_type

		bad_by_traj = {}
		for item in bad_points:
			bad_by_traj.setdefault(item["traj_idx"], []).append(item)

		for idx, items in bad_by_traj.items():
			points = self.result["trajectories"][idx]["points"]
			for item in sorted(items, key=lambda x: x["point_idx"], reverse=True):
				points.pop(item["point_idx"])

		self._update_line(self.result)

		items = bad_by_traj.get(traj_idx)
		if not items:
			return pos, expected_freq, expected_width, prominence, peak_type

		first_bad = self._first_bad_by_indices(items, indices)
		new_params = self._select_peak_params_for_point(first_bad["x"], first_bad["freq"], f"Correction T{traj_idx + 1}")
		self._trajectory_params[traj_idx] = new_params
		self.result["trajectories"][traj_idx]["peak_params"] = new_params

		points = self.result["trajectories"][traj_idx]["points"]
		points[:] = self._trim_points_before(points, first_bad["x"], indices)

		start_pos = int(np.abs(self._x_values[indices] - first_bad["x"]).argmin())

		expected_freq = new_params.get("peak_freq")
		expected_width = new_params.get("peak_width", expected_width)
		peak_type = new_params.get("peak_type", peak_type)
		prominence = new_params.get("prominence")
		if prominence is None:
			peak_value = new_params.get("peak_value")
			plateau = new_params.get("plateau")
			if peak_value is not None and plateau is not None:
				prominence = abs(peak_value - plateau)

		return start_pos, expected_freq, expected_width, prominence, peak_type

	def _select_bad_points_rectangle(self):
		axis = self.plotter.get_axis_for_marker()
		self._selection_box = None
		self._selection_done = False

		def onselect(eclick, erelease):
			x0, y0 = eclick.xdata, eclick.ydata
			x1, y1 = erelease.xdata, erelease.ydata
			if None in (x0, y0, x1, y1):
				return
			self._selection_box = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))

		selector = RectangleSelector(axis, onselect, useblit=True, button=[1], interactive=True)

		def on_key(event):
			if event.key in ("enter", "return", "escape"):
				self._selection_done = True

		fig = self.plotter.get_figure()
		cid = fig.canvas.mpl_connect("key_press_event", on_key)

		while not self._selection_done:
			plt.pause(0.1)

		selector.set_visible(False)
		selector.disconnect_events()
		fig.canvas.mpl_disconnect(cid)
		fig.canvas.draw_idle()

		if self._selection_box is None:
			return []

		x0, x1, y0, y1 = self._selection_box
		bad_points = []
		for traj_idx, traj in enumerate(self.result.get("trajectories", [])):
			for point_idx, p in enumerate(traj.get("points", [])):
				if x0 <= p["x"] <= x1 and y0 <= p["freq"] <= y1:
					bad_points.append(
						{
							"traj_idx": traj_idx,
							"point_idx": point_idx,
							"x": p["x"],
							"freq": p["freq"],
						}
					)
		return bad_points

	def _select_peak_params_for_point(self, x_value, freq_value, suffix):
		cut_executor = ECut(self.data, f"{self.stage_name} - Cut {suffix}", axis="x", initial_params={"cut_value": x_value})
		cut_executor.execute_with_validation()
		cut_data = cut_executor.get_result()
		self._add_highlight_point(cut_data, freq_value)
		peak_executor = EPeakParams(cut_data, f"{self.stage_name} - Peak {suffix}")
		peak_executor.execute_with_validation()
		return peak_executor.get_result()

	def _handle_corrections_for_trajectory(self, traj_idx, trajectory):
		bad_points = self._select_bad_points_rectangle()
		if not bad_points:
			return
		bad_by_traj = {}
		for item in bad_points:
			bad_by_traj.setdefault(item["traj_idx"], []).append(item)
		for idx, items in bad_by_traj.items():
			points = self.result["trajectories"][idx]["points"]
			for item in sorted(items, key=lambda x: x["point_idx"], reverse=True):
				points.pop(item["point_idx"])

		items = bad_by_traj.get(traj_idx)
		if not items:
			return
		indices = self._trajectory_indices.get(traj_idx) or self._get_indices_for_trajectory(trajectory)
		self._trajectory_indices[traj_idx] = indices
		first_bad = self._first_bad_by_indices(items, indices)
		new_params = self._select_peak_params_for_point(first_bad["x"], first_bad["freq"], f"Correction T{traj_idx + 1}")
		self._trajectory_params[traj_idx] = new_params
		self.result["trajectories"][traj_idx]["peak_params"] = new_params
		points = self.result["trajectories"][traj_idx]["points"]
		points[:] = self._trim_points_before(points, first_bad["x"], indices)
		self._update_line(self.result)

	def _first_bad_by_indices(self, items, indices):
		positions = []
		for item in items:
			pos = int(np.abs(self._x_values[indices] - item["x"]).argmin())
			positions.append((pos, item))
		return min(positions, key=lambda v: v[0])[1]

	def _trim_points_before(self, points, x_value, indices):
		if len(indices) < 2:
			return [p for p in points if p["x"] < x_value]
		descending = self._x_values[indices[0]] > self._x_values[indices[-1]]
		if descending:
			return [p for p in points if p["x"] > x_value]
		return [p for p in points if p["x"] < x_value]

	def _confirm_trajectory(self, traj_idx):
		validator = VContourf(f"{self.stage_name} - Confirm T{traj_idx + 1}", self.data)
		axis = validator.plotter.get_axis_for_marker()
		points = self.result["trajectories"][traj_idx]["points"]
		if points:
			xs = [p["x"] for p in points]
			ys = [p["freq"] for p in points]
			axis.plot(xs, ys, color="red", linewidth=1, marker="o", markersize=2)
		plt.show(block=False)
		while "Validation" not in validator.get_params():
			plt.pause(0.1)
		return bool(validator.get_params().get("Validation", False))