import numpy as np
import matplotlib.pyplot as plt

import config_physics
import utils.algorithms as alg

from markers import MPoints
from plotters import PContour
from ui_selectors import SPeakTrajectories
from executors import ECut, EPeakParams, ETrajectories

from .InterruptibleExecutor import InterruptibleExecutor


class IEPeakWatcher(InterruptibleExecutor):
	def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
		super().__init__(data, stage_name, initial_params)
		self.plotter = None
		self.marker = None
		self.result = {"trajectories": []}
		self._trajectory_params = []
		self._bad_trajectories = []

	def execute(self):
		if self.plotter is None:
			self.plotter = PContour(self.data)
			self.marker = MPoints(self.plotter.get_axis_for_marker())
			fig = self.plotter.get_figure()
			fig.canvas.draw_idle()
			plt.show(block=False)
			plt.pause(0.001)

		self.remain_data = self.data.copy()
		self.interrupted = False
		self._setup_execution_plot(self.result)

		while True:
			partial_result = self.interruptible_function()
			self.result.update(partial_result)

			if self.interrupted:
				self.delete_wrong_results()
				self.select_correcting_params()

				self.interrupted = False
			else:
				break

		plt.show()
		self._close_execution_plot()
		return self.result

	def select_correcting_params(self):
		if self.initial_params.get("trajectories") is None:
			trajectories = self._select_trajectories()
			self.initial_params = {"trajectories": trajectories}

		trajectories = self.initial_params.get("trajectories", [])
		if not self._trajectory_params or len(self._trajectory_params) != len(trajectories):
			self._trajectory_params = self._select_peak_params_for_trajectories(trajectories)
			self.initial_params["peak_params"] = self._trajectory_params

		if self._bad_trajectories:
			first_bad = self._bad_trajectories[0]
			first_params = self._select_peak_params_for_trajectory(first_bad, "Correction")
			bad_params = [first_params]
			for bad in self._bad_trajectories[1:]:
				bad_params.append(self._find_params_for_trajectory(bad) or first_params)
			self.correcting_params = {
				"trajectories": self._bad_trajectories,
				"peak_params": bad_params,
			}
			self._bad_trajectories = []
			return

		self.correcting_params = {}

	def delete_wrong_results(self) -> None:
		current = self.result.get("trajectories", [])
		if not current:
			return
		selector = SPeakTrajectories(f"{self.stage_name} - Remove bad", self.data)
		plt.show()
		selected = selector.get_params().get("trajectories", [])
		if not selected:
			return

		bad_indices = self._match_selected_trajectories(selected, current)
		self._bad_trajectories = [current[i] for i in bad_indices]
		self.result = {
			"trajectories": [t for i, t in enumerate(current) if i not in bad_indices]
		}
		self._update_line(self.result)

	def _update_line(self, result) -> None:
		if self.marker is None:
			raise ValueError("Marker is not initialized.")
		points = []
		for traj in result.get("trajectories", []):
			for p in traj.get("points", []):
				points.append((p["x"], p["freq"]))
		self.marker.update_ticks(points)

	def _select_trajectories(self):
		traj_executor = ETrajectories(self.data, f"{self.stage_name} - Trajectories")
		traj_executor.execute_with_validation()
		return traj_executor.get_result().get("trajectories", [])

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

	def _find_params_for_trajectory(self, trajectory):
		trajectory = self._normalize_trajectory(trajectory)
		trajectories = self.initial_params.get("trajectories", [])
		params_list = self.initial_params.get("peak_params", [])
		if not trajectories or not params_list:
			return None
		idx = self._match_single_trajectory(trajectory, trajectories)
		if idx is None or idx >= len(params_list):
			return None
		return params_list[idx]

	def _normalize_trajectory(self, trajectory):
		if isinstance(trajectory, dict):
			return trajectory.get("start"), trajectory.get("end")
		return trajectory

	def _match_single_trajectory(self, target, trajectories):
		start_t, end_t = self._normalize_trajectory(target)
		best_idx = None
		best_dist = None
		for i, traj in enumerate(trajectories):
			start, end = self._normalize_trajectory(traj)
			dist = (start[0] - start_t[0]) ** 2 + (start[1] - start_t[1]) ** 2
			dist += (end[0] - end_t[0]) ** 2 + (end[1] - end_t[1]) ** 2
			if best_dist is None or dist < best_dist:
				best_dist = dist
				best_idx = i
		return best_idx

	def _match_selected_trajectories(self, selected, current):
		used = set()
		indices = []
		for sel in selected:
			start_s, end_s = sel
			best_idx = None
			best_dist = None
			for i, traj in enumerate(current):
				if i in used:
					continue
				start, end = traj.get("start", (None, None)), traj.get("end", (None, None))
				if start[0] is None or end[0] is None:
					continue
				dist = (start[0] - start_s[0]) ** 2 + (start[1] - start_s[1]) ** 2
				dist += (end[0] - end_s[0]) ** 2 + (end[1] - end_s[1]) ** 2
				if best_dist is None or dist < best_dist:
					best_dist = dist
					best_idx = i
			if best_idx is not None:
				used.add(best_idx)
				indices.append(best_idx)
		return indices

	def interruptible_function(self) -> dict:
		params = self.correcting_params or self.initial_params
		trajectories = params.get("trajectories", [])
		peak_params_list = params.get("peak_params")
		if peak_params_list is None:
			peak_params_list = self.initial_params.get("peak_params", [])

		if peak_params_list and len(peak_params_list) != len(trajectories):
			raise ValueError("peak_params must match trajectories length")

		x_values = np.array(self.data.get("x", []))
		y_values = np.array(self.data.get("y", []))
		z_values = np.array(self.data.get("z", []))
		if x_values.size == 0 or y_values.size == 0 or z_values.size == 0:
			raise ValueError("data must contain non-empty 'x', 'y', and 'z' arrays")

		result_trajectories = list(self.result.get("trajectories", []))
		fig = self.plotter.get_figure()

		for idx, traj in enumerate(trajectories):
			if self.interrupted:
				break

			start, end = self._normalize_trajectory(traj)

			peak_params = peak_params_list[idx] if peak_params_list else {}
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
				expected_width = max(1e-12, float(np.ptp(y_values)) * 0.02)
			if prominence is None:
				prominence = float(np.nanstd(z_values)) * 0.5

			x1, y1 = start
			x2, y2 = end
			x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
			mask = (x_values >= x_min) & (x_values <= x_max)
			indices = np.where(mask)[0]
			if x2 < x1:
				indices = indices[::-1]
			points = []

			for i in indices:
				if self.interrupted:
					break
				x = float(x_values[i])
				if expected_freq is None:
					if x2 != x1:
						expected_freq = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
					else:
						expected_freq = y1

				s_values = z_values[i, :]
				try:
					peak = alg.find_peak(
						y_values,
						s_values,
						expected_freq,
						expected_width,
						prominence,
						peak_type=peak_type,
					)
				except Exception:
					continue

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

				self._update_line({"trajectories": result_trajectories + [{"points": points}]})
				fig.canvas.flush_events()

			result_trajectories.append(
				{
					"start": (float(x1), float(y1)),
					"end": (float(x2), float(y2)),
					"points": points,
					"peak_params": peak_params,
				}
			)

		result = {"trajectories": result_trajectories}
		self.result = result
		return result