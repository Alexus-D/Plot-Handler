import matplotlib.pyplot as plt
import numpy as np

import config_physics

from markers import MPoints
from plotters import PContour
from validators import Validator

from .Executor import Executor


class VContourfBounds(Validator):
	def __init__(self, stage_name: str, plot_data: dict, boundary_points=None, peak_points=None):
		self._boundary_points = list(boundary_points or [])
		self._peak_points = list(peak_points or [])
		super().__init__(stage_name, plot_data)
		self._update_markers()

	def _create_plotter(self, plot_data: dict):
		return PContour(plot_data)

	def _create_markers(self, plotter: PContour):
		axis = plotter.get_axis_for_marker()
		return {}

	def _update_markers(self):
		axis = self.plotter.get_axis_for_marker()
		
		# Рисуем пики траекторий (черные кружки)
		if self._peak_points:
			peak_x = [p[0] for p in self._peak_points]
			peak_y = [p[1] for p in self._peak_points]
			axis.scatter(peak_x, peak_y, c='black', marker='o', s=20, label='Peaks', zorder=10)
		
		# Рисуем границы (красные крестики)
		if self._boundary_points:
			bound_x = [p[0] for p in self._boundary_points]
			bound_y = [p[1] for p in self._boundary_points]
			axis.scatter(bound_x, bound_y, c='red', marker='x', s=50, label='Bounds', zorder=11)
		
		axis.legend(loc='upper left', fontsize=10)


class ERangerApproximator(Executor):
	"""
	Находит границы по частоте для будущей аппроксимации пиков.

	Входные данные (data):
		- x, y, z: контурные данные (обычно результат EFilter)
		- trajectories: результат IEPeakWatcher

	Выходные данные (result):
		- trajectories: список словарей с полями:
			fields, freq, width, lower_bound, upper_bound, peak_type
	"""

	def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
		super().__init__(data, stage_name, initial_params)
		self.input_trajectories = data.get("trajectories", [])
		if not self.input_trajectories:
			raise ValueError("No trajectories in input data. Run IEPeakWatcher first.")

	def select_initial_params(self):
		return

	def _get_cut_at_field(self, field):
		fields = self.data["x"]
		freqs = self.data["y"]
		z = self.data["z"]

		field_idx = np.abs(fields - field).argmin()
		s_values = z[field_idx, :]
		return freqs, s_values

	def _is_opposite_extremum(self, s_values, idx, peak_type):
		if idx <= 0 or idx >= len(s_values) - 1:
			return False

		if peak_type == "maximum":
			return s_values[idx] < s_values[idx - 1] and s_values[idx] < s_values[idx + 1]
		return s_values[idx] > s_values[idx - 1] and s_values[idx] > s_values[idx + 1]

	def _find_bounds_for_peak(self, freqs, s_values, peak_freq, width, peak_type):
		if width is None or width <= 0:
			width = 0.01  # Минимальная ширина 10 MHz
		width = abs(width)

		peak_idx = int(np.abs(freqs - peak_freq).argmin())
		
		# Минимальное окно поиска - не меньше ширины пика
		search_window = max(width, 0.01)
		min_freq = peak_freq - search_window
		max_freq = peak_freq + search_window

		# Ищем левую границу
		left_idx = peak_idx
		i = peak_idx - 1
		while i >= 0 and freqs[i] >= min_freq:
			if self._is_opposite_extremum(s_values, i, peak_type):
				# Нашли противоположный экстремум - граница перед ним
				left_idx = i + 1
				break
			left_idx = i
			i -= 1

		# Ищем правую границу
		right_idx = peak_idx
		i = peak_idx + 1
		while i < len(freqs) and freqs[i] <= max_freq:
			if self._is_opposite_extremum(s_values, i, peak_type):
				# Нашли противоположный экстремум - граница перед ним
				right_idx = i - 1
				break
			right_idx = i
			i += 1

		# Гарантируем, что границы не схлопнулись в одну точку
		if left_idx >= peak_idx:
			left_idx = max(0, peak_idx - 1)
		if right_idx <= peak_idx:
			right_idx = min(len(freqs) - 1, peak_idx + 1)
		
		# Финальная проверка: границы должны быть разные
		if left_idx == right_idx:
			if left_idx > 0:
				left_idx -= 1
			if right_idx < len(freqs) - 1:
				right_idx += 1

		return left_idx, right_idx

	def execute(self):
		self.result = {"trajectories": []}

		for traj in self.input_trajectories:
			fields = list(traj.get("fields", []))
			freqs = list(traj.get("freq", []))
			widths = list(traj.get("width", []))
			peak_type = traj.get("peak_type_fixed") or traj.get("peak_type") or config_physics.PEAK_TYPE

			lower_bounds = []
			upper_bounds = []

			for field, peak_freq, width in zip(fields, freqs, widths):
				cut_freqs, cut_values = self._get_cut_at_field(field)
				left_idx, right_idx = self._find_bounds_for_peak(
					cut_freqs,
					cut_values,
					peak_freq,
					width,
					peak_type
				)
				lower_bounds.append(float(cut_freqs[left_idx]))
				upper_bounds.append(float(cut_freqs[right_idx]))

			# Копируем все поля из исходной траектории и добавляем границы
			output_traj = dict(traj)
			output_traj["lower_bound"] = lower_bounds
			output_traj["upper_bound"] = upper_bounds
			
			self.result["trajectories"].append(output_traj)

	def prepare_for_visualization(self):
		boundary_points = []
		peak_points = []
		
		for traj in self.result.get("trajectories", []):
			fields = traj.get("fields", [])
			freqs = traj.get("freq", [])
			lower = traj.get("lower_bound", [])
			upper = traj.get("upper_bound", [])
			
			for field, freq, low, high in zip(fields, freqs, lower, upper):
				peak_points.append((field, freq))
				boundary_points.append((field, low))
				boundary_points.append((field, high))

		# Создаем новый словарь только с контурными данными
		self.data_for_visualization = {
			"x": self.data["x"],
			"y": self.data["y"],
			"z": self.data["z"],
			"xlabel": self.data.get("xlabel", "X-axis"),
			"ylabel": self.data.get("ylabel", "Y-axis"),
			"zlabel": self.data.get("zlabel", "Z-axis"),
			"title": self.data.get("title", "Contour Plot"),
			"boundary_points": boundary_points,
			"peak_points": peak_points
		}

	def validate(self):
		validator = VContourfBounds(
			self.stage_name,
			self.data_for_visualization,
			boundary_points=self.data_for_visualization.get("boundary_points", []),
			peak_points=self.data_for_visualization.get("peak_points", [])
		)
		plt.show()
		return validator.get_params().get("Validation")
