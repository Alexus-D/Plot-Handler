import numpy as np
import matplotlib.pyplot as plt

import config_physics
import utils.algorithms as alg

from .IEPeakWatcher import IEPeakWatcher


class IEPeakWatcherHierarchy(IEPeakWatcher):
	"""
	IEPeakWatcher with hierarchical tracking:
	each next trajectory is tracked strictly below the previous one.
	The upper search bound at current field is the first opposite extremum
	above the trajectory (minimum for maxima, maximum for minima). If it
	cannot be determined, fallback to midpoint between peaks on previous field.
	The lower search bound is defined analogously below the trajectory
	(with midpoint fallback to the next trajectory on previous field).
	"""

	def _get_midpoint_upper_bound(self, traj_idx: int, step_idx: int, traj_data: dict):
		if traj_idx <= 0 or step_idx <= 0:
			return None

		prev_traj = self.result.get("trajectories", [])[traj_idx - 1]
		prev_fields = prev_traj.get("fields", [])
		prev_freqs = prev_traj.get("freq", [])

		if not prev_fields or not prev_freqs:
			return None

		if step_idx - 1 >= len(traj_data.get("fields", [])):
			return None

		current_prev_freq = traj_data["freq"][step_idx - 1]
		prev_field = traj_data["fields"][step_idx - 1]

		prev_fields_arr = np.asarray(prev_fields)
		prev_idx = int(np.abs(prev_fields_arr - prev_field).argmin())
		if prev_idx >= len(prev_freqs):
			return None

		prev_freq = prev_freqs[prev_idx]
		return (prev_freq + current_prev_freq) / 2.0

	def _get_midpoint_lower_bound(self, traj_idx: int, step_idx: int, traj_data: dict):
		if traj_idx >= len(self.trajectories) - 1 or step_idx <= 0:
			return None

		next_traj = self.result.get("trajectories", [])[traj_idx + 1]
		next_fields = next_traj.get("fields", [])
		next_freqs = next_traj.get("freq", [])

		if not next_fields or not next_freqs:
			return None

		if step_idx - 1 >= len(traj_data.get("fields", [])):
			return None

		current_prev_freq = traj_data["freq"][step_idx - 1]
		next_field = traj_data["fields"][step_idx - 1]

		next_fields_arr = np.asarray(next_fields)
		next_idx = int(np.abs(next_fields_arr - next_field).argmin())
		if next_idx >= len(next_freqs):
			return None

		next_freq = next_freqs[next_idx]
		return (next_freq + current_prev_freq) / 2.0

	def _find_first_opposite_extremum(self, freqs, s_values, start_freq, peak_type, direction: str):
		if start_freq is None:
			return None

		if direction == "above":
			indices = np.where(freqs > start_freq)[0]
			if len(indices) < 3:
				return None
			start_idx = indices[0]
			segment_freqs = freqs[start_idx:]
			segment_values = s_values[start_idx:]
			step_range = range(1, len(segment_values) - 1)
		else:
			indices = np.where(freqs < start_freq)[0]
			if len(indices) < 3:
				return None
			end_idx = indices[-1] + 1
			segment_freqs = freqs[:end_idx]
			segment_values = s_values[:end_idx]
			step_range = range(len(segment_values) - 2, 0, -1)

		for i in step_range:
			prev_v = segment_values[i - 1]
			curr_v = segment_values[i]
			next_v = segment_values[i + 1]

			if peak_type == "maximum":
				if prev_v > curr_v < next_v:
					return segment_freqs[i]
			else:
				if prev_v < curr_v > next_v:
					return segment_freqs[i]

		return None

	def _get_upper_bound(
		self,
		traj_idx: int,
		step_idx: int,
		traj_data: dict,
		freqs,
		s_values,
		current_params: dict,
		fallback_midpoint: bool = True,
	):
		peak_type = current_params.get("peak_type", config_physics.PEAK_TYPE)
		expected_freq = current_params.get("peak_freq")

		opposite_extremum = self._find_first_opposite_extremum(
			freqs,
			s_values,
			expected_freq,
			peak_type,
			"above"
		)
		if opposite_extremum is not None:
			return opposite_extremum

		if fallback_midpoint:
			return self._get_midpoint_upper_bound(traj_idx, step_idx, traj_data)
		return None

	def _get_lower_bound(
		self,
		traj_idx: int,
		step_idx: int,
		traj_data: dict,
		freqs,
		s_values,
		current_params: dict,
		fallback_midpoint: bool = True,
	):
		peak_type = current_params.get("peak_type", config_physics.PEAK_TYPE)
		expected_freq = current_params.get("peak_freq")

		opposite_extremum = self._find_first_opposite_extremum(
			freqs,
			s_values,
			expected_freq,
			peak_type,
			"below"
		)
		if opposite_extremum is not None:
			return opposite_extremum

		if fallback_midpoint:
			return self._get_midpoint_lower_bound(traj_idx, step_idx, traj_data)
		return None

	def interruptible_function(self):
		"""
		Основной цикл отслеживания пиков по всем траекториям
		с ограничением диапазона частот относительно предыдущей траектории.
		"""
		print("[IEPeakWatcherHierarchy] Старт: отслеживание пиков")
		print(f"[DEBUG] interrupted={self.interrupted}, correcting_params={bool(self.correcting_params)}")

		start_idx = self._start_traj_idx
		self._start_traj_idx = 0

		for traj_idx in range(start_idx, len(self.trajectories)):
			start, end = self.trajectories[traj_idx]
			self._current_traj_idx = traj_idx
			print(f"[IEPeakWatcherHierarchy] Траектория {traj_idx+1}: старт")

			if traj_idx >= len(self.result["trajectories"]):
				self.result["trajectories"].append({
					"fields": [],
					"freq": [],
					"magnitude": [],
					"prominence": [],
					"width": [],
					"method": []
				})

			traj_data = self.result["trajectories"][traj_idx]

			self._field_indices = self._get_fields_range(start[0], end[0])
			print(f"[IEPeakWatcherHierarchy] Траектория {traj_idx+1}: шагов {len(self._field_indices)}")

			if self._continue_from_idx is not None:
				start_idx = self._continue_from_idx
				print(f"[DEBUG] Продолжаем с индекса {start_idx}, сбрасываем _continue_from_idx")
				self._continue_from_idx = None
			else:
				start_idx = len(traj_data["fields"])
				print(f"[DEBUG] Начинаем с индекса {start_idx} (текущая длина результатов)")

			if self.correcting_params:
				current_params = self.correcting_params.copy()
				print(f"[DEBUG] Корректирующие параметры: {current_params}")
				self.correcting_params = {}
				print(f"[IEPeakWatcherHierarchy] Траектория {traj_idx+1}: использую корректирующие параметры")
			elif start_idx > 0 and len(traj_data["freq"]) > 0:
				current_params = {
					"peak_freq": traj_data["freq"][-1],
					"peak_width": traj_data["width"][-1],
					"prominence": traj_data["prominence"][-1],
					"peak_type": self.initial_peak_params[traj_idx].get("peak_type", config_physics.PEAK_TYPE)
				}
				print(f"[IEPeakWatcherHierarchy] Траектория {traj_idx+1}: использую последние параметры")
			else:
				current_params = self.initial_peak_params[traj_idx].copy()
				print(f"[IEPeakWatcherHierarchy] Траектория {traj_idx+1}: использую начальные параметры")

			print(f"[DEBUG] Цикл по полям: start_idx={start_idx}, total_indices={len(self._field_indices)}")
			for i in range(start_idx, len(self._field_indices)):
				field_idx = self._field_indices[i]
				field = self.data["x"][field_idx]
				freqs = self.data["y"]
				s_values = self.data["z"][field_idx, :]

				upper_bound = None
				lower_bound = None

				is_first = traj_idx == 0
				is_last = traj_idx == len(self.trajectories) - 1

				if not is_last:
					lower_bound = self._get_lower_bound(
						traj_idx,
						i,
						traj_data,
						freqs,
						s_values,
						current_params,
						fallback_midpoint=not is_first,
					)
				elif is_last:
					lower_bound = self._get_lower_bound(
						traj_idx,
						i,
						traj_data,
						freqs,
						s_values,
						current_params,
						fallback_midpoint=False,
					)

				if not is_first:
					upper_bound = self._get_upper_bound(
						traj_idx,
						i,
						traj_data,
						freqs,
						s_values,
						current_params,
						fallback_midpoint=True,
					)
				else:
					upper_bound = self._get_upper_bound(
						traj_idx,
						i,
						traj_data,
						freqs,
						s_values,
						current_params,
						fallback_midpoint=False,
					)

				mask = np.ones_like(freqs, dtype=bool)
				if lower_bound is not None:
					mask &= freqs > lower_bound
				if upper_bound is not None:
					mask &= freqs < upper_bound

				freqs_for_search = freqs[mask]
				s_values_for_search = s_values[mask]

				print(
					f"[IEPeakWatcherHierarchy] Траектория {traj_idx+1}: поиск пика, i={i}, поле={field}, "
					f"lower_bound={lower_bound}, upper_bound={upper_bound}"
				)

				try:
					expected_freq = current_params["peak_freq"]
					if len(freqs_for_search) == 0:
						raise ValueError("No data points in constrained search window")

					freq_step = abs(freqs_for_search[1] - freqs_for_search[0]) if len(freqs_for_search) > 1 else 0.001
					if upper_bound is not None and expected_freq >= upper_bound:
						expected_freq = upper_bound - freq_step
					if lower_bound is not None and expected_freq <= lower_bound:
						expected_freq = lower_bound + freq_step

					peak = alg.find_peak(
						freqs_for_search, s_values_for_search,
						expected_freq=expected_freq,
						expected_width=current_params.get("peak_width", current_params.get("width", 0.1)),
						expected_prominence=current_params["prominence"],
						peak_type=current_params.get("peak_type", config_physics.PEAK_TYPE)
					)
				except Exception as e:
					print(f"Peak finding failed at field {field}: {e}")
					self.interrupted = True
					self.skip_delete_wrong_results = True
					self._continue_from_idx = i
					self._start_traj_idx = traj_idx
					return self.result

				traj_data["fields"].append(field)
				traj_data["freq"].append(peak["freq"])
				traj_data["magnitude"].append(peak["magnitude"])
				traj_data["prominence"].append(peak["prominence"])
				traj_data["width"].append(peak["width"])
				traj_data["method"].append(peak["method"])

				current_params["peak_freq"] = peak["freq"]
				current_params["peak_width"] = peak["width"]
				current_params["prominence"] = peak["prominence"]

				print(f"[DEBUG] Обновление визуализации, текущих точек: {len(traj_data['fields'])}")
				self._update_line(self.result)

				plt.pause(0.001)

				if self.interrupted:
					print(f"[IEPeakWatcherHierarchy] Траектория {traj_idx+1}: прервано пользователем")
					print(f"[DEBUG] Прерывание: устанавливаем _continue_from_idx={i + 1}")
					self._continue_from_idx = i + 1
					self._start_traj_idx = traj_idx
					return self.result

			if not self._validate_trajectory_end():
				print(f"[IEPeakWatcherHierarchy] Траектория {traj_idx+1}: отклонена пользователем")
				self.interrupted = True
				self._start_traj_idx = traj_idx
				return self.result
			print(f"[IEPeakWatcherHierarchy] Траектория {traj_idx+1}: подтверждена")

			self._continue_from_idx = None

		return self.result
