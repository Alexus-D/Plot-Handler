from typing import Dict, List, Optional, Tuple

from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt

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
				("done", "Done"),
			]

		super().__init__(stage_name, plot_data, buttons, clear_button, save_button)

		self._peak_point = None
		self._width_points = []
		self._plateau_point = None
		self._peak_type_locked = False
		self._highlight_points = list(plot_data.get("highlight_points", []))
		self._half_height_line = None
		self._done = False
		if self._highlight_points:
			self._redraw_markers()

	def _set_mode(self, mode):
		if mode == "done":
			print("[DEBUG SPeakParams] КНОПКА DONE НАЖАТА!")
			self._done = True
			# НЕ закрываем окно здесь - это блокирует plt.pause()
			# Окно будет закрыто после выхода из цикла ожидания
			return
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
		print(f"[DEBUG] mouse_on_click вызван")
		print(f"[DEBUG] event.inaxes: {event.inaxes}")
		print(f"[DEBUG] event.xdata: {event.xdata}, event.ydata: {event.ydata}")
		
		if event.inaxes is None:
			print(f"[DEBUG] event.inaxes is None, пропускаю")
			return
		if event.xdata is None or event.ydata is None:
			print(f"[DEBUG] xdata или ydata is None, пропускаю")
			return
		
		# Проверяем, что клик был по правильной оси (не по кнопке)
		plot_axis = self.plotter.get_axis_for_marker()
		print(f"[DEBUG] plot_axis: {plot_axis}")
		print(f"[DEBUG] event.inaxes == plot_axis: {event.inaxes == plot_axis}")
		
		if event.inaxes != plot_axis:
			print(f"[DEBUG] Клик не по графику, а по другой оси (возможно, кнопка), пропускаю")
			return

		if self.mode == "select_peak":
			self._peak_point = (event.xdata, event.ydata)
			self.params["peak_freq"] = event.xdata
			self.params["peak_value"] = event.ydata
			print(f"Peak selected: freq={event.xdata:.6e}, value={event.ydata:.6e}")
			self._infer_peak_type()
			print(f"Peak type: {self.params.get('peak_type')}")
			self._update_half_height_line()
			self._redraw_markers()
		elif self.mode == "select_width":
			self._width_points.append((event.xdata, event.ydata))
			print(f"Width point {len(self._width_points)} selected: freq={event.xdata:.6e}, value={event.ydata:.6e}")
			if len(self._width_points) >= 2:
				width = abs(self._width_points[1][0] - self._width_points[0][0])
				self.params["peak_width"] = width
				print(f"Peak width: {width:.6e}")
				self._width_points = self._width_points[:2]
			self._redraw_markers()
			self._restore_half_height_line()
		elif self.mode == "select_plateau":
			self._plateau_point = (event.xdata, event.ydata)
			self.params["plateau"] = event.ydata
			print(f"Plateau selected: freq={event.xdata:.6e}, value={event.ydata:.6e}")
			self._infer_peak_type()
			print(f"Peak type: {self.params.get('peak_type')}")
			self._update_half_height_line()
			self._redraw_markers()
			self._restore_half_height_line()
			self._restore_half_height_line()

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

	def _update_half_height_line(self):
		"""Рисует горизонтальную линию на полувысоте пика (для данных в дБ)."""
		print(f"[DEBUG] _update_half_height_line() вызван")
		peak_value = self.params.get("peak_value")
		plateau = self.params.get("plateau")
		print(f"[DEBUG] peak_value={peak_value}, plateau={plateau}")
		
		if peak_value is None or plateau is None:
			print(f"[DEBUG] Пропускаю: один из параметров None")
			return
		
		# Переводим из дБ в линейный масштаб
		peak_linear = 10 ** (peak_value / 20.0)
		plateau_linear = 10 ** (plateau / 20.0)
		print(f"[DEBUG] Линейные значения: peak={peak_linear:.6e}, plateau={plateau_linear:.6e}")
		
		# Вычисляем полувысоту в линейном масштабе
		half_height_linear = (peak_linear + plateau_linear) / 2.0
		
		# Переводим обратно в дБ
		import numpy as np
		half_height = 20.0 * np.log10(half_height_linear)
		print(f"[DEBUG] Полувысота (дБ): {half_height:.6e}")
		
		# Удаляем старую линию, если была
		if self._half_height_line is not None:
			print(f"[DEBUG] Удаляю старую линию")
			try:
				self._half_height_line.remove()
			except Exception as e:
				print(f"[DEBUG] Ошибка при удалении: {e}")
		
		# Рисуем новую линию
		axis = self.plotter.get_axis_for_marker()
		print(f"[DEBUG] Ось получена: {axis}")
		print(f"[DEBUG] Текущие лимиты Y: {axis.get_ylim()}")
		self._half_height_line = axis.axhline(y=half_height, color='red', linestyle='--', linewidth=2, alpha=0.9, zorder=1000)
		print(f"[DEBUG] Линия создана: {self._half_height_line}")
		print(f"[DEBUG] Линия в списке lines: {self._half_height_line in axis.lines}")
		print(f"Half-height line drawn at: {half_height:.6e} dB")
		
		# Перерисовываем
		print(f"[DEBUG] Вызываю draw() и flush_events()")
		self.plotter.get_figure().canvas.draw()
		self.plotter.get_figure().canvas.flush_events()
		print(f"[DEBUG] Перерисовка завершена")

	def _restore_half_height_line(self):
		"""Восстанавливает линию полувысоты, если она была удалена при перерисовке."""
		print(f"[DEBUG] _restore_half_height_line() вызван")
		peak_value = self.params.get("peak_value")
		plateau = self.params.get("plateau")
		
		if peak_value is None or plateau is None:
			print(f"[DEBUG] Пропускаю восстановление: параметры не заданы")
			return
		
		axis = self.plotter.get_axis_for_marker()
		line_exists = self._half_height_line is not None and self._half_height_line in axis.lines
		print(f"[DEBUG] Линия существует в axis.lines: {line_exists}")
		print(f"[DEBUG] self._half_height_line: {self._half_height_line}")
		print(f"[DEBUG] Количество линий на оси: {len(axis.lines)}")
		
		if self._half_height_line is not None and self._half_height_line not in self.plotter.get_axis_for_marker().lines:
			# Линия была удалена, восстанавливаем
			print(f"[DEBUG] Линия была удалена, восстанавливаю")
			import numpy as np
			peak_linear = 10 ** (peak_value / 20.0)
			plateau_linear = 10 ** (plateau / 20.0)
			half_height_linear = (peak_linear + plateau_linear) / 2.0
			half_height = 20.0 * np.log10(half_height_linear)
			
			self._half_height_line = axis.axhline(y=half_height, color='red', linestyle='--', linewidth=2, alpha=0.9, zorder=1000)
			print(f"[DEBUG] Линия восстановлена: {self._half_height_line}")
			self.plotter.get_figure().canvas.draw()
			self.plotter.get_figure().canvas.flush_events()
		else:
			print(f"[DEBUG] Восстановление не требуется")

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
	
	def is_done(self) -> bool:
		"""Проверяет, была ли нажата кнопка Done."""
		return self._done
