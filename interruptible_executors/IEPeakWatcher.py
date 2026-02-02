import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector

import config_physics
import utils.algorithms as alg

from markers import MPoints
from executors import ECut, EPeakParams, ETrajectories

from .InterruptibleExecutor import InterruptibleExecutor


class IEPeakWatcher(InterruptibleExecutor):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
        self.trajectories = []
        self.initial_peak_params = []
        self._current_traj_idx = 0
        self._field_indices = []
        self._continue_from_idx = None
        self._start_traj_idx = 0  # С какой траектории начинать при возобновлении
        
        # Графики
        self.figure = None
        self.axes = None
        self.ax_contour = None
        self.ax_freq = None
        self.ax_mag = None
        self.ax_width = None
        
        # Линии для графиков параметров
        self._freq_lines = []
        self._mag_lines = []
        self._width_lines = []

    def select_initial_params(self):
        """
        1) Выбор траекторий с помощью ETrajectories
        2) Для каждой траектории: срез в начальной точке + выбор параметров пика
        """
        print("[IEPeakWatcher] Старт: выбор начальных параметров")
        # 1. Выбор траекторий
        traj_executor = ETrajectories(self.data, "Select Trajectories")
        traj_executor.execute_with_validation()
        self.trajectories = traj_executor.get_result()["trajectories"]
        
        if not self.trajectories:
            raise ValueError("No trajectories selected")
        
        # 2. Для каждой траектории выбираем начальные параметры пика
        self.initial_peak_params = []
        
        for i, (start, end) in enumerate(self.trajectories):
            start_field = start[0]
            start_freq = start[1]
            print(f"[IEPeakWatcher] Траектория {i+1}: срез в поле {start_field}")
            
            # Делаем срез в начальной точке траектории
            cut_executor = ECut(
                self.data,
                f"Cut for trajectory {i+1}",
                axis="x",
                initial_params={"cut_value": start_field}
            )
            cut_executor.execute()
            cut_data = cut_executor.get_result()
            
            # Добавляем highlight точки начала траектории
            cut_data["highlight_points"] = [(start_freq, self._get_z_at_point(start_field, start_freq))]
            
            # Выбираем параметры пика на срезе
            peak_executor = EPeakParams(cut_data, f"Peak params for trajectory {i+1}")
            print(f"[IEPeakWatcher] Траектория {i+1}: выбор параметров пика")
            peak_executor.execute_with_validation()
            peak_params = peak_executor.get_result()
            
            self.initial_peak_params.append(peak_params)
        
        # Инициализируем структуру результата
        self.result = {"trajectories": []}
        print("[IEPeakWatcher] Готово: начальные параметры выбраны")

    def _get_z_at_point(self, field, freq):
        """Получает значение Z в точке (field, freq)."""
        fields = self.data["x"]
        freqs = self.data["y"]
        z = self.data["z"]
        
        field_idx = np.abs(fields - field).argmin()
        freq_idx = np.abs(freqs - freq).argmin()
        
        return z[field_idx, freq_idx]

    def _setup_execution_plot(self, result):
        """
        Создаёт окно с 4 графиками: контурный + 3 графика параметров.
        """
        print("[IEPeakWatcher] Инициализация окна визуализации")
        plt.ion()
        
        self.figure, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Левый верхний - контурный график
        self.ax_contour = self.axes[0, 0]
        self._draw_contour(self.ax_contour)
        
        # Правый верхний - частоты
        self.ax_freq = self.axes[0, 1]
        self.ax_freq.set_xlabel(self.data.get("xlabel", "Field"))
        self.ax_freq.set_ylabel(self.data.get("ylabel", "Frequency"))
        self.ax_freq.set_title("Peak Frequencies")
        self.ax_freq.grid(True, alpha=0.3)
        
        # Левый нижний - магнитуды
        self.ax_mag = self.axes[1, 0]
        self.ax_mag.set_xlabel(self.data.get("xlabel", "Field"))
        self.ax_mag.set_ylabel(self.data.get("zlabel", "Magnitude"))
        self.ax_mag.set_title("Peak Magnitudes")
        self.ax_mag.grid(True, alpha=0.3)
        
        # Правый нижний - ширины
        self.ax_width = self.axes[1, 1]
        self.ax_width.set_xlabel(self.data.get("xlabel", "Field"))
        self.ax_width.set_ylabel("Width")
        self.ax_width.set_title("Peak Widths")
        self.ax_width.grid(True, alpha=0.3)
        
        # Маркер для контурного графика
        self.marker = MPoints(self.ax_contour)
        
        # Инициализируем линии для графиков параметров
        self._freq_lines = []
        self._mag_lines = []
        self._width_lines = []
        
        colors = plt.cm.tab10.colors
        for i in range(len(self.trajectories)):
            color = colors[i % len(colors)]
            freq_line, = self.ax_freq.plot([], [], 'o-', color=color, label=f"Traj {i+1}", markersize=3)
            mag_line, = self.ax_mag.plot([], [], 'o-', color=color, label=f"Traj {i+1}", markersize=3)
            width_line, = self.ax_width.plot([], [], 'o-', color=color, label=f"Traj {i+1}", markersize=3)
            self._freq_lines.append(freq_line)
            self._mag_lines.append(mag_line)
            self._width_lines.append(width_line)
        
        self.ax_freq.legend(loc='upper right', fontsize=8)
        self.ax_mag.legend(loc='upper right', fontsize=8)
        self.ax_width.legend(loc='upper right', fontsize=8)
        
        # Кнопка прерывания
        self._interrupt_btn = self._make_interrupt_button(self.figure)
        
        plt.tight_layout()
        self.figure.subplots_adjust(bottom=0.1)
        
        # Начальное обновление
        self._update_line(result)
        print("[IEPeakWatcher] Окно визуализации готово")

    def _draw_contour(self, ax):
        """Рисует контурный график на указанной оси."""
        x = self.data.get("x")
        y = self.data.get("y")
        z = self.data.get("z")
        
        X, Y = np.meshgrid(x, y)
        Z = np.array(z).transpose()
        
        contour = ax.contourf(X, Y, Z, cmap='viridis', levels=50)
        cbar = self.figure.colorbar(contour, ax=ax)
        cbar.set_label(self.data.get("zlabel", "Z-axis"))
        ax.set_title(self.data.get("title", "Contour Plot"))
        ax.set_xlabel(self.data.get("xlabel", "X-axis"))
        ax.set_ylabel(self.data.get("ylabel", "Y-axis"))

    def _get_fields_range(self, start_field, end_field):
        """
        Возвращает список индексов полей от start_field до end_field.
        """
        fields = self.data["x"]
        
        start_idx = np.abs(fields - start_field).argmin()
        end_idx = np.abs(fields - end_field).argmin()
        
        if start_idx <= end_idx:
            indices = list(range(start_idx, end_idx + 1))
        else:
            indices = list(range(start_idx, end_idx - 1, -1))
        
        return indices

    def interruptible_function(self):
        """
        Основной цикл отслеживания пиков по всем траекториям.
        """
        print("[IEPeakWatcher] Старт: отслеживание пиков")
        print(f"[DEBUG] interrupted={self.interrupted}, correcting_params={bool(self.correcting_params)}")
        
        # Начинаем с сохраненной траектории (если возобновляем после прерывания)
        start_idx = self._start_traj_idx
        self._start_traj_idx = 0  # Сбрасываем после использования
        
        for traj_idx in range(start_idx, len(self.trajectories)):
            start, end = self.trajectories[traj_idx]
            self._current_traj_idx = traj_idx
            print(f"[IEPeakWatcher] Траектория {traj_idx+1}: старт")
            print(f"[DEBUG] Траектория {traj_idx+1}: interrupted={self.interrupted}")
            
            # Инициализируем структуру для траектории, если её нет
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
            
            # Получаем индексы полей для этой траектории
            self._field_indices = self._get_fields_range(start[0], end[0])
            print(f"[IEPeakWatcher] Траектория {traj_idx+1}: шагов {len(self._field_indices)}")
            
            # Определяем, с какого индекса начинать
            if self._continue_from_idx is not None:
                start_idx = self._continue_from_idx
                print(f"[DEBUG] Продолжаем с индекса {start_idx}, сбрасываем _continue_from_idx")
                self._continue_from_idx = None
            else:
                start_idx = len(traj_data["fields"])
                print(f"[DEBUG] Начинаем с индекса {start_idx} (текущая длина результатов)")
            
            # Получаем начальные параметры
            if self.correcting_params:
                current_params = self.correcting_params.copy()
                print(f"[DEBUG] Корректирующие параметры: {current_params}")
                self.correcting_params = {}
                print(f"[IEPeakWatcher] Траектория {traj_idx+1}: использую корректирующие параметры")
            elif start_idx > 0 and len(traj_data["freq"]) > 0:
                # Используем последние найденные параметры
                current_params = {
                    "peak_freq": traj_data["freq"][-1],
                    "peak_width": traj_data["width"][-1],
                    "prominence": traj_data["prominence"][-1],
                    "peak_type": self.initial_peak_params[traj_idx].get("peak_type", config_physics.PEAK_TYPE)
                }
                print(f"[IEPeakWatcher] Траектория {traj_idx+1}: использую последние параметры")
            else:
                current_params = self.initial_peak_params[traj_idx].copy()
                print(f"[IEPeakWatcher] Траектория {traj_idx+1}: использую начальные параметры")
            
            # Цикл по полям
            print(f"[DEBUG] Цикл по полям: start_idx={start_idx}, total_indices={len(self._field_indices)}")
            for i in range(start_idx, len(self._field_indices)):
                field_idx = self._field_indices[i]
                field = self.data["x"][field_idx]
                freqs = self.data["y"]
                s_values = self.data["z"][field_idx, :]
                print(f"[IEPeakWatcher] Траектория {traj_idx+1}: поиск пика, i={i}, поле={field}")
                
                # Ищем пик
                try:
                    peak = alg.find_peak(
                        freqs, s_values,
                        expected_freq=current_params["peak_freq"],
                        expected_width=current_params.get("peak_width", current_params.get("width", 0.1)),
                        expected_prominence=current_params["prominence"],
                        peak_type=current_params.get("peak_type", config_physics.PEAK_TYPE)
                    )
                except Exception as e:
                    print(f"Peak finding failed at field {field}: {e}")
                    self.interrupted = True
                    self.skip_delete_wrong_results = True
                    self._continue_from_idx = i
                    self._start_traj_idx = traj_idx  # Сохраняем индекс траектории
                    return self.result
                
                # Сохраняем результат
                traj_data["fields"].append(field)
                traj_data["freq"].append(peak["freq"])
                traj_data["magnitude"].append(peak["magnitude"])
                traj_data["prominence"].append(peak["prominence"])
                traj_data["width"].append(peak["width"])
                traj_data["method"].append(peak["method"])
                
                # Обновляем параметры для следующего шага
                current_params["peak_freq"] = peak["freq"]
                current_params["peak_width"] = peak["width"]
                current_params["prominence"] = peak["prominence"]
                
                # Обновляем визуализацию
                print(f"[DEBUG] Обновление визуализации, текущих точек: {len(traj_data['fields'])}")
                self._update_line(self.result)
                
                # Пауза 0.1 секунды
                plt.pause(0.001)
                
                # Проверяем прерывание
                if self.interrupted:
                    print(f"[IEPeakWatcher] Траектория {traj_idx+1}: прервано пользователем")
                    print(f"[DEBUG] Прерывание: устанавливаем _continue_from_idx={i + 1}")
                    self._continue_from_idx = i + 1
                    self._start_traj_idx = traj_idx  # Сохраняем индекс траектории для возобновления
                    return self.result
            
            # Конец траектории - валидация
            if not self._validate_trajectory_end():
                print(f"[IEPeakWatcher] Траектория {traj_idx+1}: отклонена пользователем")
                self.interrupted = True
                self._start_traj_idx = traj_idx  # Сохраняем для повторной обработки этой траектории
                return self.result
            print(f"[IEPeakWatcher] Траектория {traj_idx+1}: подтверждена")
            
            # Сбрасываем индексы после успешного завершения траектории
            self._continue_from_idx = None
        
        return self.result

    def _update_line(self, points):
        """
        Обновляет визуализацию: маркеры на контурном графике и графики параметров.
        Параметр points используется как result для совместимости с базовым классом.
        """
        result = points if isinstance(points, dict) else self.result
        
        # Собираем все найденные точки из всех траекторий
        all_points = []
        for idx, traj_data in enumerate(result.get("trajectories", [])):
            fields = traj_data.get("fields", [])
            freqs = traj_data.get("freq", [])
            print(f"[DEBUG] _update_line: траектория {idx+1}, точек={len(fields)}")
            for f, freq in zip(fields, freqs):
                all_points.append((f, freq))
        
        print(f"[DEBUG] _update_line: всего точек для отображения={len(all_points)}")
        # Обновляем маркеры на контурном графике
        self.marker.update_ticks(all_points)
        
        # Обновляем графики параметров
        self._update_param_plots(result)
        
        # Перерисовываем
        if self.figure is not None:
            print(f"[DEBUG] Вызов draw_idle и flush_events")
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
        else:
            print(f"[DEBUG] WARNING: self.figure is None!")

    def _update_param_plots(self, result):
        """Обновляет графики параметров (freq, magnitude, width)."""
        for i, traj_data in enumerate(result.get("trajectories", [])):
            if i >= len(self._freq_lines):
                break
            
            fields = traj_data.get("fields", [])
            freqs = traj_data.get("freq", [])
            mags = traj_data.get("magnitude", [])
            widths = traj_data.get("width", [])
            
            self._freq_lines[i].set_data(fields, freqs)
            self._mag_lines[i].set_data(fields, mags)
            self._width_lines[i].set_data(fields, widths)
        
        # Автомасштабирование осей
        for ax in [self.ax_freq, self.ax_mag, self.ax_width]:
            ax.relim()
            ax.autoscale_view()

    def delete_wrong_results(self):
        """
        Позволяет пользователю выбрать рамочкой неправильные маркеры 
        и удаляет соответствующие результаты.
        """
        print("[IEPeakWatcher] Выделите рамкой точки для удаления")
        self._selected_rect = None
        self._deletion_confirmed = False
        
        def on_select(eclick, erelease):
            print(f"[DEBUG IEPeakWatcher] RectangleSelector: выделена область")
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])
            self._selected_rect = (x1, x2, y1, y2)
            print(f"[DEBUG IEPeakWatcher] Область: x=[{x1:.2f}, {x2:.2f}], y=[{y1:.2f}, {y2:.2f}]")
        
        # Создаём RectangleSelector
        rect_selector = RectangleSelector(
            self.ax_contour, on_select,
            useblit=True,
            button=[1],
            interactive=True
        )
        
        # Добавляем кнопку подтверждения удаления
        ax_btn = self.figure.add_axes([0.35, 0.01, 0.1, 0.05])
        btn_confirm = Button(ax_btn, 'Delete Selected')
        
        def on_confirm(event):
            print("[DEBUG IEPeakWatcher] КНОПКА DELETE SELECTED НАЖАТА!")
            self._deletion_confirmed = True
        
        btn_confirm.on_clicked(on_confirm)
        
        # Ждём подтверждения
        print("[DEBUG IEPeakWatcher] Ожидание нажатия Delete Selected...")
        while not self._deletion_confirmed:
            plt.pause(0.1)
        print("[IEPeakWatcher] Удаление выбранных точек")
        
        # Удаляем точки, попавшие в рамку
        if self._selected_rect is not None:
            self._remove_points_in_rect(*self._selected_rect)
        
        # Убираем кнопку и селектор
        ax_btn.remove()
        rect_selector.set_active(False)
        
        self.figure.canvas.draw_idle()

    def _remove_points_in_rect(self, x1, x2, y1, y2):
        """
        Удаляет из результатов точки, попавшие в прямоугольник [x1,x2] x [y1,y2].
        """
        print("[IEPeakWatcher] Удаляю точки в выделенной области")
        traj_data = self.result["trajectories"][self._current_traj_idx]
        
        fields = traj_data["fields"]
        freqs = traj_data["freq"]
        
        # Находим индексы точек для удаления
        indices_to_remove = []
        for i, (f, freq) in enumerate(zip(fields, freqs)):
            if x1 <= f <= x2 and y1 <= freq <= y2:
                indices_to_remove.append(i)
        
        # Удаляем с конца, чтобы не сбивались индексы
        for i in reversed(indices_to_remove):
            for key in ["fields", "freq", "magnitude", "prominence", "width", "method"]:
                if key in traj_data and len(traj_data[key]) > i:
                    traj_data[key].pop(i)
        
        # Запоминаем, с какого индекса продолжить
        if indices_to_remove:
            self._continue_from_idx = min(indices_to_remove)
            print(f"[IEPeakWatcher] Продолжу с индекса {self._continue_from_idx}")
        
        # Обновляем визуализацию
        self._update_line(self.result)

    def select_correcting_params(self):
        """
        Позволяет пользователю выбрать новые параметры пика 
        для продолжения отслеживания с места ошибки.
        """
        print("[IEPeakWatcher] Выбор корректирующих параметров")
        traj_data = self.result["trajectories"][self._current_traj_idx]
        
        # Определяем поле, с которого продолжить
        if self._continue_from_idx is not None and self._continue_from_idx < len(self._field_indices):
            continue_field_idx = self._field_indices[self._continue_from_idx]
            continue_field = self.data["x"][continue_field_idx]
        elif len(traj_data["fields"]) > 0:
            # Берём следующее поле после последнего успешного
            last_field = traj_data["fields"][-1]
            last_field_idx = np.abs(self.data["x"] - last_field).argmin()
            # Определяем направление
            if len(self._field_indices) > 1:
                direction = 1 if self._field_indices[1] > self._field_indices[0] else -1
            else:
                direction = 1
            continue_field_idx = last_field_idx + direction
            continue_field = self.data["x"][continue_field_idx]
        else:
            # Начинаем с начала траектории
            start, _ = self.trajectories[self._current_traj_idx]
            continue_field = start[0]
        
        # Делаем срез в этом поле
        cut_executor = ECut(
            self.data,
            "Correcting params cut",
            axis="x",
            initial_params={"cut_value": continue_field}
        )
        print(f"[IEPeakWatcher] Срез для корректировки, поле={continue_field}")
        cut_executor.execute()
        cut_data = cut_executor.get_result()
        
        # Добавляем highlight для последней успешной частоты
        if len(traj_data["freq"]) > 0:
            last_freq = traj_data["freq"][-1]
            # Получаем значение на новом срезе
            freq_idx = np.abs(self.data["y"] - last_freq).argmin()
            field_idx = np.abs(self.data["x"] - continue_field).argmin()
            last_mag = self.data["z"][field_idx, freq_idx]
            cut_data["highlight_points"] = [(last_freq, last_mag)]
        
        # Даём выбрать новые параметры
        # НЕ выключаем интерактивный режим - это блокирует новое окно
        print("[DEBUG] Создаём EPeakParams для выбора корректирующих параметров")
        
        peak_executor = EPeakParams(cut_data, "Select correcting peak params")
        print("[IEPeakWatcher] Выберите новые параметры пика")
        print("[DEBUG] Вызываем execute_with_validation()")
        peak_executor.execute_with_validation()
        print("[DEBUG] execute_with_validation() завершился")
        
        # Восстанавливаем видимость основного окна
        if self.figure is not None:
            print("[DEBUG] Восстанавливаем основное окно")
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
        
        # Сохраняем скорректированные параметры
        self.correcting_params = peak_executor.get_result()
        print("[IEPeakWatcher] Корректирующие параметры сохранены")
        print(f"[DEBUG] Сохранённые параметры: {self.correcting_params}")

    def _validate_trajectory_end(self):
        """
        Показывает кнопки подтверждения в конце траектории.
        Возвращает True если пользователь подтвердил, False если отклонил.
        """
        print("[IEPeakWatcher] Подтвердите или отклоните траекторию")
        self._trajectory_confirmed = None
        
        # Создаём кнопки Confirm и Reject
        ax_confirm = self.figure.add_axes([0.35, 0.01, 0.1, 0.05])
        btn_confirm = Button(ax_confirm, 'Confirm')
        
        ax_reject = self.figure.add_axes([0.46, 0.01, 0.1, 0.05])
        btn_reject = Button(ax_reject, 'Reject')
        
        def on_confirm(event):
            print("[DEBUG IEPeakWatcher] КНОПКА CONFIRM ТРАЕКТОРИИ НАЖАТА!")
            self._trajectory_confirmed = True
        
        def on_reject(event):
            print("[DEBUG IEPeakWatcher] КНОПКА REJECT ТРАЕКТОРИИ НАЖАТА!")
            self._trajectory_confirmed = False
        
        btn_confirm.on_clicked(on_confirm)
        btn_reject.on_clicked(on_reject)
        
        # Ждём нажатия
        print("[DEBUG IEPeakWatcher] Ожидание нажатия Confirm/Reject...")
        while self._trajectory_confirmed is None:
            plt.pause(0.1)
        print(f"[IEPeakWatcher] Решение по траектории получено: {self._trajectory_confirmed}")
        
        # Убираем кнопки
        ax_confirm.remove()
        ax_reject.remove()
        
        self.figure.canvas.draw_idle()
        
        return self._trajectory_confirmed

    def _close_execution_plot(self):
        """Закрывает окно с графиками."""
        print("[IEPeakWatcher] Закрытие окна визуализации")
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
        plt.ioff()

    def validate(self):
        """Валидация результата - всегда True, т.к. валидация происходит интерактивно."""
        return True

    def prepare_for_visualization(self):
        """Подготовка данных для визуализации."""
        self.data_for_visualization = self.result