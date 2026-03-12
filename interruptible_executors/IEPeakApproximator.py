import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector

from markers import MPoints
from executors import ECut, EPeakParams
from utils.phys_models import Fano, Lorentzian
import utils.algorithms as alg

from .InterruptibleExecutor import InterruptibleExecutor


class IEPeakApproximator(InterruptibleExecutor):
    """
    Аппроксимирует пики моделью Fano или Lorentzian по данным от IEPeakWatcher.
    
    Входные данные (data):
        - x, y, z: контурные данные
        - trajectories: результат IEPeakWatcher с полями fields, freq, width, magnitude, prominence
    
    Выходные данные (result):
        - trajectories: список словарей с параметрами фита (fields, x0, gamma, A, y0, q, model)
    """
    
    SUPPORTED_MODELS = {
        "Fano": Fano,
        "Lorentzian": Lorentzian
    }
    
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None, save_figure_path: str = None):
        super().__init__(data, stage_name, initial_params, save_figure_path)
        
        # Параметры модели
        initial_params = initial_params or {}
        model_name = initial_params.get("model", "Fano")
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}. Use one of {list(self.SUPPORTED_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_func = self.SUPPORTED_MODELS[model_name]
        self.fit_window_multiplier = initial_params.get("fit_window_multiplier", 0.5)
        
        # Траектории из входных данных
        self.input_trajectories = data.get("trajectories", [])
        if not self.input_trajectories:
            raise ValueError("No trajectories in input data. Run IEPeakWatcher first.")
        
        # Границы области фита для каждой траектории: [(left_shift, right_shift), ...]
        # Если задано, то окно фита будет (x0 - left_shift, x0 + right_shift)
        self.fit_windows = initial_params.get("fit_windows", None)
        if self.fit_windows is not None:
            if len(self.fit_windows) != len(self.input_trajectories):
                raise ValueError(
                    f"fit_windows length ({len(self.fit_windows)}) must match "
                    f"number of trajectories ({len(self.input_trajectories)})"
                )
            print(f"[IEPeakApproximator] Используются заданные границы фита: {self.fit_windows}")
        else:
            print(f"[IEPeakApproximator] Используется fit_window_multiplier={self.fit_window_multiplier}")
        
        self._current_traj_idx = 0
        self._current_point_idx = 0
        self._continue_from_idx = None
        self._start_traj_idx = 0  # С какой траектории начинать при возобновлении
        
        # Текущие данные среза для отображения
        self._current_cut_freqs = None
        self._current_cut_values = None
        self._current_fit_freqs = None
        self._current_fit_values = None
        
        # Графики
        self.figure = None
        self.axes = None
        self.ax_contour = None
        self.ax_x0 = None
        self.ax_cut = None
        self.ax_gamma = None
        
        # Линии для графиков параметров
        self._x0_lines = []
        self._gamma_lines = []
        
        # Линии для графика среза
        self._cut_data_line = None
        self._cut_fit_line = None

    def select_initial_params(self):
        """
        Начальные параметры берутся автоматически из данных IEPeakWatcher.
        Не требует интерактивного выбора.
        """
        print("[IEPeakApproximator] Начальные параметры берутся из входных данных IEPeakWatcher")
        self.result = {"trajectories": []}

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
        Создаёт окно с 4 графиками: контурный, x0(field), срез+фит, gamma(field).
        """
        print("[IEPeakApproximator] Инициализация окна визуализации")
        plt.ion()
        
        self.figure, self.axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Левый верхний - контурный график
        self.ax_contour = self.axes[0, 0]
        self._draw_contour(self.ax_contour)
        
        # Правый верхний - x0 (резонансная частота)
        self.ax_x0 = self.axes[0, 1]
        self.ax_x0.set_xlabel(self.data.get("xlabel", "Field"))
        self.ax_x0.set_ylabel(self.data.get("ylabel", "Frequency"))
        self.ax_x0.set_title("Resonance Frequency (x0)")
        self.ax_x0.grid(True, alpha=0.3)
        
        # Левый нижний - текущий срез + фит
        self.ax_cut = self.axes[1, 0]
        self.ax_cut.set_xlabel(self.data.get("ylabel", "Frequency"))
        self.ax_cut.set_ylabel(self.data.get("zlabel", "Magnitude"))
        self.ax_cut.set_title("Current Cut + Fit")
        self.ax_cut.grid(True, alpha=0.3)
        
        # Правый нижний - gamma (ширина)
        self.ax_gamma = self.axes[1, 1]
        self.ax_gamma.set_xlabel(self.data.get("xlabel", "Field"))
        self.ax_gamma.set_ylabel("Width (gamma)")
        self.ax_gamma.set_title("Resonance Width (gamma)")
        self.ax_gamma.grid(True, alpha=0.3)
        
        # Маркер для контурного графика
        self.marker = MPoints(self.ax_contour)
        
        # Инициализируем линии для графиков параметров
        self._x0_lines = []
        self._gamma_lines = []
        
        colors = plt.cm.tab10.colors
        for i in range(len(self.input_trajectories)):
            color = colors[i % len(colors)]
            x0_line, = self.ax_x0.plot([], [], 'o-', color=color, label=f"Traj {i+1}", markersize=3)
            gamma_line, = self.ax_gamma.plot([], [], 'o-', color=color, label=f"Traj {i+1}", markersize=3)
            self._x0_lines.append(x0_line)
            self._gamma_lines.append(gamma_line)
        
        self.ax_x0.legend(loc='upper right', fontsize=8)
        self.ax_gamma.legend(loc='upper right', fontsize=8)
        
        # Линии для графика среза
        self._cut_data_line, = self.ax_cut.plot([], [], 'o', markersize=3, label='Data', alpha=0.7)
        self._cut_fit_line, = self.ax_cut.plot([], [], '-', linewidth=2, label=f'{self.model_name} fit')
        self.ax_cut.legend(loc='upper right', fontsize=8)
        
        # Кнопка прерывания
        self._interrupt_btn = self._make_interrupt_button(self.figure)
        
        plt.tight_layout()
        self.figure.subplots_adjust(bottom=0.1)
        
        # Начальное обновление
        self._update_line(result)
        print("[IEPeakApproximator] Окно визуализации готово")

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

    def _peak_params_to_fit_p0(self, freq, width, magnitude, prominence):
        """
        Конвертирует параметры пика в начальное приближение для фита.
        
        Returns:
            list: [x0, gamma, q, A, y0] для Fano или [x0, gamma, A, y0] для Lorentzian
        """
        x0 = freq
        gamma = width
        A = prominence
        y0 = magnitude - prominence  # baseline
        
        if self.model_name == "Fano":
            q = 1.0  # начальное приближение для параметра асимметрии
            return [x0, gamma, q, A, y0]
        else:  # Lorentzian
            return [x0, gamma, A, y0]

    def _fit_peak(self, freqs, s_values, p0):
        """
        Выполняет фит пика моделью.
        
        Returns:
            dict: параметры фита {x0, gamma, A, y0, q (только для Fano)}
        """
        # Задаем границы для параметров
        # x0 может меняться не более чем на 10% от ожидаемого значения
        x0_expected = p0[0]
        x0_lower = x0_expected * 0.9
        x0_upper = x0_expected * 1.1
        
        if self.model_name == "Fano":
            # Параметры: [x0, gamma, q, A, y0]
            lower_bounds = [-np.inf, 0, -np.inf, -np.inf, -np.inf]
            upper_bounds = [np.inf, np.inf, np.inf, np.inf, np.inf]
        else:  # Lorentzian
            # Параметры: [x0, gamma, A, y0]
            lower_bounds = [-np.inf, 0, -np.inf, -np.inf]
            upper_bounds = [np.inf, np.inf, np.inf, np.inf]
        
        popt = alg.approximation_by_phys_model(
            freqs, s_values, 
            self.model_func, 
            p0, 
            {"maxfev": 50000, "bounds": (lower_bounds, upper_bounds)}
        )
        
        if self.model_name == "Fano":
            return {
                "x0": popt[0],
                "gamma": popt[1],
                "q": popt[2],
                "A": popt[3],
                "y0": popt[4]
            }
        else:  # Lorentzian
            return {
                "x0": popt[0],
                "gamma": popt[1],
                "q": None,
                "A": popt[2],
                "y0": popt[3]
            }

    def interruptible_function(self):
        """
        Основной цикл аппроксимации пиков по всем траекториям.
        """
        print("[IEPeakApproximator] Старт: аппроксимация пиков")
        
        # Начинаем с сохраненной траектории (если возобновляем после прерывания)
        start_idx = self._start_traj_idx
        self._start_traj_idx = 0  # Сбрасываем после использования
        
        for traj_idx in range(start_idx, len(self.input_trajectories)):
            input_traj = self.input_trajectories[traj_idx]
            self._current_traj_idx = traj_idx
            print(f"[IEPeakApproximator] Траектория {traj_idx+1}: старт")
            
            # Инициализируем структуру для траектории, если её нет
            if traj_idx >= len(self.result["trajectories"]):
                self.result["trajectories"].append({
                    "fields": [],
                    "x0": [],
                    "gamma": [],
                    "A": [],
                    "y0": [],
                    "q": [],
                    "model": self.model_name
                })
            
            traj_data = self.result["trajectories"][traj_idx]
            
            # Определяем, с какого индекса начинать
            if self._continue_from_idx is not None:
                start_idx = self._continue_from_idx
                self._continue_from_idx = None
            else:
                start_idx = len(traj_data["fields"])
            
            input_fields = input_traj["fields"]
            input_freqs = input_traj["freq"]
            input_widths = input_traj["width"]
            input_mags = input_traj["magnitude"]
            input_proms = input_traj["prominence"]
            
            print(f"[IEPeakApproximator] Траектория {traj_idx+1}: {len(input_fields)} точек")
            
            # Цикл по точкам траектории
            for i in range(start_idx, len(input_fields)):
                self._current_point_idx = i
                field = input_fields[i]
                expected_freq = input_freqs[i]
                expected_width = input_widths[i]
                expected_mag = input_mags[i]
                expected_prom = input_proms[i]
                
                print(f"[IEPeakApproximator] Траектория {traj_idx+1}, точка {i+1}: поле={field}")
                
                # Получаем данные среза
                field_idx = np.abs(self.data["x"] - field).argmin()
                freqs = self.data["y"]
                s_values = self.data["z"][field_idx, :]
                
                # Определяем окно фита
                if self.fit_windows is not None:
                    # Используем заданные границы для данной траектории
                    left_shift, right_shift = self.fit_windows[traj_idx]
                    freq_mask = (freqs >= expected_freq - left_shift) & (freqs <= expected_freq + right_shift)
                else:
                    # Используем старый метод с multiplier
                    fit_window = expected_width * self.fit_window_multiplier
                    freq_mask = (freqs >= expected_freq - fit_window/2) & (freqs <= expected_freq + fit_window/2)
                
                fit_freqs = freqs[freq_mask]
                fit_values = s_values[freq_mask]
                
                if len(fit_freqs) < 5:
                    print(f"[IEPeakApproximator] Недостаточно точек в окне фита")
                    self.interrupted = True
                    self.skip_delete_wrong_results = True
                    self._continue_from_idx = i
                    self._start_traj_idx = traj_idx  # Сохраняем индекс траектории
                    return self.result
                
                # Получаем начальные параметры
                if self.correcting_params:
                    p0 = self._peak_params_to_fit_p0(
                        self.correcting_params["peak_freq"],
                        self.correcting_params["peak_width"],
                        self.correcting_params["peak_value"],
                        self.correcting_params["prominence"]
                    )
                    self.correcting_params = {}
                else:
                    p0 = self._peak_params_to_fit_p0(
                        expected_freq, expected_width, expected_mag, expected_prom
                    )
                
                # Выполняем фит
                try:
                    fit_result = self._fit_peak(fit_freqs, fit_values, p0)
                except Exception as e:
                    print(f"[IEPeakApproximator] Фит не удался: {e}")
                    self.interrupted = True
                    self.skip_delete_wrong_results = True
                    self._continue_from_idx = i
                    self._start_traj_idx = traj_idx  # Сохраняем индекс траектории
                    return self.result
                
                # Сохраняем результат
                traj_data["fields"].append(field)
                traj_data["x0"].append(fit_result["x0"])
                traj_data["gamma"].append(fit_result["gamma"])
                traj_data["A"].append(fit_result["A"])
                traj_data["y0"].append(fit_result["y0"])
                traj_data["q"].append(fit_result["q"])
                
                # Обновляем данные для визуализации среза
                self._current_cut_freqs = fit_freqs
                self._current_cut_values = fit_values
                
                # Генерируем кривую фита для отображения
                fit_freqs_dense = np.linspace(fit_freqs.min(), fit_freqs.max(), 200)
                if self.model_name == "Fano":
                    self._current_fit_values = self.model_func(
                        fit_freqs_dense, 
                        fit_result["x0"], fit_result["gamma"], 
                        fit_result["q"], fit_result["A"], fit_result["y0"]
                    )
                else:
                    self._current_fit_values = self.model_func(
                        fit_freqs_dense,
                        fit_result["x0"], fit_result["gamma"],
                        fit_result["A"], fit_result["y0"]
                    )
                self._current_fit_freqs = fit_freqs_dense
                
                # Обновляем визуализацию
                self._update_line(self.result)
                
                plt.pause(0.001)
                
                # Проверяем прерывание
                if self.interrupted:
                    print(f"[IEPeakApproximator] Траектория {traj_idx+1}: прервано пользователем")
                    self._continue_from_idx = i + 1
                    self._start_traj_idx = traj_idx  # Сохраняем индекс траектории для возобновления
                    return self.result
            
            # Конец траектории - валидация
            if not self._validate_trajectory_end():
                print(f"[IEPeakApproximator] Траектория {traj_idx+1}: отклонена пользователем")
                self.interrupted = True
                self._start_traj_idx = traj_idx  # Сохраняем для повторной обработки этой траектории
                return self.result
            print(f"[IEPeakApproximator] Траектория {traj_idx+1}: подтверждена")
            
            # Сбрасываем индексы после успешного завершения траектории
            self._continue_from_idx = None
        
        return self.result

    def _update_line(self, points):
        """
        Обновляет визуализацию: маркеры на контурном графике, графики параметров и срез.
        """
        result = points if isinstance(points, dict) else self.result
        
        # Собираем все точки для маркеров на контуре
        all_points = []
        for traj_data in result.get("trajectories", []):
            fields = traj_data.get("fields", [])
            x0s = traj_data.get("x0", [])
            for f, x0 in zip(fields, x0s):
                all_points.append((f, x0))
        
        self.marker.update_ticks(all_points)
        
        # Обновляем графики параметров
        self._update_param_plots(result)
        
        # Обновляем график среза
        self._update_cut_plot()
        
        # Перерисовываем
        if self.figure is not None:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()

    def _update_param_plots(self, result):
        """Обновляет графики параметров (x0, gamma)."""
        for i, traj_data in enumerate(result.get("trajectories", [])):
            if i >= len(self._x0_lines):
                break
            
            fields = traj_data.get("fields", [])
            x0s = traj_data.get("x0", [])
            gammas = traj_data.get("gamma", [])
            
            self._x0_lines[i].set_data(fields, x0s)
            self._gamma_lines[i].set_data(fields, gammas)
        
        # Автомасштабирование осей
        for ax in [self.ax_x0, self.ax_gamma]:
            ax.relim()
            ax.autoscale_view()

    def _update_cut_plot(self):
        """Обновляет график текущего среза с фитом."""
        if self._current_cut_freqs is not None and self._current_cut_values is not None:
            self._cut_data_line.set_data(self._current_cut_freqs, self._current_cut_values)
        
        if self._current_fit_freqs is not None and self._current_fit_values is not None:
            self._cut_fit_line.set_data(self._current_fit_freqs, self._current_fit_values)
        
        self.ax_cut.relim()
        self.ax_cut.autoscale_view()

    def delete_wrong_results(self):
        """
        Позволяет пользователю выбрать рамочкой неправильные маркеры 
        и удаляет соответствующие результаты.
        """
        print("[IEPeakApproximator] Выделите рамкой точки для удаления")
        self._selected_rect = None
        self._deletion_confirmed = False
        
        def on_select(eclick, erelease):
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])
            self._selected_rect = (x1, x2, y1, y2)
            print(f"[IEPeakApproximator] Область: x=[{x1:.2f}, {x2:.2f}], y=[{y1:.2f}, {y2:.2f}]")
        
        rect_selector = RectangleSelector(
            self.ax_contour, on_select,
            useblit=True,
            button=[1],
            interactive=True
        )
        
        ax_btn = self.figure.add_axes([0.35, 0.01, 0.1, 0.05])
        btn_confirm = Button(ax_btn, 'Delete Selected')
        
        def on_confirm(event):
            self._deletion_confirmed = True
        
        btn_confirm.on_clicked(on_confirm)
        
        while not self._deletion_confirmed:
            plt.pause(0.1)
        
        if self._selected_rect is not None:
            self._remove_points_in_rect(*self._selected_rect)
        
        ax_btn.remove()
        rect_selector.set_active(False)
        
        self.figure.canvas.draw_idle()

    def _remove_points_in_rect(self, x1, x2, y1, y2):
        """
        Удаляет из результатов точки, попавшие в прямоугольник [x1,x2] x [y1,y2].
        """
        print("[IEPeakApproximator] Удаляю точки в выделенной области")
        traj_data = self.result["trajectories"][self._current_traj_idx]
        
        fields = traj_data["fields"]
        x0s = traj_data["x0"]
        
        indices_to_remove = []
        for i, (f, x0) in enumerate(zip(fields, x0s)):
            if x1 <= f <= x2 and y1 <= x0 <= y2:
                indices_to_remove.append(i)
        
        for i in reversed(indices_to_remove):
            for key in ["fields", "x0", "gamma", "A", "y0", "q"]:
                if key in traj_data and len(traj_data[key]) > i:
                    traj_data[key].pop(i)
        
        if indices_to_remove:
            self._continue_from_idx = min(indices_to_remove)
            print(f"[IEPeakApproximator] Продолжу с индекса {self._continue_from_idx}")
        
        self._update_line(self.result)

    def select_correcting_params(self):
        """
        Позволяет пользователю выбрать новые параметры пика 
        для продолжения аппроксимации с места ошибки.
        """
        print("[IEPeakApproximator] Выбор корректирующих параметров")
        
        # Определяем поле, с которого продолжить
        if self._continue_from_idx is not None:
            input_traj = self.input_trajectories[self._current_traj_idx]
            continue_field = input_traj["fields"][self._continue_from_idx]
        else:
            traj_data = self.result["trajectories"][self._current_traj_idx]
            if len(traj_data["fields"]) > 0:
                # Берём следующее поле
                last_idx = len(traj_data["fields"])
                input_traj = self.input_trajectories[self._current_traj_idx]
                if last_idx < len(input_traj["fields"]):
                    continue_field = input_traj["fields"][last_idx]
                else:
                    continue_field = traj_data["fields"][-1]
            else:
                input_traj = self.input_trajectories[self._current_traj_idx]
                continue_field = input_traj["fields"][0]
        
        # Делаем срез в этом поле
        cut_executor = ECut(
            self.data,
            "Correcting params cut",
            axis="x",
            initial_params={"cut_value": continue_field}
        )
        print(f"[IEPeakApproximator] Срез для корректировки, поле={continue_field}")
        cut_executor.execute()
        cut_data = cut_executor.get_result()
        
        # Добавляем highlight для ожидаемой частоты из входных данных
        input_traj = self.input_trajectories[self._current_traj_idx]
        if self._continue_from_idx is not None and self._continue_from_idx < len(input_traj["freq"]):
            expected_freq = input_traj["freq"][self._continue_from_idx]
            freq_idx = np.abs(self.data["y"] - expected_freq).argmin()
            field_idx = np.abs(self.data["x"] - continue_field).argmin()
            mag = self.data["z"][field_idx, freq_idx]
            cut_data["highlight_points"] = [(expected_freq, mag)]
        
        # Даём выбрать новые параметры
        peak_executor = EPeakParams(cut_data, "Select correcting peak params")
        print("[IEPeakApproximator] Выберите новые параметры пика")
        peak_executor.execute_with_validation()
        
        if self.figure is not None:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()
        
        self.correcting_params = peak_executor.get_result()
        print(f"[IEPeakApproximator] Корректирующие параметры: {self.correcting_params}")

    def _validate_trajectory_end(self):
        """
        Показывает кнопки подтверждения в конце траектории.
        """
        print("[IEPeakApproximator] Подтвердите или отклоните траекторию")
        self._trajectory_confirmed = None
        
        ax_confirm = self.figure.add_axes([0.35, 0.01, 0.1, 0.05])
        btn_confirm = Button(ax_confirm, 'Confirm')
        
        ax_reject = self.figure.add_axes([0.46, 0.01, 0.1, 0.05])
        btn_reject = Button(ax_reject, 'Reject')
        
        def on_confirm(event):
            self._trajectory_confirmed = True
        
        def on_reject(event):
            self._trajectory_confirmed = False
        
        btn_confirm.on_clicked(on_confirm)
        btn_reject.on_clicked(on_reject)
        
        while self._trajectory_confirmed is None:
            plt.pause(0.1)
        
        ax_confirm.remove()
        ax_reject.remove()
        
        self.figure.canvas.draw_idle()
        
        return self._trajectory_confirmed

    def _close_execution_plot(self):
        """Закрывает окно с графиками."""
        print("[IEPeakApproximator] Закрытие окна визуализации")
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
