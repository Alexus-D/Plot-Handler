import numpy as np
import matplotlib.pyplot as plt

import config_physics
import utils.algorithms as alg

from .IEPeakWatcher import IEPeakWatcher


class IEPeakWatcherDraw(IEPeakWatcher):
    """
    IEPeakWatcher with two trajectories and user-drawn boundary (polyline)
    that separates their search regions.
    """

    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
        self._boundary_points = None
        self._boundary_x = None
        self._boundary_y = None
        self._traj_region = []  # "above" or "below" for each trajectory

    def select_initial_params(self):
        super().select_initial_params()

        if len(self.trajectories) != 2:
            raise ValueError("IEPeakWatcherDraw requires exactly 2 trajectories")

        self._collect_boundary()
        self._assign_trajectory_regions()

    def _collect_boundary(self):
        """
        Opens contour plot and lets user draw a polyline by clicks.
        Finish with right-click or Enter.
        """
        print("[IEPeakWatcherDraw] Нарисуйте границу ломаной линией")
        print("[IEPeakWatcherDraw] ЛКМ — точки, Enter/ПКМ — завершить")

        fig, ax = plt.subplots(figsize=(8, 6))
        self.figure = fig
        self._draw_contour(ax)

        points = []
        line, = ax.plot([], [], "r-", linewidth=2)
        done = {"value": False}

        def redraw_line():
            if points:
                xs, ys = zip(*points)
                line.set_data(xs, ys)
                fig.canvas.draw_idle()

        def on_click(event):
            if event.inaxes != ax:
                return
            if event.button == 1:
                points.append((event.xdata, event.ydata))
                redraw_line()
            elif event.button == 3:
                done["value"] = True

        def on_key(event):
            if event.key in ("enter", "escape"):
                done["value"] = True

        cid_click = fig.canvas.mpl_connect("button_press_event", on_click)
        cid_key = fig.canvas.mpl_connect("key_press_event", on_key)

        plt.show(block=False)
        while not done["value"]:
            plt.pause(0.05)

        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)

        if len(points) < 2:
            plt.close(fig)
            raise ValueError("Boundary must contain at least 2 points")

        self._boundary_points = np.array(points, dtype=float)
        self._prepare_boundary_interpolator()

        plt.close(fig)
        self.figure = None

    def _prepare_boundary_interpolator(self):
        xs = self._boundary_points[:, 0]
        ys = self._boundary_points[:, 1]

        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]

        # Remove duplicate x values by keeping the last occurrence
        unique_xs, unique_indices = np.unique(xs, return_index=True)
        xs = unique_xs
        ys = ys[unique_indices]

        self._boundary_x = xs
        self._boundary_y = ys

    def _boundary_at(self, field_value: float) -> float:
        """Interpolated boundary frequency at given field value."""
        return float(np.interp(field_value, self._boundary_x, self._boundary_y))

    def _assign_trajectory_regions(self):
        """Assign each trajectory to 'above' or 'below' region."""
        self._traj_region = []
        for start, _ in self.trajectories:
            start_field = start[0]
            start_freq = start[1]
            boundary_freq = self._boundary_at(start_field)
            region = "above" if start_freq >= boundary_freq else "below"
            self._traj_region.append(region)
        print(f"[IEPeakWatcherDraw] Trajectory regions: {self._traj_region}")

    def _constrain_by_boundary(self, freqs, s_values, field_value, region):
        boundary_freq = self._boundary_at(field_value)
        if region == "above":
            mask = freqs > boundary_freq
        else:
            mask = freqs < boundary_freq
        return freqs[mask], s_values[mask], boundary_freq

    def interruptible_function(self):
        """
        Основной цикл отслеживания пиков по двум траекториям
        с ограничением поиска областью по пользовательской границе.
        """
        print("[IEPeakWatcherDraw] Старт: отслеживание пиков")
        print(f"[DEBUG] interrupted={self.interrupted}, correcting_params={bool(self.correcting_params)}")

        start_idx = self._start_traj_idx
        self._start_traj_idx = 0

        for traj_idx in range(start_idx, len(self.trajectories)):
            start, end = self.trajectories[traj_idx]
            self._current_traj_idx = traj_idx
            print(f"[IEPeakWatcherDraw] Траектория {traj_idx+1}: старт")

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
            print(f"[IEPeakWatcherDraw] Траектория {traj_idx+1}: шагов {len(self._field_indices)}")

            fixed_peak_type = self._get_fixed_peak_type(
                traj_idx,
                traj_data,
                fallback_type=self.initial_peak_params[traj_idx].get("peak_type", config_physics.PEAK_TYPE)
            )

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
                print(f"[IEPeakWatcherDraw] Траектория {traj_idx+1}: использую корректирующие параметры")
            elif start_idx > 0 and len(traj_data["freq"]) > 0:
                current_params = {
                    "peak_freq": traj_data["freq"][-1],
                    "peak_width": traj_data["width"][-1],
                    "prominence": traj_data["prominence"][-1],
                    "peak_type": self.initial_peak_params[traj_idx].get("peak_type", config_physics.PEAK_TYPE)
                }
                print(f"[IEPeakWatcherDraw] Траектория {traj_idx+1}: использую последние параметры")
            else:
                current_params = self.initial_peak_params[traj_idx].copy()
                print(f"[IEPeakWatcherDraw] Траектория {traj_idx+1}: использую начальные параметры")

            if fixed_peak_type:
                current_params["peak_type"] = fixed_peak_type

            print(f"[DEBUG] Цикл по полям: start_idx={start_idx}, total_indices={len(self._field_indices)}")
            for i in range(start_idx, len(self._field_indices)):
                field_idx = self._field_indices[i]
                field = self.data["x"][field_idx]
                freqs = self.data["y"]
                s_values = self.data["z"][field_idx, :]

                region = self._traj_region[traj_idx]
                freqs_for_search, s_values_for_search, boundary_freq = self._constrain_by_boundary(
                    freqs, s_values, field, region
                )

                print(
                    f"[IEPeakWatcherDraw] Траектория {traj_idx+1}: поиск пика, i={i}, поле={field}, "
                    f"region={region}, boundary={boundary_freq}"
                )

                try:
                    if len(freqs_for_search) == 0:
                        raise ValueError("No data points in boundary constrained window")

                    expected_freq = current_params["peak_freq"]
                    freq_step = abs(freqs_for_search[1] - freqs_for_search[0]) if len(freqs_for_search) > 1 else 0.001
                    if region == "above" and expected_freq <= boundary_freq:
                        expected_freq = boundary_freq + freq_step
                    if region == "below" and expected_freq >= boundary_freq:
                        expected_freq = boundary_freq - freq_step

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

                if fixed_peak_type is None:
                    inferred_type = self._infer_peak_type(
                        freqs,
                        s_values,
                        peak["freq"],
                        current_params["peak_freq"],
                        current_params.get("peak_width", current_params.get("width", 0.1))
                    )
                    self._set_fixed_peak_type(traj_idx, traj_data, inferred_type)
                    fixed_peak_type = inferred_type
                    current_params["peak_type"] = inferred_type

                current_params["peak_freq"] = peak["freq"]
                current_params["peak_width"] = peak["width"]
                current_params["prominence"] = peak["prominence"]

                print(f"[DEBUG] Обновление визуализации, текущих точек: {len(traj_data['fields'])}")
                self._update_line(self.result)

                plt.pause(0.001)

                if self.interrupted:
                    print(f"[IEPeakWatcherDraw] Траектория {traj_idx+1}: прервано пользователем")
                    print(f"[DEBUG] Прерывание: устанавливаем _continue_from_idx={i + 1}")
                    self._continue_from_idx = i + 1
                    self._start_traj_idx = traj_idx
                    return self.result

            if not self._validate_trajectory_end():
                print(f"[IEPeakWatcherDraw] Траектория {traj_idx+1}: отклонена пользователем")
                self.interrupted = True
                self._start_traj_idx = traj_idx
                return self.result
            print(f"[IEPeakWatcherDraw] Траектория {traj_idx+1}: подтверждена")

            self._continue_from_idx = None

        return self.result