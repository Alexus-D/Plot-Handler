import numpy as np
import matplotlib.pyplot as plt

from executors import ECut, EPeakParams
from executors.EPolylineTrajectories import EPolylineTrajectories

from .IEPeakWatcher import IEPeakWatcher


class IEPeakWatcherPolyline(IEPeakWatcher):
    """
    Peak tracker where the user draws an approximate polyline trajectory,
    and peaks are searched near that polyline at every field step.

    Differences from :class:`IEPeakWatcher`:
    - Trajectories are drawn as multi-point polylines (not just start/end pairs).
    - At each field value the expected frequency is interpolated from the
      user-drawn polyline rather than taken from the previously found peak.
      This makes the search "snap to the drawn path" instead of tracking.
    - Width and prominence still adapt from the found peaks.
    - All interruption / correction / deletion logic is inherited unchanged.
    """

    def __init__(
        self,
        data: dict,
        stage_name: str,
        initial_params: dict = None,
        save_figure_path: str = None,
    ):
        super().__init__(data, stage_name, initial_params, save_figure_path)
        # List[List[Tuple[float, float]]] – polylines in click order
        self._polylines = []

    # ------------------------------------------------------------------
    # Overridden: initial parameter selection
    # ------------------------------------------------------------------

    def select_initial_params(self):
        """
        1) Collect polyline trajectories via EPolylineTrajectories.
        2) For each trajectory, make a cut at the first point and let the user
           select initial peak parameters.
        """
        print("[IEPeakWatcherPolyline] Старт: рисование ломаных траекторий")

        traj_executor = EPolylineTrajectories(self.data, "Draw Polyline Trajectories")
        traj_executor.execute_with_validation()
        self._polylines = traj_executor.get_result()["trajectories"]

        if not self._polylines:
            raise ValueError("No trajectories drawn")

        # Expose as (start, end) pairs so the base-class machinery works
        self.trajectories = [(p[0], p[-1]) for p in self._polylines]

        # Get initial peak params at the start of each trajectory
        self.initial_peak_params = []
        for i, polyline in enumerate(self._polylines):
            start_field, start_freq = polyline[0]
            print(f"[IEPeakWatcherPolyline] Траектория {i + 1}: срез в поле {start_field}")

            cut_executor = ECut(
                self.data,
                f"Cut for polyline trajectory {i + 1}",
                axis="x",
                initial_params={"cut_value": start_field},
            )
            cut_executor.execute()
            cut_data = cut_executor.get_result()
            cut_data["highlight_points"] = [
                (start_freq, self._get_z_at_point(start_field, start_freq))
            ]

            peak_executor = EPeakParams(
                cut_data,
                f"Peak params for polyline trajectory {i + 1}",
            )
            print(f"[IEPeakWatcherPolyline] Траектория {i + 1}: выбор параметров пика")
            peak_executor.execute_with_validation()
            self.initial_peak_params.append(peak_executor.get_result())

        self.result = {"trajectories": []}
        print("[IEPeakWatcherPolyline] Готово: начальные параметры выбраны")

    # ------------------------------------------------------------------
    # Overridden: expected frequency comes from the polyline
    # ------------------------------------------------------------------

    def _get_expected_freq(self, traj_idx: int, field: float, current_params: dict) -> float:
        """Interpolates the expected frequency from the drawn polyline."""
        return self._interpolate_polyline(traj_idx, field)

    def _interpolate_polyline(self, traj_idx: int, field: float) -> float:
        """
        Linearly interpolates the frequency value from polyline ``traj_idx``
        at the given ``field`` value.

        Handles polylines going in either direction (left-to-right or
        right-to-left) by sorting before interpolation.
        Extrapolation at edges returns the nearest endpoint value.
        """
        pts = self._polylines[traj_idx]
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)

        order = np.argsort(xs)
        xs_sorted = xs[order]
        ys_sorted = ys[order]

        # np.interp clamps at the edges — no extrapolation surprises
        return float(np.interp(field, xs_sorted, ys_sorted))

    # ------------------------------------------------------------------
    # Overridden: show polylines on the execution plot
    # ------------------------------------------------------------------

    def _setup_execution_plot(self, result):
        """Draw the execution window and overlay the drawn polylines."""
        super()._setup_execution_plot(result)
        self._draw_polylines_on_contour()

    def _draw_polylines_on_contour(self):
        """Overlays the user-drawn polylines on the contour axes."""
        if self.ax_contour is None:
            return
        colors = plt.cm.tab10.colors
        for i, polyline in enumerate(self._polylines):
            color = colors[i % len(colors)]
            xs = [p[0] for p in polyline]
            ys = [p[1] for p in polyline]
            self.ax_contour.plot(
                xs, ys,
                "--",
                color=color,
                linewidth=1.5,
                alpha=0.7,
                label=f"Polyline {i + 1}",
            )
        self.ax_contour.legend(loc="upper right", fontsize=7)
        if self.figure is not None:
            self.figure.canvas.draw_idle()
