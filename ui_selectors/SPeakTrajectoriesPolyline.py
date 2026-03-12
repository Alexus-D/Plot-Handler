from typing import Dict, List, Optional, Tuple

from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt

from .Selector import Selector
from plotters import Plotter, PContour
from markers import MPoints


class SPeakTrajectoriesPolyline(Selector):
    """
    Selector that lets the user draw multi-point polyline trajectories on a contour plot.

    Workflow:
        1. Press "Mark Trajectory" — enter drawing mode.
        2. Click points on the plot to build a polyline for the current trajectory.
        3. Press "End Trajectory" — finalize the current polyline (≥ 2 points required).
        4. Repeat steps 1–3 for additional trajectories.
        5. Press "Finish" — close the selector.

    Result: ``params["trajectories"]`` — list of polylines, each polyline is a list of
    ``(field, freq)`` tuples in click order.
    """

    def __init__(
        self,
        stage_name: str,
        plot_data: dict,
        buttons: Optional[List[Tuple[str, str]]] = None,
        clear_button: bool = False,
        save_button: bool = False,
    ):
        if buttons is None:
            buttons = [
                ("mark_trajectory", "Mark Trajectory"),
                ("end_trajectory", "End Trajectory"),
                ("finish", "Finish"),
            ]

        super().__init__(stage_name, plot_data, buttons, clear_button, save_button)

        self._current_polyline: List[Tuple[float, float]] = []
        self._line_artists: List = []          # Artists for finalized polylines
        self._current_line_artist = None       # Artist for the in-progress polyline
        self.params["trajectories"] = []

    # ------------------------------------------------------------------
    # Selector interface
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str):
        if mode == "finish":
            self.mode = None
            plt.close(self.figure)
            return

        if mode == "end_trajectory":
            if len(self._current_polyline) >= 2:
                self.params["trajectories"].append(list(self._current_polyline))
                self._current_polyline = []
                self._redraw_polylines()
            else:
                print("[SPeakTrajectoriesPolyline] Нужно минимум 2 точки для траектории")
            self.mode = None
            return

        super()._set_mode(mode)

    def mouse_on_click(self, event: MouseEvent) -> None:
        if self.mode != "mark_trajectory":
            return
        main_ax = self.markers["MPoints"].axes
        if event.inaxes is not main_ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._current_polyline.append((event.xdata, event.ydata))
        self._redraw_polylines()

    def _create_plotter(self, plot_data: dict) -> Plotter:
        return PContour(plot_data)

    def _create_markers(self, plotter: Plotter) -> Dict[str, MPoints]:
        # MPoints is kept for compatibility with base class, but drawing is
        # handled directly via _redraw_polylines().
        axis = plotter.get_axis_for_marker()
        marker = MPoints(axis)
        return {"MPoints": marker}

    def _redraw_markers(self):
        self._redraw_polylines()

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _redraw_polylines(self):
        """Remove stale artists and redraw all finalized + current polylines."""
        ax = self.markers["MPoints"].axes
        colors = plt.cm.tab10.colors

        # Preserve axis limits so auto-scaling doesn't zoom out
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Remove old line artists
        for artist in self._line_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._line_artists = []

        if self._current_line_artist is not None:
            try:
                self._current_line_artist.remove()
            except Exception:
                pass
            self._current_line_artist = None

        # Draw finalized polylines
        for i, polyline in enumerate(self.params["trajectories"]):
            color = colors[i % len(colors)]
            xs = [p[0] for p in polyline]
            ys = [p[1] for p in polyline]
            (line,) = ax.plot(
                xs, ys, "-o",
                color=color, linewidth=2, markersize=5,
                label=f"Traj {i + 1}",
            )
            self._line_artists.append(line)

        # Draw current in-progress polyline (dashed)
        if self._current_polyline:
            i = len(self.params["trajectories"])
            color = colors[i % len(colors)]
            xs = [p[0] for p in self._current_polyline]
            ys = [p[1] for p in self._current_polyline]
            (line,) = ax.plot(
                xs, ys, "--o",
                color=color, linewidth=1.5, markersize=5, alpha=0.7,
            )
            self._current_line_artist = line

        if ax.figure is not None:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.figure.canvas.draw_idle()
