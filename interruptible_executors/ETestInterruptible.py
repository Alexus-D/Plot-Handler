import time

import matplotlib.pyplot as plt

from .InterruptibleExecutor import InterruptibleExecutor
from markers import MPoints
from plotters import PContour
from ui_selectors import SFreq
from validators import VContourf


TOTAL_PROCESSING_TIME = 2.0


class ETestInterruptible(InterruptibleExecutor):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
        self.plotter = None
        self.marker = None

    def execute(self):
        if self.plotter is None:
            self.plotter = PContour(self.data)
            self.marker = MPoints(self.plotter.get_axis_for_marker())
            fig = self.plotter.get_figure()
            fig.canvas.draw_idle()
            plt.show(block=False)
            plt.pause(0.001)
        return super().execute()

    def validate(self, data_for_visualization) -> bool:
        validator = VContourf(self.stage_name, data_for_visualization)
        points = data_for_visualization.get("points", [])
        if points:
            axis = validator.plotter.get_axis_for_marker()
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            axis.plot(xs, ys, color="red", linewidth=1, marker="o", markersize=2)
        plt.show()
        return bool(validator.get_params().get("Validation", False))

    def select_initial_params(self):
        print("Select initial height: click on graph after pressing 'Select Frequency', then close window")
        selector = SFreq(self.stage_name, self.data)
        plt.show()
        params = selector.get_params()
        height = params.get('select_freq', 0)
        print(f"Selected height: {height}")
        return {'height': height}

    def select_correcting_params(self) -> None:
        print("Select new height for remaining points")
        selector = SFreq(f"{self.stage_name} - Correction", self.data)
        plt.show()
        params = selector.get_params()
        height = params.get('select_freq', 0)
        print(f"Selected new height: {height}")
        self.correcting_params = {'height': height}
        return None

    def prepare_for_visualization(self, result) -> dict:
        plot_data = dict(self.data)
        plot_data["points"] = self._result_to_points(result)
        return plot_data

    def delete_wrong_results(self) -> None:
        return

    def _update_line(self, points) -> None:
        if self.marker is None:
            raise ValueError("Marker is not initialized.")
        self.marker.delete_ticks()
        if points:
            if isinstance(points, dict):
                points = self._result_to_points(points)
            self.marker.set_ticks(points)
        self.marker.redraw()

    def _result_to_points(self, result_dict: dict) -> list:
        if not result_dict:
            return []
        return sorted(result_dict.items(), key=lambda item: item[0])

    def interruptible_function(self) -> dict:
        params = self.correcting_params or self.initial_params
        height = params.get('height', 0)

        x_values = list(self.remain_data.get('x', []))
        x_values = sorted(x_values)
        result = {}
        processed_x = []

        fig = self.plotter.get_figure()
        total = len(x_values)
        total_source = len(self.data.get('x', [1])) or 1
        target_time = TOTAL_PROCESSING_TIME * (len(x_values) / total_source)
        print(f"Processing {total} points at height={height:.2f} (target: {target_time:.1f}s)")

        start_time = time.time()
        base_points = self._result_to_points(self.result)
        for i, x in enumerate(x_values):
            if self.interrupted:
                break

            result[x] = height
            processed_x.append(x)

            new_points = base_points + list(result.items())
            self._update_line(new_points)
            fig.canvas.flush_events()

            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{total}")

            elapsed = time.time() - start_time
            expected = target_time * (i + 1) / total if total else 0
            if elapsed < expected:
                time.sleep(expected - elapsed)

        if "x" in self.remain_data:
            remaining = [x for x in list(self.remain_data["x"]) if x not in set(processed_x)]
            self.remain_data["x"] = remaining

        actual_time = time.time() - start_time
        print(
            f"Done: {len(processed_x)} points in {actual_time:.1f}s" +
            (" (interrupted)" if self.interrupted else "")
        )
        return result
