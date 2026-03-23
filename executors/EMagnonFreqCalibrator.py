"""Executor that calibrates magnon frequency as a linear function of field.

Given two user-selected points (field1, freq1) and (field2, freq2), computes:
    slope     = (freq2 - freq1) / (field2 - field1)
    intercept = freq1 - slope * field1

Then magnon_freq(field) = slope * field + intercept.
"""

import matplotlib.pyplot as plt

from ui_selectors import STwoPointsCalibration

from .Executor import Executor


class EMagnonFreqCalibrator(Executor):
    """
    Calibrates magnon frequency vs. field using a two-point linear fit.

    Input ``data`` dict:
        Any contour-plot dict (must have 'x', 'y', 'z' keys for visualization).

    ``initial_params`` (optional):
        - calibration_points: [(field1, freq1), (field2, freq2)]
          If not provided, the user will be prompted to click two points.

    Result dict (``get_result()``):
        - slope:     (freq2 - freq1) / (field2 - field1)   [GHz / Oe]
        - intercept: freq1 - slope * field1                 [GHz]
        - point1:    (field1, freq1)
        - point2:    (field2, freq2)
    """

    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)

    # ------------------------------------------------------------------ #
    # Executor interface                                                   #
    # ------------------------------------------------------------------ #

    def validate(self):
        """Always accepts the calibration (user can re-run if needed)."""
        return True

    def execute(self):
        points = self.initial_params.get("calibration_points")
        if points is None or len(points) != 2:
            raise ValueError(
                "calibration_points must be a list of exactly 2 tuples: "
                "[(field1, freq1), (field2, freq2)]"
            )

        (field1, freq1), (field2, freq2) = points

        # Compute linear fit coefficients
        if abs(field2 - field1) < 1e-9:
            raise ValueError(
                f"Field values are too close ({field1:.4f} ≈ {field2:.4f}); "
                "cannot compute slope."
            )

        slope     = (freq2 - freq1) / (field2 - field1)
        intercept = freq1 - slope * field1

        self.result = {
            "slope":     slope,
            "intercept": intercept,
            "point1":    (field1, freq1),
            "point2":    (field2, freq2),
        }

        print(
            f"[EMagnonFreqCalibrator] Calibration complete:\n"
            f"  slope     = {slope:.6f} GHz/Oe\n"
            f"  intercept = {intercept:.6f} GHz\n"
            f"  => magnon_freq(H) = {slope:.6f} * H + {intercept:.6f}"
        )

    def select_initial_params(self):
        if "calibration_points" in self.initial_params:
            points = self.initial_params["calibration_points"]
            if len(points) == 2:
                return

        print("[EMagnonFreqCalibrator] Waiting for user to select 2 calibration points...")
        selector = STwoPointsCalibration(self.stage_name, self.data)
        plt.show()
        params = selector.get_params()
        self.initial_params["calibration_points"] = params.get("calibration_points", [])

        if len(self.initial_params["calibration_points"]) != 2:
            raise ValueError("User must select exactly 2 points for calibration.")

    def prepare_for_visualization(self):
        """Stores the calibration result for optional plotting."""
        self.data_for_visualization = self.result
