"""Loader for EMagnonFreqCalibrator results."""

import numpy as np

from .Loader import Loader


class LoadEMagnonFreqCalibrator(Loader):
    """
    Loader / saver for :class:`EMagnonFreqCalibrator` results.

    Result dict format::

        {
            "slope":     float,   # GHz/Oe
            "intercept": float,   # GHz
            "point1":    (field1, freq1),
            "point2":    (field2, freq2),
        }

    Text file format::

        # EMagnonFreqCalibrator results
        slope: 0.001234
        intercept: 0.567
        point1: 2800.0 3.65
        point2: 3000.0 3.85
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.data_path   = self.params.get("data_path",   None)
        self.result_path = self.params.get("result_path", None)

    # ------------------------------------------------------------------ #
    # Load                                                                #
    # ------------------------------------------------------------------ #

    def load_data(self) -> dict:
        if self.data_path is None:
            raise ValueError("data_path is not set.")

        data = {}

        with open(self.data_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue

                key, raw_values = line.split(":", 1)
                key        = key.strip()
                raw_values = raw_values.strip()

                if key == "slope":
                    data["slope"] = float(raw_values)
                elif key == "intercept":
                    data["intercept"] = float(raw_values)
                elif key in ("point1", "point2"):
                    parts = raw_values.split()
                    if len(parts) == 2:
                        data[key] = (float(parts[0]), float(parts[1]))

        return data

    # ------------------------------------------------------------------ #
    # Save                                                                #
    # ------------------------------------------------------------------ #

    def save_data(self, data: dict):
        if self.result_path is None:
            raise ValueError("result_path is not set.")

        slope     = data.get("slope", 0.0)
        intercept = data.get("intercept", 0.0)
        point1    = data.get("point1", (0.0, 0.0))
        point2    = data.get("point2", (0.0, 0.0))

        with open(self.result_path, "w", encoding="utf-8") as f:
            f.write("# EMagnonFreqCalibrator results\n\n")
            f.write(f"slope: {slope}\n")
            f.write(f"intercept: {intercept}\n")
            f.write(f"point1: {point1[0]} {point1[1]}\n")
            f.write(f"point2: {point2[0]} {point2[1]}\n")
