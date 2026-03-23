"""Executor that fits a Lorentzian profile to every field step of one peak
trajectory produced by IEPeakWatcher / IEPeakWatcherPolyline.
"""

import numpy as np
import scipy.optimize

from utils.phys_models import Lorentzian
from utils.unit_transformations import convert_dB_to_linear, convert_linear_to_dB

from .Executor import Executor


class ELorentzianApproximator(Executor):
    """
    Fits a Lorentzian to each field point of a single peak trajectory.

    Input ``data`` dict:
        - x          : 1-D array of field values
        - y          : 1-D array of frequency values
        - z          : 2-D array, shape (n_fields, n_freqs)
        - trajectories : list of trajectory dicts from IEPeakWatcher /
          IEPeakWatcherPolyline.  Each element must contain the keys
          ``fields``, ``freq``, ``width``, ``magnitude``, ``prominence``.

    ``initial_params`` (all optional):
        - traj_idx          : int   – which trajectory to fit (default 0).
        - fit_window_factor : float – fit window = width * factor (default 1.0).
          factor=1.0 means the window spans exactly the tracked peak width
          (±width/2 around the expected frequency).

    Result dict (``get_result()``):
        - traj_idx : int
        - fields   : list[float]
        - x0       : list[float]  – fitted Lorentzian centres (GHz)
        - gamma    : list[float]  – half-width at half-maximum in linear scale (GHz)
        - A        : list[float]  – amplitude in linear scale (negative for a dip)
        - y0       : list[float]  – baseline in linear scale
        - model    : "Lorentzian"
    """

    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
        self._traj_idx: int = 0
        self._fit_window_factor: float = 1.0

    # ------------------------------------------------------------------ #
    # Executor interface                                                   #
    # ------------------------------------------------------------------ #

    def select_initial_params(self):
        self._traj_idx = int(self.initial_params.get("traj_idx", 0))
        self._fit_window_factor = float(
            self.initial_params.get("fit_window_factor", 1.0)
        )

        trajectories = self.data.get("trajectories", [])
        if not trajectories:
            raise ValueError(
                "No trajectories found in data. Run IEPeakWatcher first."
            )
        if self._traj_idx >= len(trajectories):
            raise ValueError(
                f"traj_idx={self._traj_idx} is out of range "
                f"(only {len(trajectories)} trajectories in data)"
            )

    def execute(self):
        traj = self.data["trajectories"][self._traj_idx]

        fields       = traj["fields"]
        freqs_track  = traj["freq"]
        widths       = traj["width"]
        magnitudes   = traj["magnitude"]
        prominences  = traj["prominence"]

        # Pre-sliced data may be supplied by IELorentzianApproximator for speed.
        # Each entry is (fit_freqs, fit_vals_lin, p0) for the i-th field step.
        # A None sentinel means the step was skipped (window too narrow).
        precomputed = self.initial_params.get("precomputed_slices")

        all_freqs = self.data["y"]
        z         = self.data["z"]
        x_arr     = self.data["x"]

        result_fields, x0s, gammas, As, y0s = [], [], [], [], []

        for i, field in enumerate(fields):
            if precomputed is not None:
                entry = precomputed[i]
                if entry is None:
                    print(
                        f"[ELorentzianApproximator] Skipping field={field:.3f}: "
                        "too few points in fit window"
                    )
                    continue
                fit_freqs, fit_vals_lin, p0 = entry
            else:
                field_idx = int(np.abs(x_arr - field).argmin())
                s_values  = z[field_idx, :]

                expected_freq = freqs_track[i]
                width         = widths[i]
                half_win      = width * self._fit_window_factor / 2.0

                mask = (
                    (all_freqs >= expected_freq - half_win)
                    & (all_freqs <= expected_freq + half_win)
                )
                fit_freqs = all_freqs[mask]
                fit_vals  = s_values[mask]

                if len(fit_freqs) < 5:
                    print(
                        f"[ELorentzianApproximator] Skipping field={field:.3f}: "
                        "too few points in fit window"
                    )
                    continue

                # Convert dB data to linear scale before fitting.
                fit_vals_lin = convert_dB_to_linear(fit_vals)

                magnitude_lin = convert_dB_to_linear(magnitudes[i])
                baseline_lin  = convert_dB_to_linear(magnitudes[i] + prominences[i])
                p0 = [
                    expected_freq,
                    width / 2,
                    magnitude_lin - baseline_lin,
                    baseline_lin,
                ]

            try:
                popt, _ = scipy.optimize.curve_fit(
                    Lorentzian,
                    fit_freqs,
                    fit_vals_lin,
                    p0=p0,
                    method="lm",
                    maxfev=2000,
                )
            except Exception as exc:
                print(
                    f"[ELorentzianApproximator] Fit failed at "
                    f"field={field:.3f}: {exc}"
                )
                continue

            result_fields.append(field)
            x0s.append(popt[0])
            gammas.append(abs(popt[1]))
            As.append(popt[2])
            y0s.append(abs(popt[3]))

        self.result = {
            "traj_idx": self._traj_idx,
            "fields":   result_fields,
            "x0":       x0s,
            "gamma":    gammas,
            "A":        As,
            "y0":       y0s,
            "model":    "Lorentzian",
        }

    def prepare_for_visualization(self):
        """Stores the last successfully fitted cut + curve for inspection."""
        if not self.result.get("fields"):
            self.data_for_visualization = {}
            return

        all_freqs = self.data["y"]
        z         = self.data["z"]
        x_arr     = self.data["x"]

        field     = self.result["fields"][-1]
        field_idx = int(np.abs(x_arr - field).argmin())
        s_values  = z[field_idx, :]

        x0    = self.result["x0"][-1]
        gamma = self.result["gamma"][-1]
        A     = self.result["A"][-1]
        y0    = self.result["y0"][-1]

        fit_freqs_dense = np.linspace(all_freqs.min(), all_freqs.max(), 500)
        fit_curve_lin   = Lorentzian(fit_freqs_dense, x0, gamma, A, y0)
        fit_curve_db    = convert_linear_to_dB(fit_curve_lin)

        self.data_for_visualization = {
            "x":         all_freqs,
            "y":         s_values,
            "fit_x":     fit_freqs_dense,
            "fit_curve": fit_curve_db,
            "x0":        x0,
            "gamma":     gamma,
            "A":         A,
            "y0":        y0,
            "field":     field,
        }

    def validate(self):
        """Always accepts; inspect ``data_for_visualization`` if needed."""
        return True
