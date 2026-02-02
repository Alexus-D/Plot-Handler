import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from ui_selectors import SNPoints, SPeakParams
from utils.algorithms import make_cut
from utils.unit_transformations import convert_dB_to_linear, convert_linear_to_dB

from .Executor import Executor


def estimate_cavity_params(res_magnitude, resonance_freq, cavity_width, plato):
    con = np.abs(convert_dB_to_linear(res_magnitude) - convert_dB_to_linear(plato))

    kappa = con * cavity_width / 2
    beta = cavity_width / 2 * (1 - con)

    return {
        'kappa': kappa,
        'beta': beta,
        'resonance_freq': resonance_freq,
        'plato': plato,
        'res_magnitude': res_magnitude
    }

def fit_cavity_response(freqs, s_average, initial_params):
    p0 = [
        initial_params['kappa'],
        initial_params['beta'],
        initial_params['resonance_freq'],
        initial_params['plato']
    ]

    popt, _ = sp.optimize.curve_fit(cavity_model, freqs, s_average, p0=p0)

    fitted_params = {
        'kappa': popt[0],
        'beta': popt[1],
        'resonance_freq': popt[2],
        'plato': popt[3]
    }
    fitted_params['res_magnitude'] = cavity_model(
        fitted_params['resonance_freq'],
        fitted_params['kappa'],
        fitted_params['beta'],
        fitted_params['resonance_freq'],
        fitted_params['plato']
    )

    return fitted_params

def cavity_model(f, kappa, beta, f0, plato):
    delta_f = f - f0
    response = convert_dB_to_linear(plato) + kappa / np.sqrt(delta_f**2 + (kappa + beta)**2)
    return convert_linear_to_dB(response)


class EResonatorExtractor(Executor):
    def __init__(self, data: dict, stage_name: str, axis: str = "x", initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.axis = axis
        self.fit_enabled = self.initial_params.get("fit", True)

    def validate(self):
        return True

    def select_initial_params(self):
        required = {"cut_value", "peak_freq", "peak_value", "plateau", "peak_width"}
        if required.issubset(self.initial_params.keys()):
            return

        cut_value = self.initial_params.get("cut_value")
        x_sel = None
        y_sel = None
        if cut_value is None:
            label = "Select cut point"
            selector = SNPoints(
                self.stage_name,
                self.data,
                num_points=1,
                buttons=[("select_points", label)],
            )
            plt.show()
            points = selector.get_params().get("select_points", [])
            if not points:
                raise ValueError("No point selected for cut")
            x_sel, y_sel = points[0]
            cut_value = x_sel if self.axis == "x" else y_sel

        cut = make_cut(self.data, cut_value=cut_value, axis=self.axis)
        if self.axis == "x":
            xlabel = self.data.get("ylabel", "Y-axis")
        else:
            xlabel = self.data.get("xlabel", "X-axis")
        ylabel = self.data.get("zlabel", "Z-axis")
        title = self.data.get("title", "Contour Cut")
        title = f"{title} (cut {self.axis}={cut_value:.4g})"

        cut_data = {
            **cut,
            "xlabel": xlabel,
            "ylabel": ylabel,
            "title": title,
        }

        if self.axis == "x" and y_sel is not None:
            highlight = (y_sel, self._get_z_at_point(cut_value, y_sel))
        elif self.axis == "y" and x_sel is not None:
            highlight = (x_sel, self._get_z_at_point(x_sel, cut_value))
        else:
            highlight = None
        if highlight is not None:
            cut_data["highlight_points"] = [highlight]

        peak_executor = SPeakParams(self.stage_name, cut_data)
        while not peak_executor.is_done():
            if not plt.fignum_exists(peak_executor.figure.number):
                break
            plt.pause(0.1)
        if plt.fignum_exists(peak_executor.figure.number):
            plt.close(peak_executor.figure)
        params = peak_executor.get_params()

        self.initial_params = {
            "cut_value": cut_value,
            "peak_freq": params.get("peak_freq"),
            "peak_value": params.get("peak_value"),
            "plateau": params.get("plateau"),
            "peak_width": params.get("peak_width"),
            "peak_type": params.get("peak_type"),
            "fit": self.fit_enabled,
        }

    def _get_z_at_point(self, field, freq):
        fields = self.data["x"]
        freqs = self.data["y"]
        z = self.data["z"]

        field_idx = np.abs(fields - field).argmin()
        freq_idx = np.abs(freqs - freq).argmin()
        return z[field_idx, freq_idx]

    def execute(self):
        cut_value = self.initial_params.get("cut_value")
        resonance_freq = self.initial_params.get("peak_freq")
        res_magnitude = self.initial_params.get("peak_value")
        cavity_width = self.initial_params.get("peak_width")
        plato = self.initial_params.get("plateau")

        missing = [
            name
            for name, value in (
                ("cut_value", cut_value),
                ("peak_freq", resonance_freq),
                ("peak_value", res_magnitude),
                ("peak_width", cavity_width),
                ("plateau", plato),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"Missing parameters: {', '.join(missing)}")

        cut = make_cut(self.data, cut_value=cut_value, axis=self.axis)
        freqs = cut["x"]
        s_values = cut["y"]

        initial_cavity = estimate_cavity_params(res_magnitude, resonance_freq, cavity_width, plato)
        fitted = None
        if self.fit_enabled:
            fitted = fit_cavity_response(freqs, s_values, initial_cavity)

        fit_params = fitted if fitted is not None else initial_cavity
        fit_curve = cavity_model(freqs, fit_params["kappa"], fit_params["beta"], fit_params["resonance_freq"], fit_params["plato"])

        self.result = {
            "axis": self.axis,
            "cut_value": cut_value,
            "cut": cut,
            "initial_params": initial_cavity,
            "fitted_params": fitted,
            "fit_enabled": self.fit_enabled,
            "fit_curve": fit_curve,
            "resonance_freq": resonance_freq,
            "res_magnitude": res_magnitude,
            "cavity_width": cavity_width,
            "plato": plato,
        }

    def prepare_for_visualization(self):
        self.data_for_visualization = self.result