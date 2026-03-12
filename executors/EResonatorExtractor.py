import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from ui_selectors import SNPoints, SPeakParams
from utils.algorithms import make_cut
from utils.unit_transformations import convert_dB_to_linear, convert_linear_to_dB
from validators import VResonatorFit

from .Executor import Executor


def estimate_cavity_params(res_magnitude, resonance_freq, cavity_width, plato, peak_type='maximum'):
    """
    Estimate initial cavity parameters from peak characteristics.
    
    Args:
        res_magnitude: peak magnitude in dB
        resonance_freq: resonance frequency
        cavity_width: width of the cavity/peak
        plato: plateau level in dB
        peak_type: 'maximum' or 'minimum'
    
    Returns:
        dict with estimated parameters
    """
    con = np.abs(convert_dB_to_linear(res_magnitude) - convert_dB_to_linear(plato))

    kappa = con * cavity_width / 2
    beta = cavity_width / 2 * (1 - con)

    return {
        'kappa': kappa,
        'beta': beta,
        'resonance_freq': resonance_freq,
        'plato': plato,
        'res_magnitude': res_magnitude,
        'peak_type': peak_type
    }

def calculate_fit_weights(freqs, resonance_freq, peak_width, decay_factor=1.5):
    """
    Calculate non-uniform weights for fitting with Gaussian-like distribution.
    
    Args:
        freqs: frequency array
        resonance_freq: resonance frequency (center)
        peak_width: width of the peak region
        decay_factor: exponential decay rate (larger = faster decay)
    
    Returns:
        weights: array of weights (maximum at resonance, smoothly decaying)
    """
    freqs = np.asarray(freqs)
    
    # Distance from resonance
    distance = np.abs(freqs - resonance_freq)
    
    # Define peak region boundaries - extend to full peak_width for better edge fitting
    peak_region = peak_width  # Extended from peak_width/2
    
    # Use Gaussian-like weighting with different decay rates inside and outside peak region
    inside_peak = distance <= peak_region
    outside_peak = ~inside_peak
    
    weights = np.zeros_like(freqs, dtype=float)
    
    # Inside peak region: Gaussian decay (slower, keeping weights high)
    # sigma chosen so that at peak_region boundary weight is still significant (~0.5)
    sigma_inside = peak_region / 1.0  # At distance=peak_region: weight ≈ 0.61
    weights[inside_peak] = np.exp(-(distance[inside_peak]**2) / (2 * sigma_inside**2))
    
    # Outside peak region: exponential decay (slower for better edge fitting)
    # Start from the value at the boundary and decay further
    boundary_weight = np.exp(-(peak_region**2) / (2 * sigma_inside**2))
    weights[outside_peak] = boundary_weight * np.exp(-decay_factor * (distance[outside_peak] - peak_region) / peak_region)
    
    return weights

def fit_cavity_response(freqs, s_average, initial_params):
    """
    Two-stage cavity response fitting for improved accuracy.
    Performs fitting ONLY on data points within the peak region.
    
    Stage 1: Preliminary fit on peak region to find accurate peak position
    Stage 2: Final fit with recentered peak region
    """
    # Determine peak type (maximum or minimum)
    res_magnitude = initial_params.get('res_magnitude')
    plato = initial_params.get('plato')
    
    if res_magnitude is not None and plato is not None:
        # If peak magnitude > plateau → maximum, otherwise → minimum
        is_maximum = res_magnitude > plato
    else:
        # Default to maximum if not specified
        is_maximum = True
    
    # Select appropriate cavity model
    cavity_model = cavity_model_max if is_maximum else cavity_model_min
    
    # Get initial estimates
    initial_resonance_freq = initial_params['resonance_freq']
    peak_width = initial_params['peak_width']  # User-specified peak width
    
    # Convert s_average from dB to linear for fitting
    s_average_linear = convert_dB_to_linear(s_average)
    
    # STAGE 1: Preliminary fit using data ONLY in peak region
    # Define peak region boundaries
    peak_region_mask = np.abs(freqs - initial_resonance_freq) <= peak_width
    freqs_peak = freqs[peak_region_mask]
    s_peak = s_average_linear[peak_region_mask]
    
    # Set up initial parameters and bounds
    p0 = [
        initial_params['kappa'],
        initial_params['beta'],
        initial_params['resonance_freq']
    ]
    
    bounds = (
        [0, 0, min(freqs_peak)],  # Lower bounds
        [np.inf, np.inf, max(freqs_peak)]  # Upper bounds
    )
    
    # Preliminary fit WITHOUT weights, only on peak region data
    try:
        popt_preliminary, _ = sp.optimize.curve_fit(
            cavity_model, freqs_peak, s_peak, 
            p0=p0, bounds=bounds,
            maxfev=1000000,
            ftol=1e-12,
            xtol=1e-12
        )
        
        # Extract refined resonance frequency from preliminary fit
        refined_resonance_freq = popt_preliminary[2]
        
    except RuntimeError:
        # If preliminary fit fails, use initial parameters
        refined_resonance_freq = initial_resonance_freq
        popt_preliminary = p0

    # STAGE 2: Final fit with peak region recentered on refined resonance
    peak_region_mask_final = np.abs(freqs - refined_resonance_freq) <= peak_width
    freqs_peak_final = freqs[peak_region_mask_final]
    s_peak_final = s_average_linear[peak_region_mask_final]
    
    # Update bounds for refined peak region
    bounds_final = (
        [0, 0, min(freqs_peak_final)],
        [np.inf, np.inf, max(freqs_peak_final)]
    )

    # Use preliminary fit results as starting point for final fit
    p0_final = popt_preliminary

    # High-precision fitting on peak region only, WITHOUT weights
    popt, _ = sp.optimize.curve_fit(
        cavity_model, freqs_peak_final, s_peak_final, 
        p0=p0_final, bounds=bounds_final,
        maxfev=10000000,  # Increase max function evaluations
        ftol=1e-15,       # Function tolerance
        xtol=1e-15,       # Parameter tolerance
        gtol=1e-15        # Gradient tolerance
    )

    fitted_params = {
        'kappa': popt[0],
        'beta': popt[1],
        'resonance_freq': popt[2],
        'plato': initial_params['plato']  # Keep original plato from initial params
    }
    
    # Calculate peak response (without plato added yet)
    peak_response_linear = cavity_model(
        fitted_params['resonance_freq'],
        fitted_params['kappa'],
        fitted_params['beta'],
        fitted_params['resonance_freq']
    )
    
    # Add plato and convert to dB
    fitted_params['res_magnitude'] = convert_linear_to_dB(
        convert_dB_to_linear(fitted_params['plato']) + peak_response_linear
    )
    fitted_params['peak_type'] = 'maximum' if is_maximum else 'minimum'

    return fitted_params

def cavity_model_max(f, kappa, beta, f0):
    """
    Cavity response model for maximum (absorption peak pointing up).
    
    Args:
        f: frequency array
        kappa, beta: cavity parameters
        f0: resonance frequency
    
    Returns:
        Response in LINEAR scale (without plateau)
    """
    delta_f = f - f0
    response = kappa / np.sqrt(delta_f**2 + (kappa + beta)**2)
    return response

def cavity_model_min(f, kappa, beta, f0):
    """
    Cavity response model for minimum (transmission dip pointing down).
    
    Args:
        f: frequency array
        kappa, beta: cavity parameters
        f0: resonance frequency
    
    Returns:
        Response in LINEAR scale (negative, without plateau)
    """
    delta_f = f - f0
    response = 1 - (kappa / np.sqrt(delta_f**2 + (kappa + beta)**2))
    return response

class EResonatorExtractor(Executor):
    def __init__(self, data: dict, stage_name: str, axis: str = "x", initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        self.axis = axis
        self.fit_enabled = self.initial_params.get("fit", True)

    def validate(self):
        validator = VResonatorFit(self.stage_name, self.data_for_visualization)
        plt.show()
        return validator.get_params().get("Validation")

    def select_initial_params(self):
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

        # Determine peak type from user input or by comparing peak and plateau
        peak_type = self.initial_params.get("peak_type")
        if peak_type is None:
            # Auto-detect: if peak > plateau → maximum, else → minimum
            peak_type = "maximum" if res_magnitude > plato else "minimum"
        
        initial_cavity = estimate_cavity_params(res_magnitude, resonance_freq, cavity_width, plato, peak_type)
        # Add peak_width to initial_cavity for weighted fitting
        initial_cavity['peak_width'] = cavity_width
        
        fitted = None
        if self.fit_enabled:
            fitted = fit_cavity_response(freqs, s_values, initial_cavity)

        fit_params = fitted if fitted is not None else initial_cavity
        
        # Determine which model to use based on peak type
        peak_type = self.initial_params.get("peak_type")
        if peak_type == "minimum":
            model_func = cavity_model_min
        else:
            model_func = cavity_model_max
        
        # cavity_model works with linear values (without plato), add plato and convert result to dB
        model_response = model_func(
            freqs, fit_params["kappa"], fit_params["beta"], 
            fit_params["resonance_freq"])
        
        # Add plateau and convert to dB
        fit_curve = convert_linear_to_dB(model_response)

        # Use fitted resonance frequency for visualization if fitting was performed
        display_resonance_freq = fit_params["resonance_freq"] if fitted else resonance_freq
        display_res_magnitude = fit_params["res_magnitude"] if fitted else res_magnitude

        self.result = {
            "axis": self.axis,
            "cut_value": cut_value,
            "cut": cut,
            "initial_params": initial_cavity,
            "fitted_params": fitted,
            "fit_enabled": self.fit_enabled,
            "fit_curve": fit_curve,
            "resonance_freq": display_resonance_freq,
            "res_magnitude": display_res_magnitude,
            "cavity_width": cavity_width,
            "plato": plato,
        }

    def prepare_for_visualization(self):
        self.data_for_visualization = self.result