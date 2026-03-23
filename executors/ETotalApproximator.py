r"""Executor that fits the full 2D contour data to a coupled-resonator model.

Fits the transmission $S_{21}(f, H)$ using the formula:

$$S_{21}(f, H) = 1 - \frac{\kappa_c (i\Delta_m - \kappa_m/2) + \kappa_m (i\Delta_c - \kappa_{tot}/2) - 2\sqrt{\kappa_c \kappa_m} g}{(i\Delta_c - \kappa_{tot}/2)(i\Delta_m - \kappa_m/2) + g^2}$$

where:
- $\kappa_c$ — cavity coupling to waveguide (from resonator data: 'kappa')
- $\beta = \kappa_{int}^c$ — cavity internal losses (from resonator data: 'beta')
- $\kappa_{tot} = \kappa_c + \beta$ — total cavity decay
- $f_c$ — cavity resonance frequency (from resonator data: 'resonance_freq')
- $\Delta_c = 2\pi(f - f_c)$ — cavity detuning
- $f_m(H) = \text{slope} \cdot H + \text{intercept}$ — magnon frequency (from calibration)
- $\Delta_m = 2\pi(f - f_m(H))$ — magnon detuning

Fitting parameters:
- $g$ — coherent coupling constant
- $\kappa_m$ — magnon total decay rate
- $\gamma_{tot}$ — total magnon decay (including intrinsic losses)"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from utils.unit_transformations import convert_dB_to_linear, convert_linear_to_dB
from plotters import PTotalApproximatorComparison
from ui_selectors import S2DApproximatorParams
from validators import V2DApproximatorFit

from .Executor import Executor


def S21_coupled_resonators(freq_grid, field_grid, kappa_c, kappa_tot, f_c, magnon_slope, magnon_intercept, g, kappa_m, gamma_tot):
    """
    Coupled-resonator transmission model for 2D (freq, field) data.

    Parameters:
    -----------
    freq_grid : ndarray, shape (n_fields, n_freqs)
        Frequency meshgrid (GHz)
    field_grid : ndarray, shape (n_fields, n_freqs)
        Magnetic field meshgrid (Oe)
    kappa_c : float
        Cavity coupling to waveguide (GHz)
    kappa_tot : float
        Total cavity decay rate (GHz): kappa_c + beta
    f_c : float
        Cavity resonance frequency (GHz)
    magnon_slope : float
        Magnon dispersion slope (GHz/Oe)
    magnon_intercept : float
        Magnon dispersion intercept (GHz)
    g : float
        Coherent coupling constant (GHz)
    kappa_m : float
        Magnon total decay rate (GHz)
    gamma_tot : float
        Total magnon decay including intrinsic losses (GHz)

    Returns:
    --------
    S21 : ndarray, complex, shape (n_fields, n_freqs)
        Transmission coefficient (linear scale)
    """
    # Magnon frequency from calibration
    f_m = magnon_slope * field_grid + magnon_intercept

    # Detunings (angular frequency)
    Delta_c = (freq_grid - f_c)
    Delta_m = (freq_grid - f_m)

    # Numerator
    numerator = (
        kappa_c * (1j * Delta_m - gamma_tot / 2)
        + kappa_m * (1j * Delta_c - kappa_tot / 2)
        - 2 * np.sqrt(kappa_c * kappa_m) * g
    )

    # Denominator
    denominator = (
        (1j * Delta_c - kappa_tot / 2) * (1j * Delta_m - gamma_tot / 2)
        + g ** 2
    )

    S21 = 1 - numerator / denominator
    return S21


class ETotalApproximator(Executor):
    """
    Fits the full 2D contour data to the coupled-resonator S21 model.

    Input ``data`` dict:
        - x : 1-D array of magnetic field values (Oe)
        - y : 1-D array of frequency values (GHz)
        - z : 2-D array shape (n_fields, n_freqs) — S-parameter data in **dB**
        - kappa : cavity coupling to waveguide (GHz) — from resonator extraction
        - beta : cavity internal losses (GHz) — from resonator extraction
        - resonance_freq : cavity resonance frequency (GHz) — from resonator extraction
        - slope : magnon dispersion slope (GHz/Oe) — from magnon calibration
        - intercept : magnon dispersion intercept (GHz) — from magnon calibration

    ``initial_params`` (optional):
        - g_init : initial guess for coupling constant (default: 0.01 GHz)
        - kappa_m_init : initial guess for magnon decay (default: 0.05 GHz)
        - gamma_tot_init : initial guess for total magnon decay (default: 0.05 GHz)

    Result dict (``get_result()``):
        - g : fitted coupling constant (GHz)
        - kappa_m : fitted magnon decay rate (GHz)
        - gamma_tot : fitted total magnon decay (GHz)
        - kappa_c : cavity coupling (from input)
        - beta : cavity internal losses (from input)
        - kappa_tot : total cavity decay (kappa_c + beta)
        - f_c : cavity resonance frequency (from input)
        - magnon_slope : magnon dispersion slope (from input)
        - magnon_intercept : magnon dispersion intercept (from input)
    """

    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)

    # ------------------------------------------------------------------ #
    # Executor interface                                                   #
    # ------------------------------------------------------------------ #

    def validate(self) -> bool:
        """Show three-panel comparison and get user validation."""
        # Ensure initial_params is a dict
        if self.initial_params is None:
            self.initial_params = {}
        
        # Skip interactive validation if requested (for tests)
        if self.initial_params.get("skip_interactive", False):
            return True
        
        if not self.data_for_visualization:
            print("[ETotalApproximator] No visualization data available.")
            return True
        
        validator = V2DApproximatorFit(self.stage_name, self.data_for_visualization)
        plt.show()
        
        validation_result = validator.get_params().get("Validation", False)
        return validation_result

    def execute(self):
        # Extract known parameters from resonator and magnon calibration
        kappa_c = float(self.data.get("kappa", 0.1))          # cavity coupling
        beta    = float(self.data.get("beta", 0.01))          # cavity internal loss
        f_c     = float(self.data.get("resonance_freq", 4.0)) # cavity frequency
        magnon_slope     = float(self.data.get("slope", 0.001))
        magnon_intercept = float(self.data.get("intercept", 3.0))

        kappa_tot = kappa_c + beta

        # Extract experimental data
        fields = self.data["x"]  # 1D, Oe
        freqs  = self.data["y"]  # 1D, GHz
        z_dB   = self.data["z"]  # 2D, shape (n_fields, n_freqs), dB

        # Convert dB → linear
        z_linear = convert_dB_to_linear(z_dB)

        # Create meshgrids for the model
        field_grid, freq_grid = np.meshgrid(fields, freqs, indexing='ij')

        # Flatten for curve_fit
        z_flat = z_linear.flatten()
        
        # Apply polygon mask if provided
        polygon_mask = self.initial_params.get("polygon_mask")
        if polygon_mask is not None:
            mask_flat = polygon_mask.flatten()
            z_flat = z_flat[mask_flat]
            print(f"[ETotalApproximator] Using polygon mask: {len(z_flat)} / {len(z_linear.flatten())} points")

        # Initial guesses for fitting parameters
        g_init       = float(self.initial_params.get("g_init", 0.01))
        kappa_m_init = float(self.initial_params.get("kappa_m_init", 0.05))
        gamma_tot_init = float(self.initial_params.get("gamma_tot_init", 0.05))
        p0 = [g_init, kappa_m_init, gamma_tot_init]

        # Bounds: all three parameters must be positive
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])

        # Define the model as a function of flattened coordinates
        def model_flat(dummy, g, kappa_m, gamma_tot):
            # Compute model on full grid
            S21 = S21_coupled_resonators(
                freq_grid, field_grid,
                kappa_c, kappa_tot, f_c,
                magnon_slope, magnon_intercept,
                g, kappa_m, gamma_tot
            )
            result_flat = np.abs(S21).flatten()
            
            # Apply mask if it exists
            if polygon_mask is not None:
                result_flat = result_flat[mask_flat]
            
            return result_flat

        # Prepare dummy x array (not used, but curve_fit requires it)
        dummy_x = np.arange(len(z_flat))

        print("[ETotalApproximator] Starting global fit...")
        print(f"  Known params: kappa_c={kappa_c:.4f}, beta={beta:.4f}, f_c={f_c:.4f}")
        print(f"  Known params: magnon_slope={magnon_slope:.6f}, intercept={magnon_intercept:.4f}")
        print(f"  Initial guess: g={g_init:.4f}, kappa_m={kappa_m_init:.4f}, gamma_tot={gamma_tot_init:.4f}")

        try:
            popt, pcov = scipy.optimize.curve_fit(
                model_flat,
                dummy_x,
                np.abs(z_flat),
                p0=p0,
                bounds=bounds,
                method='trf',  # Trust Region Reflective for bounded problems
                maxfev=5000,
            )
            g_fit, kappa_m_fit, gamma_tot_fit = popt
        except Exception as exc:
            print(f"[ETotalApproximator] Fit failed: {exc}")
            g_fit = g_init
            kappa_m_fit = kappa_m_init
            gamma_tot_fit = gamma_tot_init

        print(f"[ETotalApproximator] Fit complete:")
        print(f"  g = {g_fit:.6f} GHz")
        print(f"  kappa_m = {kappa_m_fit:.6f} GHz")
        print(f"  gamma_tot = {gamma_tot_fit:.6f} GHz")

        # Compute reconstructed data for visualization
        S21_fit = S21_coupled_resonators(
            freq_grid, field_grid,
            kappa_c, kappa_tot, f_c,
            magnon_slope, magnon_intercept,
            g_fit, kappa_m_fit, gamma_tot_fit
        )
        z_fit_dB = convert_linear_to_dB(np.abs(S21_fit))

        # Include all data needed for plotting in result
        self.result = {
            "g":                g_fit,
            "kappa_m":          kappa_m_fit,
            "gamma_tot":        gamma_tot_fit,
            "kappa_c":          kappa_c,
            "beta":             beta,
            "kappa_tot":        kappa_tot,
            "f_c":              f_c,
            "magnon_slope":     magnon_slope,
            "magnon_intercept": magnon_intercept,
            # Add data for visualization/plotting
            "x":                fields,
            "y":                freqs,
            "z":                z_dB,
            "z_fit":            z_fit_dB,
        }

        # Copy result to data_for_visualization and add plotting metadata
        self.data_for_visualization = self.result.copy()
        self.data_for_visualization.update({
            "xlabel": "Field (Oe)",
            "ylabel": "Frequency (GHz)",
            "zlabel": "S21 (dB)",
            "title": "Total Approximator: Coupled Resonator Model Fit",
        })

    def select_initial_params(self):
        """Interactive selection of polygon region and initial parameters with live preview."""
        # Ensure initial_params is a dict
        if self.initial_params is None:
            self.initial_params = {}
        
        # Check if all parameters are already provided (non-interactive mode for tests)
        has_all_params = all(key in self.initial_params for key in ["g_init", "kappa_m_init", "gamma_tot_init"])
        skip_interactive = self.initial_params.get("skip_interactive", False)
        
        if has_all_params or skip_interactive:
            # Use provided values or defaults
            self.initial_params.setdefault("g_init", 0.01)
            self.initial_params.setdefault("kappa_m_init", 0.05)
            self.initial_params.setdefault("gamma_tot_init", 0.05)
            print(f"[ETotalApproximator] Using initial parameters:")
            print(f"  g_init = {self.initial_params['g_init']:.4f}")
            print(f"  kappa_m_init = {self.initial_params['kappa_m_init']:.4f}")
            print(f"  gamma_tot_init = {self.initial_params['gamma_tot_init']:.4f}")
            return
        
        # Interactive mode
        # Extract known parameters
        kappa_c = float(self.data.get("kappa", 0.1))
        beta = float(self.data.get("beta", 0.01))
        f_c = float(self.data.get("resonance_freq", 4.0))
        magnon_slope = float(self.data.get("slope", 0.001))
        magnon_intercept = float(self.data.get("intercept", 3.0))
        kappa_tot = kappa_c + beta
        
        # Prepare plot data for selector
        plot_data = {
            "x": self.data["x"],
            "y": self.data["y"],
            "z": self.data["z"],
            "model_func": lambda freq_grid, field_grid, g, kappa_m, gamma_tot: S21_coupled_resonators(
                freq_grid, field_grid, kappa_c, kappa_tot, f_c, 
                magnon_slope, magnon_intercept, g, kappa_m, gamma_tot
            ),
            "param_names": ["g", "kappa_m", "gamma_tot"],
            "param_bounds": {
                "g": (0.001, 0.1, 0.01),           # (min, max, default)
                "kappa_m": (0.001, 0.2, 0.05),
                "gamma_tot": (0.001, 0.2, 0.05),
            },
            "known_params": {},  # All known params are passed via model_func closure
            "xlabel": "Field (Oe)",
            "ylabel": "Frequency (GHz)",
            "zlabel": "S21 (dB)",
        }
        
        # Create selector and show
        selector = S2DApproximatorParams(self.stage_name, plot_data)
        plt.show()
        
        # Get selected parameters
        selected = selector.get_params()
        
        # Update initial_params with slider values
        self.initial_params["g_init"] = selected.get("g_init", 0.01)
        self.initial_params["kappa_m_init"] = selected.get("kappa_m_init", 0.05)
        self.initial_params["gamma_tot_init"] = selected.get("gamma_tot_init", 0.05)
        
        # Store polygon mask if defined
        if selected.get("polygon_mask") is not None:
            self.initial_params["polygon_mask"] = selected["polygon_mask"]
        
        print(f"[ETotalApproximator] Initial parameters selected:")
        print(f"  g_init = {self.initial_params['g_init']:.4f}")
        print(f"  kappa_m_init = {self.initial_params['kappa_m_init']:.4f}")
        print(f"  gamma_tot_init = {self.initial_params['gamma_tot_init']:.4f}")
        if "polygon_mask" in self.initial_params:
            print(f"  Polygon mask: {np.sum(self.initial_params['polygon_mask'])} points selected")

    def prepare_for_visualization(self):
        """Create visualization with three comparison plots."""
        if self.data_for_visualization:
            self.plotter = PTotalApproximatorComparison(self.data_for_visualization)
            self.figure = self.plotter.get_figure()
