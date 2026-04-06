r"""Executor that fits the full 2D contour data to a simple anticrossing model.

Fits the transmission using the anticrossing interaction model between cavity and magnon modes.

The model computes the linear response for cavity-magnon coupling:

$$S_{21}(f, H) = 1 + \frac{\kappa}{\text{cavity\_term} - \text{coupling}^2 / \text{magnon\_term}}$$

where:
- cavity_term = $i(f - f_c) - (\kappa + \beta)$
- magnon_term = $i(f - f_m(H)) - (\alpha + \gamma)$  
- coupling = $iJ + \Gamma$, where $\Gamma = \kappa \cdot \gamma$
- $f_m(H) = \text{slope} \cdot H + \text{intercept}$ — magnon frequency (from calibration)

Fitting parameters:
- $J$ — coupling strength (GHz)
- $\alpha$ — magnon damping rate (GHz)
- $\gamma$ — additional decay rate (GHz)"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from utils.unit_transformations import convert_dB_to_linear, convert_linear_to_dB
from utils.phys_models import simple_anticrossing_model
from plotters import PTotalApproximatorComparison
from ui_selectors import S2DApproximatorParams
from validators import V2DApproximatorFit

from .Executor import Executor


class EAnticrossingApproximator(Executor):
    """
    Fits the full 2D contour data to the simple anticrossing model.

    Input ``data`` dict:
        - x : 1-D array of magnetic field values (Oe)
        - y : 1-D array of frequency values (GHz)
        - z : 2-D array shape (n_fields, n_freqs) — S-parameter data in **dB**
        - kappa : cavity decay rate (GHz) — from resonator extraction
        - beta : magnon decay rate (GHz) — from resonator extraction
        - resonance_freq : cavity resonance frequency (GHz) — from resonator extraction
        - slope : magnon dispersion slope (GHz/Oe) — from magnon calibration
        - intercept : magnon dispersion intercept (GHz) — from magnon calibration

    ``initial_params`` (optional):
        - J_init : initial guess for coupling strength (default: 0.01 GHz)
        - alpha_init : initial guess for magnon damping (default: 0.05 GHz)
        - gamma_init : initial guess for additional decay (default: 0.05 GHz)

    Result dict (``get_result()``):
        - J : fitted coupling strength (GHz)
        - alpha : fitted magnon damping rate (GHz)
        - gamma : fitted additional decay rate (GHz)
        - kappa : cavity decay rate (from input)
        - beta : magnon decay rate (from input)
        - resonance_freq : cavity resonance frequency (from input)
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
            print("[EAnticrossingApproximator] No visualization data available.")
            return True
        
        validator = V2DApproximatorFit(self.stage_name, self.data_for_visualization)
        plt.show()
        
        validation_result = validator.get_params().get("Validation", False)
        return validation_result

    def execute(self):
        # Extract known parameters from resonator and magnon calibration
        kappa          = float(self.data.get("kappa", 0.1))          # cavity decay
        beta           = float(self.data.get("beta", 0.01))          # magnon decay
        resonance_freq = float(self.data.get("resonance_freq", 4.0)) # cavity frequency
        magnon_slope     = float(self.data.get("slope", 0.001))
        magnon_intercept = float(self.data.get("intercept", 3.0))

        # Extract experimental data
        fields = self.data["x"]  # 1D, Oe
        freqs  = self.data["y"]  # 1D, GHz
        z_dB   = self.data["z"]  # 2D, shape (n_fields, n_freqs), dB

        # Convert dB → linear
        z_linear = convert_dB_to_linear(z_dB)

        # Create meshgrid for masking
        field_grid, freq_grid = np.meshgrid(fields, freqs, indexing='ij')

        # Flatten for curve_fit
        z_flat = np.abs(z_linear.flatten())
        
        # Apply polygon mask if provided
        polygon_mask = self.initial_params.get("polygon_mask")
        if polygon_mask is not None:
            mask_flat = polygon_mask.flatten()
            z_flat = z_flat[mask_flat]
            print(f"[EAnticrossingApproximator] Using polygon mask: {len(z_flat)} / {len(z_linear.flatten())} points")

        # Initial guesses for fitting parameters
        J_init     = float(self.initial_params.get("J_init", 0.01))
        alpha_init = float(self.initial_params.get("alpha_init", 0.05))
        gamma_init = float(self.initial_params.get("gamma_init", 0.05))
        p0 = [J_init, alpha_init, gamma_init]

        # Bounds: all three parameters must be positive
        bounds = self._get_fit_bounds()

        # Define the model as a function that returns flattened output
        def model_flat(dummy, J, alpha, gamma):
            """
            Wrapper for simple_anticrossing_model that works with curve_fit.
            
            Note: curve_fit requires the function signature to be f(x, *params),
            but we don't use x since we capture fields/freqs from outer scope.
            """
            n_fields = len(fields)
            
            # Create parameter dict with constant values expanded to arrays
            params = {
                'alpha': np.full(n_fields, alpha),
                'beta': beta,
                'kappa': kappa,
                'gamma': np.full(n_fields, gamma),
                'J': np.full(n_fields, J),
                'magnon_freq_slope': magnon_slope,
                'magnon_freq_intercept': magnon_intercept,
                'resonance_freq': resonance_freq,
            }
            
            response = simple_anticrossing_model(freqs, fields, params)
            response_flat = np.abs(response).flatten()
            
            # Apply mask if it exists
            if polygon_mask is not None:
                response_flat = response_flat[mask_flat]
            
            return response_flat

        print("[EAnticrossingApproximator] Starting global fit...")
        print(f"  Known params: kappa={kappa:.4f}, beta={beta:.4f}, f_c={resonance_freq:.4f}")
        print(f"  Known params: magnon_slope={magnon_slope:.6f}, intercept={magnon_intercept:.4f}")
        print(f"  Initial guess: J={J_init:.4f}, alpha={alpha_init:.4f}, gamma={gamma_init:.4f}")

        try:
            # Create dummy x array for curve_fit (not actually used in model_flat)
            dummy_x = np.arange(len(z_flat))
            
            popt, pcov = scipy.optimize.curve_fit(
                model_flat,
                dummy_x,
                z_flat,
                p0=p0,
                bounds=bounds,
                method='trf',  # Trust Region Reflective for bounded problems
                maxfev=5000000,
            )
            J_fit, alpha_fit, gamma_fit = popt
        except Exception as exc:
            print(f"[EAnticrossingApproximator] Fit failed: {exc}")
            J_fit = J_init
            alpha_fit = alpha_init
            gamma_fit = gamma_init

        # Compute Gamma = sqrt(kappa * gamma)
        Gamma_fit = np.sqrt(kappa * gamma_fit)
        
        print(f"[EAnticrossingApproximator] Fit complete:")
        print(f"  J = {J_fit:.6f} GHz")
        print(f"  alpha = {alpha_fit:.6f} GHz")
        print(f"  gamma = {gamma_fit:.6f} GHz")
        print(f"  Gamma = {Gamma_fit:.6f} GHz")

        # Compute reconstructed data for visualization
        n_fields = len(fields)
        params_fit = {
            'alpha': np.full(n_fields, alpha_fit),
            'beta': beta,
            'kappa': kappa,
            'gamma': np.full(n_fields, gamma_fit),
            'J': np.full(n_fields, J_fit),
            'magnon_freq_slope': magnon_slope,
            'magnon_freq_intercept': magnon_intercept,
            'resonance_freq': resonance_freq,
        }
        
        response_fit = simple_anticrossing_model(freqs, fields, params_fit)
        z_fit_dB = convert_linear_to_dB(np.abs(response_fit))

        # Include all data needed for plotting in result
        self.result = {
            "J":                   J_fit,
            "alpha":               alpha_fit,
            "gamma":               gamma_fit,
            "Gamma":               Gamma_fit,
            "kappa":               kappa,
            "beta":                beta,
            "resonance_freq":      resonance_freq,
            "magnon_slope":        magnon_slope,
            "magnon_intercept":    magnon_intercept,
            # Add data for visualization/plotting
            "x":                   fields,
            "y":                   freqs,
            "z":                   z_dB,
            "z_fit":               z_fit_dB,
        }

        # Copy result to data_for_visualization and add plotting metadata
        self.data_for_visualization = self.result.copy()
        self.data_for_visualization.update({
            "xlabel": "Field (Oe)",
            "ylabel": "Frequency (GHz)",
            "zlabel": "S21 (dB)",
            "title": "Anticrossing Model Fit",
        })

    def select_initial_params(self):
        """Interactive selection of polygon region and initial parameters with live preview."""
        # Ensure initial_params is a dict
        if self.initial_params is None:
            self.initial_params = {}
        
        # Check if all parameters are already provided (non-interactive mode for tests)
        has_all_params = all(key in self.initial_params for key in ["J_init", "alpha_init", "gamma_init"])
        skip_interactive = self.initial_params.get("skip_interactive", False)
        
        if has_all_params or skip_interactive:
            # Use provided values or defaults
            self.initial_params.setdefault("J_init", 0.01)
            self.initial_params.setdefault("alpha_init", 0.05)
            self.initial_params.setdefault("gamma_init", 0.05)
            print(f"[EAnticrossingApproximator] Using initial parameters:")
            print(f"  J_init = {self.initial_params['J_init']:.4f}")
            print(f"  alpha_init = {self.initial_params['alpha_init']:.4f}")
            print(f"  gamma_init = {self.initial_params['gamma_init']:.4f}")
            return
        
        # Interactive mode
        # Extract known parameters
        kappa = float(self.data.get("kappa", 0.1))
        beta = float(self.data.get("beta", 0.01))
        resonance_freq = float(self.data.get("resonance_freq", 4.0))
        magnon_slope = float(self.data.get("slope", 0.001))
        magnon_intercept = float(self.data.get("intercept", 3.0))
        
        # Model function for preview (wraps simple_anticrossing_model)
        def model_preview(freq_grid, field_grid, J, alpha, gamma):
            """Wrapper for anticrossing model that works with S2DApproximatorParams."""
            fields = np.unique(field_grid[:, 0])
            freqs = np.unique(freq_grid[0, :])
            n_fields = len(fields)
            
            params = {
                'alpha': np.full(n_fields, alpha),
                'beta': beta,
                'kappa': kappa,
                'gamma': np.full(n_fields, gamma),
                'J': np.full(n_fields, J),
                'magnon_freq_slope': magnon_slope,
                'magnon_freq_intercept': magnon_intercept,
                'resonance_freq': resonance_freq,
            }
            
            return simple_anticrossing_model(freqs, fields, params)
        
        # Prepare plot data for selector
        plot_data = {
            "x": self.data["x"],
            "y": self.data["y"],
            "z": self.data["z"],
            "model_func": model_preview,
            "param_names": ["J", "alpha", "gamma"],
            "param_bounds": {
                "J": (0.001, 0.1, 0.01),           # (min, max, default)
                "alpha": (0.001, 0.2, 0.05),
                "gamma": (0.001, 0.2, 0.05),
            },
            "known_params": {},  # All known params passed via closure
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
        self.initial_params["J_init"] = selected.get("J_init", 0.01)
        self.initial_params["alpha_init"] = selected.get("alpha_init", 0.05)
        self.initial_params["gamma_init"] = selected.get("gamma_init", 0.05)
        
        # Store polygon mask if defined
        if selected.get("polygon_mask") is not None:
            self.initial_params["polygon_mask"] = selected["polygon_mask"]
        
        print(f"[EAnticrossingApproximator] Initial parameters selected:")
        print(f"  J_init = {self.initial_params['J_init']:.4f}")
        print(f"  alpha_init = {self.initial_params['alpha_init']:.4f}")
        print(f"  gamma_init = {self.initial_params['gamma_init']:.4f}")
        if "polygon_mask" in self.initial_params:
            print(f"  Polygon mask: {np.sum(self.initial_params['polygon_mask'])} points selected")

    def _get_fit_bounds(self):
        """Return (lower_bounds, upper_bounds) for curve_fit. Override in subclasses for custom bounds."""
        return ([0, 0, 0], [np.inf, np.inf, np.inf])

    def prepare_for_visualization(self):
        """Creates 3-panel comparison plot."""
        if self.data_for_visualization is None:
            print("[EAnticrossingApproximator] No data for visualization.")
            return

        self.plotter = PTotalApproximatorComparison(self.data_for_visualization)
        self.figure = self.plotter.get_figure()
