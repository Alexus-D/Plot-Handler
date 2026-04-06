r"""Variant of EAnticrossingApproximator that allows setting custom fitting bounds.

Identical to :class:`EAnticrossingApproximator` except that the interactive
parameter-selection UI exposes **lower and upper fitting bounds** for each
free parameter (J, alpha, gamma) in addition to the initial guess.

The bounds are stored in ``initial_params`` and forwarded to
``scipy.optimize.curve_fit`` via the overridden :meth:`_get_fit_bounds`.

Additional ``initial_params`` keys
-----------------------------------
- ``J_min``     / ``J_max``     : bounds for coupling strength J     (default: 0, inf)
- ``alpha_min`` / ``alpha_max`` : bounds for magnon damping alpha    (default: 0, inf)
- ``gamma_min`` / ``gamma_max`` : bounds for additional decay gamma  (default: 0, inf)
"""

import numpy as np
import matplotlib.pyplot as plt

from ui_selectors import S2DApproximatorBoundsParams
from utils.phys_models import simple_anticrossing_model

from .EAnticrossingApproximator import EAnticrossingApproximator


class EBorderAApproximator(EAnticrossingApproximator):
    """
    Variant of EAnticrossingApproximator with user-selectable fitting bounds.

    The interactive selector (:class:`S2DApproximatorBoundsParams`) shows:

    - **Lower bound** TextBox — minimum allowed value during fitting
    - **Initial guess** Slider — starting point for the optimiser
    - **Upper bound** TextBox — maximum allowed value during fitting

    All other behaviour (polygon masking, 3-panel visualisation,
    validation, saving) is identical to :class:`EAnticrossingApproximator`.

    Input ``data`` dict: same as :class:`EAnticrossingApproximator`.

    ``initial_params`` (optional):
        - J_init / alpha_init / gamma_init : initial guesses (same as parent)
        - J_min  / J_max                   : bounds for J     (default: 0, inf)
        - alpha_min / alpha_max            : bounds for alpha (default: 0, inf)
        - gamma_min / gamma_max            : bounds for gamma (default: 0, inf)

    Result dict: same as :class:`EAnticrossingApproximator`.
    """

    # ------------------------------------------------------------------
    # Bounds hook (overrides parent stub)
    # ------------------------------------------------------------------

    def _get_fit_bounds(self):
        """Return bounds read from ``initial_params`` for use in ``curve_fit``."""
        J_min     = float(self.initial_params.get("J_min",     0.0))
        J_max     = float(self.initial_params.get("J_max",     np.inf))
        alpha_min = float(self.initial_params.get("alpha_min", 0.0))
        alpha_max = float(self.initial_params.get("alpha_max", np.inf))
        gamma_min = float(self.initial_params.get("gamma_min", 0.0))
        gamma_max = float(self.initial_params.get("gamma_max", np.inf))
        return ([J_min, alpha_min, gamma_min], [J_max, alpha_max, gamma_max])

    # ------------------------------------------------------------------
    # Interactive parameter / bounds selection
    # ------------------------------------------------------------------

    def select_initial_params(self):
        """Interactive selection of initial guesses AND fitting bounds."""
        if self.initial_params is None:
            self.initial_params = {}

        has_all_params  = all(k in self.initial_params for k in ["J_init", "alpha_init", "gamma_init"])
        skip_interactive = self.initial_params.get("skip_interactive", False)

        if has_all_params or skip_interactive:
            # Non-interactive: apply defaults for any missing keys
            self.initial_params.setdefault("J_init",     0.01)
            self.initial_params.setdefault("alpha_init", 0.05)
            self.initial_params.setdefault("gamma_init", 0.05)
            self.initial_params.setdefault("J_min",      0.0)
            self.initial_params.setdefault("J_max",      np.inf)
            self.initial_params.setdefault("alpha_min",  0.0)
            self.initial_params.setdefault("alpha_max",  np.inf)
            self.initial_params.setdefault("gamma_min",  0.0)
            self.initial_params.setdefault("gamma_max",  np.inf)
            self._print_params()
            return

        # ---- Interactive mode ----------------------------------------
        kappa            = float(self.data.get("kappa",         0.1))
        beta             = float(self.data.get("beta",          0.01))
        resonance_freq   = float(self.data.get("resonance_freq", 4.0))
        magnon_slope     = float(self.data.get("slope",         0.001))
        magnon_intercept = float(self.data.get("intercept",     3.0))

        def model_preview(freq_grid, field_grid, J, alpha, gamma):
            fields_u = np.unique(field_grid[:, 0])
            freqs_u  = np.unique(freq_grid[0, :])
            n_fields = len(fields_u)
            params = {
                "alpha":                np.full(n_fields, alpha),
                "beta":                 beta,
                "kappa":                kappa,
                "gamma":                np.full(n_fields, gamma),
                "J":                    np.full(n_fields, J),
                "magnon_freq_slope":    magnon_slope,
                "magnon_freq_intercept": magnon_intercept,
                "resonance_freq":       resonance_freq,
            }
            return simple_anticrossing_model(freqs_u, fields_u, params)

        plot_data = {
            "x":           self.data["x"],
            "y":           self.data["y"],
            "z":           self.data["z"],
            "model_func":  model_preview,
            "param_names": ["J", "alpha", "gamma"],
            "param_bounds": {
                "J":     (0.001, 0.1,  0.01),
                "alpha": (0.001, 0.2,  0.05),
                "gamma": (0.001, 0.2,  0.05),
            },
            "known_params": {},
            "xlabel": "Field (Oe)",
            "ylabel": "Frequency (GHz)",
            "zlabel": "S21 (dB)",
        }

        selector = S2DApproximatorBoundsParams(self.stage_name, plot_data)
        plt.show()

        selected = selector.get_params()

        self.initial_params["J_init"]     = selected.get("J_init",     0.01)
        self.initial_params["alpha_init"] = selected.get("alpha_init", 0.05)
        self.initial_params["gamma_init"] = selected.get("gamma_init", 0.05)
        self.initial_params["J_min"]      = selected.get("J_min",      0.0)
        self.initial_params["J_max"]      = selected.get("J_max",      np.inf)
        self.initial_params["alpha_min"]  = selected.get("alpha_min",  0.0)
        self.initial_params["alpha_max"]  = selected.get("alpha_max",  np.inf)
        self.initial_params["gamma_min"]  = selected.get("gamma_min",  0.0)
        self.initial_params["gamma_max"]  = selected.get("gamma_max",  np.inf)

        if selected.get("polygon_mask") is not None:
            self.initial_params["polygon_mask"] = selected["polygon_mask"]

        self._print_params()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _print_params(self):
        ip = self.initial_params
        print("[EBorderAApproximator] Parameters:")
        print(f"  J     = {ip['J_init']:.4f}  "
              f"[{ip['J_min']:.4g}, {ip['J_max']:.4g}]")
        print(f"  alpha = {ip['alpha_init']:.4f}  "
              f"[{ip['alpha_min']:.4g}, {ip['alpha_max']:.4g}]")
        print(f"  gamma = {ip['gamma_init']:.4f}  "
              f"[{ip['gamma_min']:.4g}, {ip['gamma_max']:.4g}]")
        if "polygon_mask" in ip:
            print(f"  Polygon mask: {np.sum(ip['polygon_mask'])} points selected")
