"""Loader for ETotalApproximator results (alias for Load2DApproximator)."""

from .Load2DApproximator import Load2DApproximator


class LoadETotalApproximator(Load2DApproximator):
    """
    Loader / saver for :class:`ETotalApproximator` results.
    
    This is now an alias for the universal :class:`Load2DApproximator`.
    
    Result dict format::

        {
            "g":                  float,  # GHz - coupling constant (fitted)
            "kappa_m":            float,  # GHz - magnon decay rate (fitted)
            "gamma_tot":          float,  # GHz - total magnon decay (fitted)
            "kappa_c":            float,  # GHz - cavity decay (from resonator)
            "beta":               float,  # GHz - additional losses (from resonator)
            "kappa_tot":          float,  # GHz - total cavity decay
            "f_c":                float,  # GHz - cavity resonance (from resonator)
            "magnon_slope":       float,  # GHz/Oe - magnon dispersion slope
            "magnon_intercept":   float,  # GHz - magnon dispersion intercept
        }

    Text file format::

        # 2D Approximator results
        # Parameters:
        beta: 0.01
        f_c: 3.8
        g: 0.01234
        gamma_tot: 0.05123
        kappa_c: 0.05
        kappa_m: 0.04567
        kappa_tot: 0.06
        magnon_intercept: 2.5
        magnon_slope: 0.001
    """
    pass
