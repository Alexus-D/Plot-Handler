"""Loader for EAnticrossingApproximator results (alias for Load2DApproximator)."""

from .Load2DApproximator import Load2DApproximator


class LoadEAnticrossingApproximator(Load2DApproximator):
    """
    Loader / saver for :class:`EAnticrossingApproximator` results.
    
    This is now an alias for the universal :class:`Load2DApproximator`.
    
    Result dict format::

        {
            "J":                  float,  # GHz - coupling strength (fitted)
            "alpha":              float,  # GHz - magnon damping rate (fitted)
            "gamma":              float,  # GHz - additional decay rate (fitted)
            "kappa":              float,  # GHz - cavity decay (from resonator)
            "beta":               float,  # GHz - magnon decay (from resonator)
            "resonance_freq":     float,  # GHz - cavity resonance (from resonator)
            "magnon_slope":       float,  # GHz/Oe - magnon dispersion slope
            "magnon_intercept":   float,  # GHz - magnon dispersion intercept
        }

    Text file format::

        # 2D Approximator results
        # Parameters:
        J: 0.01234
        alpha: 0.04567
        beta: 0.01
        gamma: 0.05123
        kappa: 0.05
        magnon_intercept: 2.5
        magnon_slope: 0.001
        resonance_freq: 3.8
    """
    pass
