"""Validator for 2D approximator fit results with three-panel comparison."""

import matplotlib.pyplot as plt
from plotters import PTotalApproximatorComparison
from .Validator import Validator


class V2DApproximatorFit(Validator):
    """
    Validator for 2D approximator fitting results.
    Shows three-panel comparison: experimental data, fitted model, and residual.
    
    Expected plot_data keys:
    - x: 1D array, field values (Oe)
    - y: 1D array, frequency values (GHz)
    - z: 2D array, experimental data (dB)
    - z_fit: 2D array, fitted model data (dB)
    - xlabel: str, x-axis label
    - ylabel: str, y-axis label
    - zlabel: str, colorbar label
    - title: str, main title
    - All fitted parameters (e.g., g, kappa_m, gamma_tot or J, alpha, gamma)
    """
    
    def __init__(self, stage_name: str, plot_data: dict):
        super().__init__(stage_name, plot_data, clear_button=False, save_button=False)
    
    def _create_plotter(self, plot_data: dict):
        """Create three-panel comparison plotter."""
        return PTotalApproximatorComparison(plot_data)
