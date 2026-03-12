from plotters import PCutResonatorFit
from .Validator import Validator


class VResonatorFit(Validator):
    """
    Validator for resonator fitting results.
    Shows experimental data points, fitted/model curve, and peak boundaries.
    
    Expected plot_data keys:
    - cut: dict with 'x' and 'y' keys (experimental data)
    - fit_curve: array of fitted values
    - resonance_freq: float, resonance frequency
    - cavity_width: float, width of the cavity/peak
    - axis: str, 'x' or 'y'
    - cut_value: float
    """
    
    def __init__(self, stage_name: str, plot_data: dict):
        super().__init__(stage_name, plot_data)

    def _create_plotter(self, plot_data: dict):
        # Extract cut data
        cut = plot_data.get("cut", {})
        x = cut.get("x")
        y = cut.get("y")
        
        # Extract fit curve
        fit_curve = plot_data.get("fit_curve")
        
        # Extract peak parameters for visualization
        resonance_freq = plot_data.get("resonance_freq")
        cavity_width = plot_data.get("cavity_width")
        
        # Prepare data for plotter
        axis = plot_data.get("axis", "x")
        cut_value = plot_data.get("cut_value")
        
        plotter_data = {
            "x": x,
            "y": y,
            "fit_curve": fit_curve,
            "resonance_freq": resonance_freq,
            "peak_width": cavity_width,
            "title": f"Resonator Fit Validation (cut {axis}={cut_value:.4g})",
            "xlabel": "Frequency",
            "ylabel": "S21 (dB)",
        }
        
        return PCutResonatorFit(plotter_data)
