from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt


class PCutWithFit(Plotter):
    """
    General plotter for displaying experimental data and fitted/model curve.
    
    Expected plot_data keys:
    - x: array, x-axis values
    - y: array, experimental y values
    - fit_curve: array, fitted/model y values (optional)
    - title: str, plot title
    - xlabel: str, x-axis label
    - ylabel: str, y-axis label
    """
    
    def __init__(self, plot_data):
        super().__init__(plot_data)

    def create_figure(self):
        x = self.plot_data.get("x")
        y = self.plot_data.get("y")
        fit_curve = self.plot_data.get("fit_curve")

        if x is None or y is None:
            raise ValueError("plot_data must contain 'x' and 'y' keys.")

        x = np.array(x)
        y = np.array(y)

        self.figure, ax = plt.subplots()
        
        # Plot experimental data as points
        ax.plot(x, y, 'o', label='Experimental data', markersize=4, alpha=0.6)
        
        # Plot fit curve if available
        if fit_curve is not None:
            fit_curve = np.array(fit_curve)
            ax.plot(x, fit_curve, '-', label='Model fit', linewidth=2)

        ax.set_title(self.plot_data.get("title", "Fit Validation"))
        ax.set_xlabel(self.plot_data.get("xlabel", "X-axis"))
        ax.set_ylabel(self.plot_data.get("ylabel", "Y-axis"))
        ax.legend()
        ax.grid(True, alpha=0.3)

        return self.figure
    
    def redraw(self):
        if self.figure is None:
            raise ValueError("No plot has been created to redraw.")
        plt.clf()
        self.create_figure()

    def get_axis_for_marker(self):
        if self.figure is None:
            raise ValueError("No plot has been created to get axis from.")
        return self.figure.axes[0]


class PCutResonatorFit(PCutWithFit):
    """
    Specialized plotter for resonator fitting with peak boundaries visualization.
    
    Additional expected plot_data keys (on top of PCutWithFit):
    - resonance_freq: float, resonance frequency
    - peak_width: float, width of the peak region
    """
    
    def create_figure(self):
        # Call parent's create_figure to get the basic plot
        super().create_figure()
        
        # Add resonator-specific visualization
        ax = self.figure.axes[0]
        
        resonance_freq = self.plot_data.get("resonance_freq")
        peak_width = self.plot_data.get("peak_width")
        
        if resonance_freq is not None and peak_width is not None:
            left_boundary = resonance_freq - peak_width / 2
            right_boundary = resonance_freq + peak_width / 2
            
            ax.axvline(left_boundary, color='red', linestyle='--', alpha=0.7, 
                      linewidth=1.5, label='Peak boundaries')
            ax.axvline(right_boundary, color='red', linestyle='--', alpha=0.7, 
                      linewidth=1.5)
            ax.axvline(resonance_freq, color='green', linestyle=':', alpha=0.7, 
                      linewidth=1.5, label='Resonance')
            
            # Refresh legend to include new elements
            ax.legend()
        
        return self.figure
