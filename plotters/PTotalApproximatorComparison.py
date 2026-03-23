from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt


class PTotalApproximatorComparison(Plotter):
    """
    Universal plotter for comparing experimental and fitted data from 2D approximators.
    
    Displays three contour plots side-by-side:
    1. Experimental data
    2. Fitted/reconstructed data
    3. Residual (difference)
    
    Expected plot_data keys:
    - x: array, field values (Oe)
    - y: array, frequency values (GHz)
    - z: 2D array, experimental data in dB
    - z_fit: 2D array, fitted data in dB
    - xlabel: str, x-axis label (default: "Field (Oe)")
    - ylabel: str, y-axis label (default: "Frequency (GHz)")
    - zlabel: str, colorbar label (default: "S21 (dB)")
    - title: str, main title (default: "2D Approximator: Comparison")
    
    Any additional numeric parameters (excluding known_params) will be displayed
    as fitted parameters in the middle panel title.
    """
    
    def __init__(self, plot_data):
        super().__init__(plot_data)

    def create_figure(self):
        x = self.plot_data.get("x")
        y = self.plot_data.get("y")
        z = self.plot_data.get("z")
        z_fit = self.plot_data.get("z_fit")

        if x is None or y is None or z is None or z_fit is None:
            raise ValueError("plot_data must contain 'x', 'y', 'z', and 'z_fit' keys.")

        x = np.array(x)
        y = np.array(y)
        z = np.array(z)
        z_fit = np.array(z_fit)

        # Compute residual
        residual = z - z_fit

        # Create meshgrid for contour plots
        X, Y = np.meshgrid(x, y)

        # Create figure with 3 subplots
        self.figure, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Get labels
        xlabel = self.plot_data.get("xlabel", "Field (Oe)")
        ylabel = self.plot_data.get("ylabel", "Frequency (GHz)")
        zlabel = self.plot_data.get("zlabel", "S21 (dB)")
        
        # Get fitted parameters for display (exclude visualization and known params)
        viz_keys = {"x", "y", "z", "z_fit", "xlabel", "ylabel", "zlabel", "title"}
        known_keys = {"kappa", "kappa_c", "beta", "kappa_tot", "f_c", "resonance_freq",
                     "magnon_slope", "magnon_intercept", "slope", "intercept"}
        
        fitted_params = {k: v for k, v in self.plot_data.items() 
                        if k not in viz_keys and k not in known_keys 
                        and isinstance(v, (int, float))}

        # 1. Experimental data
        levels = 50
        contour0 = axes[0].contourf(X, Y, z.T, levels=levels, cmap='viridis')
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title("Experimental Data")
        cbar0 = plt.colorbar(contour0, ax=axes[0])
        cbar0.set_label(zlabel)

        # 2. Fitted data
        contour1 = axes[1].contourf(X, Y, z_fit.T, levels=levels, cmap='viridis')
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        
        # Add fitted parameters to title if available
        title_fit = "Fitted Data"
        if fitted_params:
            param_str = ", ".join([f"{k}={v:.4f}" for k, v in sorted(fitted_params.items())])
            title_fit += f"\n{param_str}"
        axes[1].set_title(title_fit, fontsize=10)
        
        cbar1 = plt.colorbar(contour1, ax=axes[1])
        cbar1.set_label(zlabel)

        # 3. Residual (difference)
        # Use 'seismic' colormap for residuals (red-white-blue)
        # Center colormap at zero
        vmax_res = np.max(np.abs(residual))
        contour2 = axes[2].contourf(X, Y, residual.T, levels=levels, 
                                     cmap='seismic', vmin=-vmax_res, vmax=vmax_res)
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel(ylabel)
        
        # Compute RMS error for display
        rms_error = np.sqrt(np.mean(residual**2))
        axes[2].set_title(f"Residual\nRMS = {rms_error:.4f} dB")
        
        cbar2 = plt.colorbar(contour2, ax=axes[2])
        cbar2.set_label("Residual (dB)")

        # Main title
        main_title = self.plot_data.get("title", "2D Approximator: Comparison")
        self.figure.suptitle(main_title, fontsize=14, fontweight='bold')

        plt.tight_layout(rect=(0, 0, 1, 0.96))  # Leave space for suptitle

        return self.figure
    
    def redraw(self):
        if self.figure is None:
            raise ValueError("No plot has been created to redraw.")
        plt.clf()
        self.create_figure()

    def get_axis_for_marker(self):
        if self.figure is None:
            raise ValueError("No plot has been created to get axis from.")
        # Return the first axis (experimental data plot)
        return self.figure.axes[0]
