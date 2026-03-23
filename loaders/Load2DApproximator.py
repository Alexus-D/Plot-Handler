"""Universal loader for 2D approximators (ETotalApproximator, EAnticrossingApproximator, etc.)."""

import os
import numpy as np
import matplotlib.pyplot as plt

from .Loader import Loader


class Load2DApproximator(Loader):
    """
    Universal loader / saver for 2D approximator results.

    Works with any 2D fitting executor that returns a result dict with:
    - Fitted parameters (any number, any names)
    - Known parameters from previous steps
    - Optional visualization data: x, y, z, z_fit

    Result dict format (flexible)::

        {
            # Any fitted parameters
            "param1": float,
            "param2": float,
            ...
            
            # Any known parameters
            "known_param1": float,
            ...
            
            # Optional: visualization data
            "x": np.ndarray,
            "y": np.ndarray,
            "z": np.ndarray (2D),
            "z_fit": np.ndarray (2D),
        }

    Text file format::

        # 2D Approximator results
        # Parameters:
        param1: 0.01234
        param2: 0.04567
        ...
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.data_path   = self.params.get("data_path",   None)
        self.result_path = self.params.get("result_path", None)

    # ------------------------------------------------------------------ #
    # Load                                                                #
    # ------------------------------------------------------------------ #

    def load_data(self) -> dict:
        if self.data_path is None:
            raise ValueError("data_path is not set.")

        data = {}

        with open(self.data_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" not in line:
                    continue

                key, raw_values = line.split(":", 1)
                key        = key.strip()
                raw_values = raw_values.strip()

                # All values are floats
                try:
                    data[key] = float(raw_values)
                except ValueError:
                    continue

        return data

    # ------------------------------------------------------------------ #
    # Save                                                                #
    # ------------------------------------------------------------------ #

    def save_data(self, data: dict):
        if self.result_path is None:
            raise ValueError("result_path is not set.")

        # Filter out visualization arrays (x, y, z, z_fit)
        viz_keys = {"x", "y", "z", "z_fit", "xlabel", "ylabel", "zlabel", "title"}
        
        # Save all parameters (fitted + known)
        params_to_save = {k: v for k, v in data.items() 
                         if k not in viz_keys and isinstance(v, (int, float))}

        with open(self.result_path, "w", encoding="utf-8") as f:
            f.write("# 2D Approximator results\n\n")
            f.write("# Parameters:\n")
            
            for key, value in sorted(params_to_save.items()):
                f.write(f"{key}: {value}\n")

    # ------------------------------------------------------------------ #
    # Plotting                                                            #
    # ------------------------------------------------------------------ #

    def plot_and_save(self, data: dict, plots_dir: str = None):
        """
        Generate comparison plots of original vs fitted contour.

        Parameters
        ----------
        data : dict
            Result dictionary with keys:
            - "x":     np.ndarray - field values
            - "y":     np.ndarray - frequency values
            - "z":     np.ndarray (2D) - original data (dB)
            - "z_fit": np.ndarray (2D) - fitted data (dB)
            - Any parameter keys for display

        plots_dir : str, optional
            Directory to save plots. If None, uses self.result_path directory.
        """
        if plots_dir is None:
            if self.result_path is None:
                raise ValueError("plots_dir is None and result_path is not set.")
            plots_dir = os.path.dirname(self.result_path)

        os.makedirs(plots_dir, exist_ok=True)

        x     = data.get("x")
        y     = data.get("y")
        z     = data.get("z")
        z_fit = data.get("z_fit")

        if x is None or y is None or z is None or z_fit is None:
            print("[Load2DApproximator] Missing data for plotting.")
            return

        # Extract labels
        xlabel = data.get("xlabel", "Field (Oe)")
        ylabel = data.get("ylabel", "Frequency (GHz)")
        zlabel = data.get("zlabel", "S21 (dB)")
        title = data.get("title", "2D Approximator Fit")

        # Collect fitted parameters for display (exclude known/visualization keys)
        viz_keys = {"x", "y", "z", "z_fit", "xlabel", "ylabel", "zlabel", "title"}
        known_keys = {"kappa", "kappa_c", "beta", "kappa_tot", "f_c", "resonance_freq",
                     "magnon_slope", "magnon_intercept", "slope", "intercept"}
        
        fitted_params = {k: v for k, v in data.items() 
                        if k not in viz_keys and k not in known_keys 
                        and isinstance(v, (int, float))}

        # Create figure with 3 subplots: original, fit, residual
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Original data
        levels = 50
        im0 = axes[0].contourf(x, y, z.T, levels=levels, cmap="viridis")
        axes[0].set_xlabel(xlabel)
        axes[0].set_ylabel(ylabel)
        axes[0].set_title("Experimental Data")
        plt.colorbar(im0, ax=axes[0], label=zlabel)

        # 2. Fitted data
        im1 = axes[1].contourf(x, y, z_fit.T, levels=levels, cmap="viridis")
        axes[1].set_xlabel(xlabel)
        axes[1].set_ylabel(ylabel)
        
        # Build title with fitted parameters
        title_fit = "Fitted Data"
        if fitted_params:
            param_str = ", ".join([f"{k}={v:.4f}" for k, v in sorted(fitted_params.items())])
            title_fit += f"\n{param_str}"
        axes[1].set_title(title_fit, fontsize=10)
        
        plt.colorbar(im1, ax=axes[1], label=zlabel)

        # 3. Residual (difference)
        residual = z - z_fit
        vmax_res = np.max(np.abs(residual))
        im2 = axes[2].contourf(x, y, residual.T, levels=levels, 
                               cmap="seismic", vmin=-vmax_res, vmax=vmax_res)
        axes[2].set_xlabel(xlabel)
        axes[2].set_ylabel(ylabel)
        
        rms_error = np.sqrt(np.mean(residual**2))
        axes[2].set_title(f"Residual\nRMS={rms_error:.4f} dB")
        plt.colorbar(im2, ax=axes[2], label="Residual (dB)")

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=(0, 0, 1, 0.96))

        # Save figure
        plot_path = os.path.join(plots_dir, "2d_approximator_fit_comparison.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"[Load2DApproximator] Saved comparison plot to: {plot_path}")

        # Create a 1D cut comparison at middle field
        mid_field_idx = len(x) // 2
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y, z[mid_field_idx, :], 'o-', label='Original', markersize=4)
        ax.plot(y, z_fit[mid_field_idx, :], 's-', label='Fitted', markersize=4)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(zlabel)
        ax.set_title(f"1D Cut at {xlabel.split('(')[0].strip()} = {x[mid_field_idx]:.1f} {xlabel.split('(')[1].strip(')')}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save 1D cut
        cut_path = os.path.join(plots_dir, "2d_approximator_fit_1d_cut.png")
        plt.savefig(cut_path, dpi=150)
        plt.close()

        print(f"[Load2DApproximator] Saved 1D cut plot to: {cut_path}")
