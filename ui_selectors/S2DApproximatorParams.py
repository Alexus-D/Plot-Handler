"""Interactive selector for 2D approximator: polygon region + sliders for initial params + live preview."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector, Slider, Button
from matplotlib.path import Path

from .Selector import Selector
from plotters import PContour


class S2DApproximatorParams(Selector):
    """
    Interactive selector for 2D approximator initial parameters.
    
    Features:
    1. Click to define polygon region for fitting (avoids noisy areas)
    2. Sliders to adjust initial parameter guesses
    3. Live preview: three plots (experiment, theory, residual)
    4. Done button to accept and proceed
    
    Expected plot_data keys:
    - x: 1D array, field values (Oe)
    - y: 1D array, frequency values (GHz)
    - z: 2D array, experimental S21 data (dB)
    - model_func: callable, model function(freq_grid, field_grid, **params)
    - param_names: list of str, parameter names for sliders
    - param_bounds: dict, {param_name: (min, max, default)}
    - known_params: dict, fixed parameters passed to model
    """
    
    def __init__(self, stage_name: str, plot_data: dict):
        self.x = np.array(plot_data["x"])
        self.y = np.array(plot_data["y"])
        self.z = np.array(plot_data["z"])
        self.model_func = plot_data["model_func"]
        self.param_names = plot_data["param_names"]
        self.param_bounds = plot_data["param_bounds"]
        self.known_params = plot_data.get("known_params", {})
        
        # Polygon selection
        self.polygon_vertices = []
        self.polygon_mask = None
        self.polygon_active = True
        
        # Current parameter values (from sliders)
        self.current_params = {name: bounds[2] for name, bounds in self.param_bounds.items()}
        
        # Store selector for polygon
        self.poly_selector = None
        
        # Standard Selector initialization
        buttons = [("toggle_polygon", "Toggle Polygon Mode"), ("done", "Done")]
        super().__init__(stage_name, plot_data, buttons, clear_button=False, save_button=False)
        
        # Increase figure size and create 3 subplots
        self.figure.set_size_inches(18, 6)
        
        # Clear default plot and create 3 subplots
        self.figure.clear()
        self._create_three_plots()
        
        # Recreate buttons after clearing figure
        self._create_custom_buttons()
        
        # Create sliders
        self._create_sliders()
        
        # Create polygon selector
        self._activate_polygon_selector()
        
        # Initial model computation
        self._update_model()
        
    def _create_plotter(self, plot_data: dict):
        """Create dummy plotter (we'll create 3 subplots manually)."""
        # Create empty figure that Selector expects
        from plotters.Plotter import Plotter
        class DummyPlotter(Plotter):
            def create_figure(self):
                return plt.figure()
            def get_axis_for_marker(self):
                return None
            def redraw(self):
                pass
        return DummyPlotter({})
    
    def _create_three_plots(self):
        """Create three side-by-side plots: Experiment, Model, Residual."""
        # Position: [left, bottom, width, height]
        plot_width = 0.27
        plot_height = 0.60
        plot_bottom = 0.30
        plot_spacing = 0.03
        plot_left = 0.05
        
        self.axes = []
        titles = ["Experiment", "Model", "Residual"]
        
        for i, title in enumerate(titles):
            ax = self.figure.add_axes([
                plot_left + i * (plot_width + plot_spacing),
                plot_bottom,
                plot_width,
                plot_height
            ])
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Field (Oe)", fontsize=11)
            ax.set_ylabel("Frequency (GHz)", fontsize=11)
            ax.tick_params(labelsize=10)
            self.axes.append(ax)
        
        # Initial empty plots - use standard 'xy' indexing
        X, Y = np.meshgrid(self.x, self.y)
        for ax in self.axes:
            ax.contourf(X, Y, np.zeros_like(X), levels=20, cmap='viridis')
    
    def _create_markers(self, plotter):
        """No markers needed."""
        return {}
    
    def _create_custom_buttons(self):
        """Create buttons in custom positions."""
        button_width = 0.12
        button_height = 0.04
        button_left = 0.87
        button_spacing = 0.06
        button_bottom = 0.50
        
        button_configs = [
            ("toggle_polygon", "Toggle Polygon"),
            ("done", "Done")
        ]
        
        self.buttons = {}
        for i, (mode, label) in enumerate(button_configs):
            ax_button = self.figure.add_axes([button_left, button_bottom + i * button_spacing, 
                                               button_width, button_height])
            button = Button(ax_button, label)
            button.on_clicked(lambda event, m=mode: self._set_mode(m))
            self.buttons[mode] = button
    
    def _create_sliders(self):
        """Create sliders for each fitting parameter."""
        self.sliders = {}
        slider_height = 0.025
        slider_spacing = 0.045
        slider_left = 0.15
        slider_width = 0.70
        slider_bottom = 0.05
        
        for i, param_name in enumerate(self.param_names):
            min_val, max_val, default_val = self.param_bounds[param_name]
            
            # Create axis for slider
            ax_slider = self.figure.add_axes((slider_left, slider_bottom + i * slider_spacing, 
                                              slider_width, slider_height))
            
            slider = Slider(
                ax_slider, 
                param_name, 
                min_val, 
                max_val, 
                valinit=default_val,
                valstep=(max_val - min_val) / 100.0
            )
            
            # Connect slider to update function
            slider.on_changed(lambda val, p=param_name: self._on_slider_change(p, val))
            
            self.sliders[param_name] = slider
    
    def _activate_polygon_selector(self):
        """Activate polygon selector on first plot (Experiment)."""
        ax = self.axes[0]
        
        self.poly_selector = PolygonSelector(
            ax,
            self._on_polygon_complete,
            useblit=True,
            props=dict(color='red', linestyle='-', linewidth=2, alpha=0.5),
            handle_props=dict(markersize=8, markerfacecolor='red'),
            grab_range=10
        )
        
        # Ensure the figure has focus for keyboard events
        self.figure.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        print("[S2DApproximatorParams] Polygon mode active.")
        print("  - Click to add vertices")
        print("  - Press Enter to complete polygon")
        print("  - Press Escape to cancel")
    
    def _on_key_press(self, event):
        """Handle keyboard events for polygon completion."""
        if event.key == 'enter':
            if hasattr(self.poly_selector, 'verts') and len(self.poly_selector.verts) >= 3:
                # Manually trigger completion
                self._on_polygon_complete(self.poly_selector.verts)
                print("[S2DApproximatorParams] Polygon completed with Enter key.")
        elif event.key == 'escape':
            if self.poly_selector:
                self.poly_selector.clear()
                print("[S2DApproximatorParams] Polygon cancelled.")
    
    def _on_polygon_complete(self, vertices):
        """Called when polygon is completed. PolygonSelector passes vertices as [(x,y), ...] or array."""
        vertices = np.array(vertices)
        
        if len(vertices) < 3:
            print("[S2DApproximatorParams] Polygon needs at least 3 vertices.")
            return
        
        self.polygon_vertices = vertices
        print(f"[S2DApproximatorParams] Polygon defined with {len(vertices)} vertices.")
        
        # Create mask for fitting region
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        points = np.vstack([X.flatten(), Y.flatten()]).T
        
        path = Path(vertices)
        mask_flat = path.contains_points(points)
        self.polygon_mask = mask_flat.reshape(X.shape)
        
        print(f"[S2DApproximatorParams] Mask created: {np.sum(self.polygon_mask)} points inside polygon.")
        
        # Update model with new mask
        self._update_model()
    
    def _on_slider_change(self, param_name, value):
        """Called when any slider changes."""
        self.current_params[param_name] = value
        self._update_model()
    
    def _update_model(self):
        """Recompute model with current parameters and update all three plots."""
        # Create meshgrid
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Compute model
        try:
            # Call model function with current slider values
            model_result = self.model_func(Y, X, **self.known_params, **self.current_params)
            
            # Convert to dB (assuming model returns linear amplitude)
            from utils.unit_transformations import convert_linear_to_dB
            z_model = convert_linear_to_dB(np.abs(model_result))
            
        except Exception as e:
            print(f"[S2DApproximatorParams] Model computation failed: {e}")
            return
        
        # Compute residual
        residual = self.z - z_model
        
        # Update all three plots
        self._update_three_plots(self.z, z_model, residual)
    
    def _update_three_plots(self, z_exp, z_model, residual):
        """Update the three contour plots."""
        # Standard matplotlib convention: meshgrid with default 'xy' indexing
        # X.shape = (len(y), len(x)), data must be transposed to match
        X, Y = np.meshgrid(self.x, self.y)  # shape: (len(freqs), len(fields))
        
        # Our data has shape (len(fields), len(freqs)), so transpose for contourf
        datasets = [z_exp.T, z_model.T, residual.T]
        titles = ["Experiment", "Model", "Residual"]
        cmaps = ['viridis', 'viridis', 'seismic']
        
        for i, (ax, data, title, cmap) in enumerate(zip(self.axes, datasets, titles, cmaps)):
            ax.clear()
            
            if i == 2:  # Residual - center colormap at zero
                vmax = np.max(np.abs(data))
                im = ax.contourf(X, Y, data, levels=20, cmap=cmap, vmin=-vmax, vmax=vmax)
            else:
                im = ax.contourf(X, Y, data, levels=20, cmap=cmap)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel("Field (Oe)", fontsize=11)
            ax.set_ylabel("Frequency (GHz)", fontsize=11)
            ax.tick_params(labelsize=10)
            
            # Draw polygon if defined (only on first plot)
            if i == 0 and len(self.polygon_vertices) > 0:
                poly_array = np.array(self.polygon_vertices)
                ax.plot(poly_array[:, 0], poly_array[:, 1], 'r-', linewidth=2)
                ax.plot([poly_array[-1, 0], poly_array[0, 0]], 
                       [poly_array[-1, 1], poly_array[0, 1]], 'r-', linewidth=2)
        
        self.figure.canvas.draw_idle()
    
    def _set_mode(self, mode):
        """Handle button clicks."""
        if mode == "toggle_polygon":
            self.polygon_active = not self.polygon_active
            if self.polygon_active:
                self._activate_polygon_selector()
                print("[S2DApproximatorParams] Polygon mode activated.")
            else:
                if self.poly_selector:
                    self.poly_selector.disconnect_events()
                print("[S2DApproximatorParams] Polygon mode deactivated.")
            return
        
        if mode == "done":
            # Store results
            self.params = {
                "polygon_vertices": self.polygon_vertices,
                "polygon_mask": self.polygon_mask,
                **{f"{name}_init": value for name, value in self.current_params.items()}
            }
            print(f"[S2DApproximatorParams] Parameters saved: {self.current_params}")
            plt.close(self.figure)
            return
        
        super()._set_mode(mode)
    
    def mouse_on_click(self, event):
        """Polygon selector handles clicks."""
        pass
