"""Interactive selector for 2D approximator: polygon region + sliders for initial params
+ TextBoxes for fitting bounds + live preview.

Extends :class:`S2DApproximatorParams` by showing, for each fitting parameter::

    [lower bound TextBox]  [initial guess Slider]  [upper bound TextBox]

The lower/upper bound values become the ``bounds`` argument passed to
``scipy.optimize.curve_fit`` in the executor.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

from .S2DApproximatorParams import S2DApproximatorParams


class S2DApproximatorBoundsParams(S2DApproximatorParams):
    """
    Extension of S2DApproximatorParams that also exposes adjustable fitting bounds.

    For each fitting parameter the bottom panel shows three controls:

    - **Lower bound** TextBox  (editable float, validated on submission)
    - **Initial guess** Slider (same live-preview behaviour as the parent)
    - **Upper bound** TextBox  (editable float or ``inf``)

    ``get_params()`` returns all keys from the parent plus:

    - ``{name}_min`` : lower fitting bound for each parameter
    - ``{name}_max`` : upper fitting bound for each parameter
    """

    # ------------------------------------------------------------------
    # Slider + TextBox creation (overrides parent)
    # ------------------------------------------------------------------

    def _create_sliders(self):
        """Create one row per parameter: [min TextBox] [init Slider] [max TextBox]."""
        self.sliders = {}
        self.text_boxes_min = {}
        self.text_boxes_max = {}
        self.bounds_min = {}
        self.bounds_max = {}

        slider_height  = 0.025
        slider_spacing = 0.055   # slightly larger than parent to avoid crowding
        slider_bottom  = 0.06

        # Layout fractions (figure is 18" wide × 6" tall)
        min_tb_left  = 0.02
        min_tb_width = 0.06
        slider_left  = 0.10
        slider_width = 0.57
        max_tb_left  = 0.69
        max_tb_width = 0.06

        # Column-header labels
        n = len(self.param_names)
        header_y = slider_bottom + n * slider_spacing + 0.005
        self.figure.text(min_tb_left  + min_tb_width  / 2, header_y,
                         "Lower bound",  ha="center", va="bottom", fontsize=9, color="dimgray")
        self.figure.text(slider_left  + slider_width  / 2, header_y,
                         "Initial guess", ha="center", va="bottom", fontsize=9, color="dimgray")
        self.figure.text(max_tb_left  + max_tb_width  / 2, header_y,
                         "Upper bound",  ha="center", va="bottom", fontsize=9, color="dimgray")

        for i, param_name in enumerate(self.param_names):
            min_val, max_val, default_val = self.param_bounds[param_name]
            row_bottom = slider_bottom + i * slider_spacing

            # Store initial bounds
            self.bounds_min[param_name] = min_val
            self.bounds_max[param_name] = max_val

            # --- Lower bound TextBox ---
            ax_min = self.figure.add_axes(
                (min_tb_left, row_bottom, min_tb_width, slider_height)
            )
            tb_min = TextBox(ax_min, "", initial=f"{min_val:.4g}")
            tb_min.on_submit(lambda text, p=param_name: self._on_min_change(p, text))
            self.text_boxes_min[param_name] = tb_min

            # --- Initial guess Slider ---
            ax_slider = self.figure.add_axes(
                (slider_left, row_bottom, slider_width, slider_height)
            )
            slider = Slider(
                ax_slider,
                param_name,
                min_val,
                max_val,
                valinit=default_val,
                valstep=(max_val - min_val) / 100.0,
            )
            slider.on_changed(lambda val, p=param_name: self._on_slider_change(p, val))
            self.sliders[param_name] = slider

            # --- Upper bound TextBox ---
            ax_max = self.figure.add_axes(
                (max_tb_left, row_bottom, max_tb_width, slider_height)
            )
            tb_max = TextBox(ax_max, "", initial=f"{max_val:.4g}")
            tb_max.on_submit(lambda text, p=param_name: self._on_max_change(p, text))
            self.text_boxes_max[param_name] = tb_max

    # ------------------------------------------------------------------
    # TextBox callbacks
    # ------------------------------------------------------------------

    def _on_min_change(self, param_name: str, text: str):
        """Validate and store the lower bound, then update the slider range."""
        try:
            val = float(text.strip())
        except ValueError:
            print(f"[S2DApproximatorBoundsParams] Invalid lower bound for {param_name}: '{text}'")
            return
        if val < self.bounds_max.get(param_name, np.inf):
            self.bounds_min[param_name] = val
            print(f"[S2DApproximatorBoundsParams] {param_name} lower bound → {val:.6g}")
            self._update_slider_range(param_name)
        else:
            print(f"[S2DApproximatorBoundsParams] Lower bound must be < upper bound.")

    def _on_max_change(self, param_name: str, text: str):
        """Validate and store the upper bound, then update the slider range."""
        s = text.strip().lower()
        try:
            val = np.inf if s in ("inf", "infinity", "") else float(s)
        except ValueError:
            print(f"[S2DApproximatorBoundsParams] Invalid upper bound for {param_name}: '{text}'")
            return
        if val > self.bounds_min.get(param_name, 0.0):
            self.bounds_max[param_name] = val
            print(f"[S2DApproximatorBoundsParams] {param_name} upper bound → {val:.6g}")
            self._update_slider_range(param_name)
        else:
            print(f"[S2DApproximatorBoundsParams] Upper bound must be > lower bound.")

    def _update_slider_range(self, param_name: str):
        """Resize the slider to [bounds_min, bounds_max], clamping the current value."""
        slider = self.sliders[param_name]
        new_min = self.bounds_min[param_name]
        new_max = self.bounds_max[param_name]

        # Slider cannot display infinity — keep the visual max if the bound is inf
        slider_min = new_min
        slider_max = new_max if np.isfinite(new_max) else slider.valmax

        if slider_min >= slider_max:
            slider_max = slider_min + 1e-9  # safety guard

        slider.valmin  = slider_min
        slider.valmax  = slider_max
        slider.valstep = (slider_max - slider_min) / 100.0
        slider.ax.set_xlim(slider_min, slider_max)

        # Clamp the slider's current value into the new range
        clamped = float(np.clip(slider.val, slider_min, slider_max))
        if clamped != slider.val:
            slider.set_val(clamped)
            self.current_params[param_name] = clamped

        self.figure.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Override 'done' to include bounds in saved parameters
    # ------------------------------------------------------------------

    def _set_mode(self, mode: str):
        """Override to append bounds to the params dict when 'done' is clicked."""
        if mode == "done":
            self.params = {
                "polygon_vertices": self.polygon_vertices,
                "polygon_mask":     self.polygon_mask,
                **{f"{name}_init": value for name, value in self.current_params.items()},
                **{f"{name}_min":  self.bounds_min.get(name, 0.0)    for name in self.param_names},
                **{f"{name}_max":  self.bounds_max.get(name, np.inf) for name in self.param_names},
            }
            print(f"[S2DApproximatorBoundsParams] Init values:   {self.current_params}")
            print(f"[S2DApproximatorBoundsParams] Lower bounds:  {self.bounds_min}")
            print(f"[S2DApproximatorBoundsParams] Upper bounds:  {self.bounds_max}")
            plt.close(self.figure)
            return
        # All other modes (toggle_polygon, etc.) handled by the parent chain
        super()._set_mode(mode)
