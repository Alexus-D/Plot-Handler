"""Interruptible executor that fits a Lorentzian profile to every field step
of ALL peak trajectories from IEPeakWatcher / IEPeakWatcherPolyline.

Four-panel live display:
  Top-left     : original experimental contour plot (static)
  Top-right    : current cut + Lorentzian fit (updated each step)
  Bottom-left  : reconstructed contour — sum of ALL fitted Lorentzians
                 (rows where no trajectory has been fitted yet are blank)
  Bottom-right : gamma (HWHM) vs. field, one line per trajectory
"""

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector

from markers import MPoints
from executors import ECut, EPeakParams
from utils.phys_models import Lorentzian
from utils.unit_transformations import convert_dB_to_linear, convert_linear_to_dB

from .InterruptibleExecutor import InterruptibleExecutor


class IELorentzianApproximator(InterruptibleExecutor):
    """
    Iterates over ALL trajectories from IEPeakWatcher / IEPeakWatcherPolyline
    and fits a Lorentzian to each field step using dB→linear→dB conversion.

    The reconstructed contour (bottom-left) accumulates contributions from
    every fitted trajectory.  At a given field row:
      reconstruction_linear(f) = mean(y0_fitted) + Σ_j A_j·γ_j²/((f−x0_j)²+γ_j²)
    where the sum runs only over trajectories already fitted at that field.
    Trajectories not yet reached contribute zero (no dip).  Rows where no
    trajectory has been fitted at all are shown as NaN (blank).

    Input ``data`` dict:
        - x            : 1-D array of field values
        - y            : 1-D array of frequency values
        - z            : 2-D array, shape (n_fields, n_freqs), values in dB
        - trajectories : list of trajectory dicts from IEPeakWatcher /
          IEPeakWatcherPolyline.  Each element must contain the keys
          ``fields``, ``freq``, ``width``, ``magnitude``, ``prominence``.

    ``initial_params`` (all optional):
        - fit_window_factor : float – fit window = width * factor (default 1.0).

    Result dict (``get_result()``):
        - trajectories : list of dicts, one per input trajectory:
            - fields : list[float]
            - x0     : list[float]  – fitted centres (GHz)
            - gamma  : list[float]  – HWHM in GHz
            - A      : list[float]  – amplitude in linear scale
            - y0     : list[float]  – baseline in linear scale
            - model  : "Lorentzian"
    """

    def __init__(
        self,
        data: dict,
        stage_name: str,
        initial_params: dict = None,
        save_figure_path: str = None,
    ):
        super().__init__(data, stage_name, initial_params, save_figure_path)

        initial_params = initial_params or {}
        self._fit_window_factor = float(initial_params.get("fit_window_factor", 1.0))

        self.input_trajectories = data.get("trajectories", [])
        if not self.input_trajectories:
            raise ValueError(
                "No trajectories in input data. Run IEPeakWatcher first."
            )

        self._current_traj_idx  = 0
        self._continue_from_idx = None
        self._start_traj_idx    = 0       # for resuming after interruption

        # Current cut stored for live top-right display
        self._current_cut_freqs  = None
        self._current_cut_values = None   # dB
        self._current_fit_freqs  = None
        self._current_fit_values = None   # dB

        # Reconstructed z array: shape (n_fields, n_freqs); NaN = not yet fitted
        self._z_reconstructed = np.full_like(data["z"], np.nan)

        # Plot references
        self.figure     = None
        self.axes       = None
        self.ax_contour = None
        self.ax_cut     = None
        self.ax_recon   = None
        self.ax_gamma   = None

        self._cut_data_line = None
        self._cut_fit_line  = None
        self._gamma_lines   = []     # one Line2D per trajectory
        self._recon_image   = None

    # ------------------------------------------------------------------ #
    # Executor / InterruptibleExecutor interface                          #
    # ------------------------------------------------------------------ #

    def select_initial_params(self):
        self.result = {"trajectories": []}
        print(
            f"[IELorentzianApproximator] Ready: will fit "
            f"{len(self.input_trajectories)} trajectory(ies)  "
            f"(fit_window_factor={self._fit_window_factor})"
        )

    # ------------------------------------------------------------------ #
    # Batch pre-slicing                                                   #
    # ------------------------------------------------------------------ #

    def _precompute_slices(self, traj_idx: int) -> list:
        """
        Pre-slices all field steps of one trajectory at once using NumPy.

        Returns a list of length ``len(input_fields)`` where each element is
        either ``(fit_freqs, fit_vals_lin, p0, field_idx)`` or ``None`` when the
        fit window is too narrow (< 5 points).

        All field-index look-ups and z-row extractions are done in vectorized
        NumPy calls so the inner fitting loop only ever receives ready-to-use
        arrays.
        """
        input_traj   = self.input_trajectories[traj_idx]
        input_fields = np.asarray(input_traj["fields"])
        input_freqs  = np.asarray(input_traj["freq"])
        input_widths = np.asarray(input_traj["width"])
        input_mags   = np.asarray(input_traj["magnitude"])
        input_proms  = np.asarray(input_traj["prominence"])

        all_freqs = self.data["y"]
        x_arr     = self.data["x"]
        z         = self.data["z"]

        # --- vectorised field-index look-up ----------------------------
        # field_indices[i] = argmin |x_arr - input_fields[i]|
        field_indices = np.abs(
            x_arr[:, None] - input_fields[None, :]
        ).argmin(axis=0)              # shape: (n_steps,)

        # --- batch z-row extraction ------------------------------------
        z_rows = z[field_indices, :]  # shape: (n_steps, n_freqs)

        # --- per-step frequency slicing --------------------------------
        half_wins    = input_widths * self._fit_window_factor / 2.0
        freq_lo      = input_freqs - half_wins
        freq_hi      = input_freqs + half_wins

        slices = []
        for i in range(len(input_fields)):
            mask      = (all_freqs >= freq_lo[i]) & (all_freqs <= freq_hi[i])
            fit_freqs = all_freqs[mask]

            if len(fit_freqs) < 5:
                slices.append(None)
                continue

            fit_vals_lin  = convert_dB_to_linear(z_rows[i, mask])
            magnitude_lin = convert_dB_to_linear(float(input_mags[i]))
            baseline_lin  = convert_dB_to_linear(float(input_mags[i]) + float(input_proms[i]))
            p0 = [
                float(input_freqs[i]),
                float(input_widths[i]) / 2.0,
                magnitude_lin - baseline_lin,
                baseline_lin,
            ]
            slices.append((fit_freqs, fit_vals_lin, p0, int(field_indices[i])))

        return slices

    def interruptible_function(self) -> dict:
        start_traj = self._start_traj_idx
        self._start_traj_idx = 0         # reset so next call starts from 0

        for traj_idx in range(start_traj, len(self.input_trajectories)):
            input_traj             = self.input_trajectories[traj_idx]
            self._current_traj_idx = traj_idx

            # Ensure result list is long enough for this trajectory
            while len(self.result["trajectories"]) <= traj_idx:
                self.result["trajectories"].append(
                    {"fields": [], "x0": [], "gamma": [], "A": [], "y0": [],
                     "model": "Lorentzian"}
                )
            traj_data = self.result["trajectories"][traj_idx]

            start_i = (
                self._continue_from_idx
                if self._continue_from_idx is not None
                else len(traj_data["fields"])
            )
            self._continue_from_idx = None

            input_fields = input_traj["fields"]
            input_freqs  = input_traj["freq"]
            input_widths = input_traj["width"]
            input_mags   = input_traj["magnitude"]
            input_proms  = input_traj["prominence"]

            print(
                f"[IELorentzianApproximator] Trajectory {traj_idx + 1}/"
                f"{len(self.input_trajectories)}: pre-slicing {len(input_fields)} "
                f"field steps …"
            )

            # Pre-compute all slices for this trajectory (vectorised NumPy).
            # When correcting_params is set we must recompute that single step
            # on the fly because the user-supplied window differs from the default.
            precomputed = self._precompute_slices(traj_idx)

            print(
                f"[IELorentzianApproximator] Trajectory {traj_idx + 1}: "
                f"pre-slicing done, fitting from index {start_i}"
            )

            for i in range(start_i, len(input_fields)):
                field = input_fields[i]

                # If the user supplied correcting params for this step,
                # recompute the slice on the fly with the corrected window.
                if self.correcting_params and i == start_i:
                    expected_freq = self.correcting_params.get("peak_freq",  input_freqs[i])
                    width         = self.correcting_params.get("peak_width", input_widths[i])
                    magnitude     = self.correcting_params.get("peak_value", input_mags[i])
                    prominence    = self.correcting_params.get("prominence",  input_proms[i])
                    self.correcting_params = {}

                    x_arr     = self.data["x"]
                    all_freqs = self.data["y"]
                    z         = self.data["z"]

                    field_idx = int(np.abs(x_arr - field).argmin())
                    half_win  = width * self._fit_window_factor / 2.0
                    mask      = (
                        (all_freqs >= expected_freq - half_win)
                        & (all_freqs <= expected_freq + half_win)
                    )
                    fit_freqs_c = all_freqs[mask]

                    if len(fit_freqs_c) < 5:
                        entry = None
                    else:
                        fit_vals_lin  = convert_dB_to_linear(z[field_idx, mask])
                        magnitude_lin = convert_dB_to_linear(magnitude)
                        baseline_lin  = convert_dB_to_linear(magnitude + prominence)
                        p0 = [
                            expected_freq,
                            width / 2.0,
                            magnitude_lin - baseline_lin,
                            baseline_lin,
                        ]
                        entry = (fit_freqs_c, fit_vals_lin, p0, field_idx)
                    # Patch the pre-computed list so subsequent steps are unchanged
                    precomputed[i] = entry
                else:
                    entry = precomputed[i]

                if entry is None:
                    print(
                        f"[IELorentzianApproximator] Fit window too narrow at "
                        f"field={field:.3f} — interrupting"
                    )
                    self.interrupted               = True
                    self.skip_delete_wrong_results = True
                    self._continue_from_idx        = i
                    self._start_traj_idx           = traj_idx
                    return self.result

                fit_freqs, fit_vals_lin, p0, field_idx = entry

                try:
                    popt, _ = scipy.optimize.curve_fit(
                        Lorentzian,
                        fit_freqs,
                        fit_vals_lin,
                        p0=p0,
                        method="lm",
                        maxfev=2000,
                    )
                except Exception as exc:
                    print(
                        f"[IELorentzianApproximator] Fit failed at "
                        f"field={field:.3f}: {exc} — interrupting"
                    )
                    self.interrupted               = True
                    self.skip_delete_wrong_results = True
                    self._continue_from_idx        = i
                    self._start_traj_idx           = traj_idx
                    return self.result

                x0_fit    = popt[0]
                gamma_fit = abs(popt[1])
                A_fit     = popt[2]
                y0_fit    = abs(popt[3])

                traj_data["fields"].append(field)
                traj_data["x0"].append(x0_fit)
                traj_data["gamma"].append(gamma_fit)
                traj_data["A"].append(A_fit)
                traj_data["y0"].append(y0_fit)

                # Recompute the reconstructed row using ALL trajectories' contributions
                self._recompute_reconstructed_at(field_idx)

                fit_freqs_dense          = np.linspace(fit_freqs.min(), fit_freqs.max(), 200)
                self._current_cut_freqs  = fit_freqs
                self._current_cut_values = convert_linear_to_dB(fit_vals_lin)
                self._current_fit_freqs  = fit_freqs_dense
                self._current_fit_values = convert_linear_to_dB(
                    Lorentzian(fit_freqs_dense, x0_fit, gamma_fit, A_fit, y0_fit)
                )

                self._update_line(self.result, draw=False)
                plt.pause(0.001)

                if self.interrupted:
                    print(
                        f"[IELorentzianApproximator] Interrupted by user "
                        f"at trajectory {traj_idx + 1}, step {i + 1}"
                    )
                    self._continue_from_idx = i + 1
                    self._start_traj_idx    = traj_idx
                    return self.result

            if not self._validate_trajectory_end():
                print(f"[IELorentzianApproximator] Trajectory {traj_idx + 1} rejected")
                self.interrupted     = True
                self._start_traj_idx = traj_idx
                return self.result

            print(f"[IELorentzianApproximator] Trajectory {traj_idx + 1} confirmed")
            self._continue_from_idx = None

        return self.result

    # ------------------------------------------------------------------ #
    # Reconstructed-contour helper                                        #
    # ------------------------------------------------------------------ #

    def _recompute_reconstructed_at(self, field_idx: int):
        """
        Rebuilds one row of ``_z_reconstructed`` using contributions from ALL
        trajectories.

        Reconstruction is done in dB space (= multiplicative model in linear):

            S_total_dB(f) = bg_dB + Σ_j [ Lor_j_dB(f) − y0_j_dB ]

        Each term ``Lor_j_dB(f) − y0_j_dB`` is the dip depth of trajectory j
        relative to its own baseline — it is non-positive for absorption dips
        and zero far from the resonance.  Adding in dB space avoids the
        ``total_linear < 0`` artefact that occurs with a linear additive model
        when two deep dips overlap.

        If no trajectory has been fitted at this field yet, the row stays NaN.
        """
        all_freqs    = self.data["y"]
        target_field = self.data["x"][field_idx]

        fitted = []   # (y0_j, A_j, x0_j, gamma_j) per contributing trajectory

        for traj_data in self.result.get("trajectories", []):
            for j, f in enumerate(traj_data["fields"]):
                if abs(f - target_field) < 1e-9:
                    fitted.append((
                        traj_data["y0"][j],
                        traj_data["A"][j],
                        traj_data["x0"][j],
                        traj_data["gamma"][j],
                    ))
                    break

        if not fitted:
            self._z_reconstructed[field_idx, :] = np.nan
            return

        # Shared background = mean of per-trajectory baselines
        background = float(np.mean([y0 for y0, *_ in fitted]))
        total_dB   = np.full(len(all_freqs), convert_linear_to_dB(background), dtype=float)

        for y0_j, A_j, x0_j, gamma_j in fitted:
            lor_lin = y0_j + A_j * gamma_j ** 2 / ((all_freqs - x0_j) ** 2 + gamma_j ** 2)
            # Guard against unphysical fit parameters (e.g. fit found wrong sign)
            np.maximum(lor_lin, 1e-30, out=lor_lin)
            total_dB += convert_linear_to_dB(lor_lin) - convert_linear_to_dB(y0_j)

        self._z_reconstructed[field_idx, :] = total_dB

    # ------------------------------------------------------------------ #
    # Plot setup                                                          #
    # ------------------------------------------------------------------ #

    def _setup_execution_plot(self, result):
        print("[IELorentzianApproximator] Initialising visualisation window")
        plt.ion()

        self.figure, self.axes = plt.subplots(2, 2, figsize=(14, 10))

        # ── Top-left: original contour (static) ────────────────────────
        self.ax_contour = self.axes[0, 0]
        self._draw_contour(self.ax_contour)

        # ── Top-right: current cut + Lorentzian fit ─────────────────────
        self.ax_cut = self.axes[0, 1]
        self.ax_cut.set_xlabel(self.data.get("ylabel", "Frequency"))
        self.ax_cut.set_ylabel(self.data.get("zlabel", "Magnitude (dB)"))
        self.ax_cut.set_title("Current Cut + Lorentzian Fit")
        self.ax_cut.grid(True, alpha=0.3)
        self._cut_data_line, = self.ax_cut.plot(
            [], [], "o", markersize=3, alpha=0.7, label="Data"
        )
        self._cut_fit_line, = self.ax_cut.plot(
            [], [], "-", linewidth=2, color="orange", label="Lorentzian fit"
        )
        self.ax_cut.legend(loc="upper right", fontsize=8)

        # ── Bottom-left: reconstructed contour (live pcolormesh) ────────
        self.ax_recon = self.axes[1, 0]
        self.ax_recon.set_xlabel(self.data.get("xlabel", "Field"))
        self.ax_recon.set_ylabel(self.data.get("ylabel", "Frequency"))
        self.ax_recon.set_title(
            f"Reconstructed Contour "
            f"({len(self.input_trajectories)} Lorentzians)"
        )

        x_arr = self.data["x"]
        y_arr = self.data["y"]
        X, Y  = np.meshgrid(x_arr, y_arr)
        Z_rec_init = np.full((len(y_arr), len(x_arr)), np.nan)

        self._recon_image = self.ax_recon.pcolormesh(
            X, Y, Z_rec_init, cmap="viridis", shading="nearest"
        )
        # Fix colour limits to match the original data for a fair comparison
        z_data = self.data["z"]
        self._recon_image.set_clim(np.nanmin(z_data), np.nanmax(z_data))
        self.figure.colorbar(
            self._recon_image,
            ax=self.ax_recon,
            label=self.data.get("zlabel", "Z"),
        )

        # ── Bottom-right: gamma vs. field — one line per trajectory ─────
        self.ax_gamma = self.axes[1, 1]
        self.ax_gamma.set_xlabel(self.data.get("xlabel", "Field"))
        self.ax_gamma.set_ylabel("gamma  (HWHM, GHz)")
        self.ax_gamma.set_title("Lorentzian Width  gamma  (HWHM)")
        self.ax_gamma.grid(True, alpha=0.3)

        colors = plt.cm.tab10.colors
        self._gamma_lines = []
        for i in range(len(self.input_trajectories)):
            color = colors[i % len(colors)]
            line, = self.ax_gamma.plot(
                [], [], "o-", markersize=3, color=color, label=f"Traj {i + 1}"
            )
            self._gamma_lines.append(line)
        self.ax_gamma.legend(loc="upper right", fontsize=8)

        # Marker for fitted positions on the contour
        self.marker = MPoints(self.ax_contour)

        # Interrupt button (placed by base class helper)
        self._interrupt_btn = self._make_interrupt_button(self.figure)

        plt.tight_layout()
        self.figure.subplots_adjust(bottom=0.1)
        self._update_line(result)  # draw=True — initial synchronous render
        self.figure.canvas.draw()
        print("[IELorentzianApproximator] Visualisation window ready")

    def _draw_contour(self, ax):
        x = self.data["x"]
        y = self.data["y"]
        z = self.data["z"]

        X, Y    = np.meshgrid(x, y)
        Z       = np.array(z).T
        contour = ax.contourf(X, Y, Z, cmap="viridis", levels=50)
        cbar    = self.figure.colorbar(contour, ax=ax)
        cbar.set_label(self.data.get("zlabel", "Z"))
        ax.set_title(self.data.get("title", "Contour Plot"))
        ax.set_xlabel(self.data.get("xlabel", "Field"))
        ax.set_ylabel(self.data.get("ylabel", "Frequency"))

    # ------------------------------------------------------------------ #
    # Live update                                                         #
    # ------------------------------------------------------------------ #

    def _update_line(self, points, draw: bool = True) -> None:
        result    = points if isinstance(points, dict) else self.result
        traj_list = result.get("trajectories", [])

        # All fitted x0 positions from ALL trajectories → markers on contour
        all_points = []
        for traj_data in traj_list:
            all_points.extend(zip(traj_data.get("fields", []), traj_data.get("x0", [])))
        self.marker.update_ticks(all_points)

        # Top-right: current cut + fit
        if self._current_cut_freqs is not None:
            self._cut_data_line.set_data(self._current_cut_freqs, self._current_cut_values)
            self._cut_fit_line.set_data(self._current_fit_freqs,  self._current_fit_values)
            self.ax_cut.relim()
            self.ax_cut.autoscale_view()

        # Bottom-right: gamma lines — one per trajectory
        for line, traj_data in zip(self._gamma_lines, traj_list):
            line.set_data(traj_data.get("fields", []), traj_data.get("gamma", []))
        if traj_list:
            self.ax_gamma.relim()
            self.ax_gamma.autoscale_view()

        # Bottom-left: reconstructed contour
        # _z_reconstructed is (n_fields, n_freqs); pcolormesh expects (n_freqs, n_fields)
        self._recon_image.set_array(self._z_reconstructed.T)

        if draw and self.figure is not None:
            self.figure.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Deletion of wrong results                                           #
    # ------------------------------------------------------------------ #

    def delete_wrong_results(self):
        print("[IELorentzianApproximator] Select rectangle to delete wrong points")
        self._selected_rect     = None
        self._deletion_confirmed = False

        def on_select(eclick, erelease):
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])
            self._selected_rect = (x1, x2, y1, y2)

        rect_selector = RectangleSelector(
            self.ax_contour, on_select,
            useblit=True,
            button=[1],
            interactive=True,
        )

        ax_btn     = self.figure.add_axes([0.35, 0.01, 0.12, 0.05])
        btn_delete = Button(ax_btn, "Delete Selected")

        def on_confirm(event):
            self._deletion_confirmed = True

        btn_delete.on_clicked(on_confirm)

        while not self._deletion_confirmed:
            plt.pause(0.05)

        if self._selected_rect is not None:
            self._remove_points_in_rect(*self._selected_rect)

        ax_btn.remove()
        rect_selector.set_active(False)
        self.figure.canvas.draw_idle()

    def _remove_points_in_rect(self, x1, x2, y1, y2):
        traj_data = self.result["trajectories"][self._current_traj_idx]
        fields    = traj_data["fields"]
        x0s       = traj_data["x0"]
        x_arr     = self.data["x"]

        indices_to_remove = [
            i for i, (f, x0) in enumerate(zip(fields, x0s))
            if x1 <= f <= x2 and y1 <= x0 <= y2
        ]

        for i in reversed(indices_to_remove):
            field_idx = int(np.abs(x_arr - fields[i]).argmin())
            for key in ["fields", "x0", "gamma", "A", "y0"]:
                if key in traj_data and len(traj_data[key]) > i:
                    traj_data[key].pop(i)
            # Recompute this row — the deleted trajectory no longer contributes
            self._recompute_reconstructed_at(field_idx)

        if indices_to_remove:
            self._continue_from_idx = min(indices_to_remove)
            print(
                f"[IELorentzianApproximator] Deleted {len(indices_to_remove)} point(s);"
                f" resuming from index {self._continue_from_idx}"
            )

        self._update_line(self.result)

    # ------------------------------------------------------------------ #
    # Correcting params after interruption                                #
    # ------------------------------------------------------------------ #

    def select_correcting_params(self):
        print("[IELorentzianApproximator] Selecting correcting parameters")

        input_traj = self.input_trajectories[self._current_traj_idx]

        # Determine which field to correct at
        if self._continue_from_idx is not None:
            idx            = self._continue_from_idx
            continue_field = input_traj["fields"][idx]
        else:
            traj_data = self.result["trajectories"][self._current_traj_idx]
            idx       = len(traj_data["fields"])
            if idx < len(input_traj["fields"]):
                continue_field = input_traj["fields"][idx]
            else:
                continue_field = traj_data["fields"][-1] if traj_data["fields"] else input_traj["fields"][0]

        # Make a cut at that field so the user can pick better peak params
        cut_executor = ECut(
            self.data,
            "Correcting params cut",
            axis="x",
            initial_params={"cut_value": continue_field},
        )
        print(
            f"[IELorentzianApproximator] Cut for corrections, "
            f"field={continue_field:.3f}"
        )
        cut_executor.execute()
        cut_data = cut_executor.get_result()

        # Highlight the expected peak position from the input trajectory
        if idx < len(input_traj["freq"]):
            expected_freq = input_traj["freq"][idx]
            field_idx     = int(np.abs(self.data["x"] - continue_field).argmin())
            freq_idx      = int(np.abs(self.data["y"] - expected_freq).argmin())
            mag           = self.data["z"][field_idx, freq_idx]
            cut_data["highlight_points"] = [(expected_freq, mag)]

        peak_executor = EPeakParams(cut_data, "Select correcting peak params")
        print("[IELorentzianApproximator] Please select new peak parameters")
        peak_executor.execute_with_validation()

        if self.figure is not None:
            self.figure.canvas.draw_idle()
            self.figure.canvas.flush_events()

        self.correcting_params = peak_executor.get_result()
        print(
            f"[IELorentzianApproximator] Correcting params: "
            f"{self.correcting_params}"
        )

    # ------------------------------------------------------------------ #
    # Trajectory-end validation                                           #
    # ------------------------------------------------------------------ #

    def _validate_trajectory_end(self):
        print("[IELorentzianApproximator] Confirm or reject the fitted trajectory")
        self._trajectory_confirmed = None

        ax_confirm = self.figure.add_axes([0.35, 0.01, 0.1, 0.05])
        btn_confirm = Button(ax_confirm, "Confirm")

        ax_reject  = self.figure.add_axes([0.46, 0.01, 0.1, 0.05])
        btn_reject = Button(ax_reject, "Reject")

        def on_confirm(event):
            self._trajectory_confirmed = True

        def on_reject(event):
            self._trajectory_confirmed = False

        btn_confirm.on_clicked(on_confirm)
        btn_reject.on_clicked(on_reject)

        while self._trajectory_confirmed is None:
            plt.pause(0.05)

        ax_confirm.remove()
        ax_reject.remove()
        self.figure.canvas.draw_idle()

        return self._trajectory_confirmed

    # ------------------------------------------------------------------ #
    # Plot teardown                                                       #
    # ------------------------------------------------------------------ #

    def _close_execution_plot(self):
        print("[IELorentzianApproximator] Closing visualisation window")
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
        plt.ioff()

    # ------------------------------------------------------------------ #
    # Remaining abstract method stubs                                     #
    # ------------------------------------------------------------------ #

    def validate(self) -> bool:
        return True

    def prepare_for_visualization(self):
        self.data_for_visualization = self.result
