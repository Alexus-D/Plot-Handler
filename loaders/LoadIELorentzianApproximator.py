import os
import numpy as np
import matplotlib.pyplot as plt

from .Loader import Loader


class LoadIELorentzianApproximator(Loader):
    """
    Loader / saver for :class:`IELorentzianApproximator` results.

    Result dict format::

        {
            "trajectories": [
                {
                    "model":  "Lorentzian",
                    "fields": [...],   # field values (Oe)
                    "x0":     [...],   # fitted centre frequency (GHz)
                    "gamma":  [...],   # HWHM (GHz)
                    "A":      [...],   # amplitude (linear)
                    "y0":     [...],   # baseline (linear)
                },
                ...
            ]
        }

    Text file format (one ``[trajectory N]`` block per trajectory)::

        # IELorentzianApproximator results
        # trajectories: N

        [trajectory 1]
        model: Lorentzian
        fields: ...
        x0: ...
        gamma: ...
        A: ...
        y0: ...

    ``plot_and_save`` produces, for every trajectory, individual PNG files for
    x0, gamma, A, and y0 versus field.
    """

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.data_path   = self.params.get("data_path",   None)
        self.result_path = self.params.get("result_path", None)
        self.plots_dir   = self.params.get("plots_dir",   None)
        self.plot_formats = self.params.get("plot_formats", ["png"])
        self.plot_dpi     = self.params.get("plot_dpi",     150)

    # ------------------------------------------------------------------ #
    # Load                                                                #
    # ------------------------------------------------------------------ #

    def load_data(self) -> dict:
        if self.data_path is None:
            raise ValueError("data_path is not set.")

        trajectories = []
        current = {}

        with open(self.data_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                lower = line.lower()
                if lower.startswith("trajectory") or lower.startswith("[trajectory"):
                    if current:
                        trajectories.append(current)
                        current = {}
                    continue

                if ":" not in line:
                    continue

                key, raw_values = line.split(":", 1)
                key        = key.strip()
                raw_values = raw_values.strip()

                if key == "model":
                    current[key] = raw_values
                else:
                    current[key] = self._parse_list(raw_values)

        if current:
            trajectories.append(current)

        return {"trajectories": trajectories}

    # ------------------------------------------------------------------ #
    # Save raw data                                                       #
    # ------------------------------------------------------------------ #

    def save_data(self, data: dict):
        if self.result_path is None:
            raise ValueError("result_path is not set.")

        if not isinstance(data, dict) or "trajectories" not in data:
            raise ValueError("Data must contain 'trajectories' key.")

        trajectories = data["trajectories"]

        with open(self.result_path, "w", encoding="utf-8") as f:
            f.write("# IELorentzianApproximator results\n")
            f.write(f"# trajectories: {len(trajectories)}\n\n")

            for idx, traj in enumerate(trajectories, start=1):
                f.write(f"[trajectory {idx}]\n")
                f.write(f"model: {traj.get('model', 'Lorentzian')}\n")
                f.write(f"fields: {self._format_list(self._as_list(traj.get('fields', [])))}\n")
                f.write(f"x0:     {self._format_list(self._as_list(traj.get('x0',     [])))}\n")
                f.write(f"gamma:  {self._format_list(self._as_list(traj.get('gamma',  [])))}\n")
                f.write(f"A:      {self._format_list(self._as_list(traj.get('A',      [])))}\n")
                f.write(f"y0:     {self._format_list(self._as_list(traj.get('y0',     [])))}\n\n")

    # ------------------------------------------------------------------ #
    # Plot and save figures                                               #
    # ------------------------------------------------------------------ #

    def plot_and_save(self, data: dict, plots_dir: str = None):
        """
        Saves per-trajectory plots of x0, gamma, A, y0 vs. field.

        One PNG (or other format) file is written per quantity per trajectory,
        named ``lorentz_traj<N>_<quantity>.<fmt>``.
        """
        trajectories = data.get("trajectories", [])
        if not trajectories:
            print("[LoadIELorentzianApproximator] No trajectories to plot.")
            return

        plots_dir = plots_dir or self.plots_dir
        if plots_dir is None:
            if self.result_path is None:
                raise ValueError("plots_dir or result_path must be set to save plots.")
            plots_dir = os.path.dirname(self.result_path)
        os.makedirs(plots_dir, exist_ok=True)

        series_defs = [
            ("x0",    "Centre frequency x0 (GHz)",  "Frequency (GHz)"),
            ("gamma", "HWHM  gamma (GHz)",           "gamma (GHz)"),
            ("A",     "Amplitude A (linear)",        "A (linear)"),
            ("y0",    "Baseline y0 (linear)",        "y0 (linear)"),
        ]

        for traj_idx, traj in enumerate(trajectories, start=1):
            fields = np.asarray(traj.get("fields", []))
            if fields.size == 0:
                continue

            for key, title, ylabel in series_defs:
                values = np.asarray(traj.get(key, []))
                if values.size == 0:
                    continue

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(fields, values, "o-", markersize=4, linewidth=1)
                ax.set_xlabel("Field (Oe)")
                ax.set_ylabel(ylabel)
                ax.set_title(f"Trajectory {traj_idx} — {title}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()

                for fmt in self.plot_formats:
                    path = os.path.join(
                        plots_dir,
                        f"lorentz_traj{traj_idx}_{key}.{fmt}",
                    )
                    fig.savefig(path, dpi=self.plot_dpi)
                plt.close(fig)

        print(
            f"[LoadIELorentzianApproximator] Saved plots to {plots_dir}"
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _as_list(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return [value.item()]
        return list(value) if isinstance(value, (list, tuple)) else []

    def _format_list(self, values):
        if not values:
            return ""
        return " ".join(str(v) for v in values)

    def _parse_list(self, raw: str):
        raw = raw.replace(",", " ").strip()
        if not raw:
            return []
        parts = [p for p in raw.split() if p]
        return [float(p) for p in parts]
