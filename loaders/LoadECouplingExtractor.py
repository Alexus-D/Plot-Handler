import os
import numpy as np
import matplotlib.pyplot as plt

from .Loader import Loader


class LoadECouplingExtractor(Loader):
	def __init__(self, params) -> None:
		super().__init__(params)
		self.data_path = self.params.get("data_path", None)
		self.result_path = self.params.get("result_path", None)
		self.plots_dir = self.params.get("plots_dir", None)
		self.plot_formats = self.params.get("plot_formats", ["png"])
		self.plot_dpi = self.params.get("plot_dpi", 150)

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
				key = key.strip()
				raw_values = raw_values.strip()
				data[key] = self._parse_list(raw_values)

		return data

	def save_data(self, data: dict):
		if self.result_path is None:
			raise ValueError("result_path is not set.")

		with open(self.result_path, "w", encoding="utf-8") as f:
			f.write("# ECouplingExtractor results\n\n")

			for key, value in data.items():
				values = self._as_list(value)
				f.write(f"{key}: {self._format_list(values)}\n")

	def plot_and_save(self, data: dict, plots_dir: str = None):
		fields = np.asarray(data.get("field", []))
		if fields.size == 0:
			raise ValueError("No 'field' data to plot.")

		plots_dir = plots_dir or self.plots_dir
		if plots_dir is None:
			if self.result_path is None:
				raise ValueError("plots_dir or result_path must be set to save plots.")
			plots_dir = os.path.dirname(self.result_path)
		os.makedirs(plots_dir, exist_ok=True)

		series = [
			("J", "J", data.get("J", [])),
			("Gamma", "Gamma_coupling", data.get("Gamma", [])),
			("gamma", "gamma_induced", data.get("gamma", [])),
			("Magnon Frequency", "magnon_freq", data.get("magnon_freq", [])),
			("Magnon Frequency (Experimental)", "magnon_freq_experimental", data.get("magnon_freq_experimental", [])),
			("Cavity Frequency", "cavity_freq", data.get("cavity_freq", [])),
			("Alpha", "alpha", data.get("alpha", [])),
			("Kappa", "kappa", data.get("kappa", [])),
			("Beta", "beta", data.get("beta", [])),
			("Plato", "plato", [data.get("plato")] * len(fields) if data.get("plato") is not None else []),
		]

		for title, filename, values in series:
			values = np.asarray(values)
			print(f"[DEBUG] Processing: title={title}, filename={filename}, size={values.size}")
			if values.size == 0:
				print(f"[DEBUG] Skipping {filename} - empty data")
				continue
			fig, ax = plt.subplots(figsize=(6, 4))
			ax.plot(fields, values, marker="o", markersize=3, linewidth=1)
			ax.set_xlabel("Field")
			ax.set_ylabel(title)
			ax.set_title(title)
			ax.grid(True, alpha=0.3)
			fig.tight_layout()

			for fmt in self.plot_formats:
				path = os.path.join(plots_dir, f"{filename}.{fmt}")
				print(f"[DEBUG] Saving: {path}")
				fig.savefig(path, dpi=self.plot_dpi)
			plt.close(fig)

	def _as_list(self, value):
		if isinstance(value, np.ndarray):
			return value.tolist()
		if isinstance(value, np.generic):
			return [value.item()]
		return list(value) if isinstance(value, (list, tuple)) else [value]

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
