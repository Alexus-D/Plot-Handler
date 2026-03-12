import os
import numpy as np
import matplotlib.pyplot as plt

from .Loader import Loader


class LoadEOwnModesBuilder(Loader):
	"""Loader for EOwnModesBuilder results - saves and plots own_modes."""
	
	def __init__(self, params) -> None:
		super().__init__(params)
		self.data_path = self.params.get("data_path", None)
		self.result_path = self.params.get("result_path", None)
		self.plots_dir = self.params.get("plots_dir", None)
		self.plot_formats = self.params.get("plot_formats", ["png"])
		self.plot_dpi = self.params.get("plot_dpi", 150)

	def load_data(self) -> dict:
		"""Load own_modes from a text file."""
		if self.data_path is None:
			raise ValueError("data_path is not set.")

		data = {"own_modes": {}}
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

				if key == "fields":
					data["own_modes"]["fields"] = [self._parse_list(raw_values)]
				elif key.startswith("mode_"):
					if "modes" not in data["own_modes"]:
						data["own_modes"]["modes"] = []
					# Parse complex numbers
					complex_values = self._parse_complex_list(raw_values)
					data["own_modes"]["modes"].append(complex_values)

		return data

	def save_data(self, data: dict):
		"""Save own_modes to a text file."""
		if self.result_path is None:
			raise ValueError("result_path is not set.")

		own_modes = data.get("own_modes", {})
		if not own_modes:
			raise ValueError("No 'own_modes' in data to save.")

		fields = own_modes.get("fields", [[]])
		if isinstance(fields, list) and len(fields) > 0:
			fields = fields[0]
		fields = np.asarray(fields)

		modes = own_modes.get("modes", [])

		with open(self.result_path, "w", encoding="utf-8") as f:
			f.write("# EOwnModesBuilder results\n")
			f.write("# Own modes: complex values as (real, imag)\n\n")

			# Save fields
			f.write(f"fields: {self._format_list(fields)}\n\n")

			# Save modes
			for idx, mode in enumerate(modes, 1):
				mode_array = np.asarray(mode)
			f.write(f"# Mode {idx}: real and imaginary parts\n")
			f.write(f"mode_{idx}: {self._format_complex_list(mode_array)}\n\n")

	def plot_and_save(self, data: dict, plots_dir: str = None):
		"""Create and save plots for own_modes."""
		own_modes = data.get("own_modes", {})
		if not own_modes:
			raise ValueError("No 'own_modes' in data to plot.")

		fields = own_modes.get("fields", [[]])
		if isinstance(fields, list) and len(fields) > 0:
			fields = fields[0]
		fields = np.asarray(fields)

		if fields.size == 0:
			raise ValueError("No field data to plot.")

		modes = own_modes.get("modes", [])
		if len(modes) == 0:
			raise ValueError("No modes data to plot.")

		plots_dir = plots_dir or self.plots_dir
		if plots_dir is None:
			if self.result_path is None:
				raise ValueError("plots_dir or result_path must be set to save plots.")
			plots_dir = os.path.dirname(self.result_path)
		os.makedirs(plots_dir, exist_ok=True)

		# Extract real and imaginary parts from complex modes
		mode_real = []
		mode_imag = []
		for mode in modes:
			mode_array = np.asarray(mode)
			mode_real.append(mode_array.real)
			mode_imag.append(mode_array.imag)

		# Plot 1: Real parts vs field
		fig, ax = plt.subplots(figsize=(8, 5))
		for idx, real_vals in enumerate(mode_real, 1):
			ax.plot(fields, real_vals, marker="o", markersize=4, linewidth=1.5, label=f"Mode {idx}")
		ax.set_xlabel("Field (Oe)")
		ax.set_ylabel("Real Part (GHz)")
		ax.set_title("Own Modes: Real Parts vs Field")
		ax.legend()
		ax.grid(True, alpha=0.3)
		fig.tight_layout()
		
		for fmt in self.plot_formats:
			path = os.path.join(plots_dir, f"own_modes_real.{fmt}")
			fig.savefig(path, dpi=self.plot_dpi)
		plt.close(fig)

		# Plot 2: Imaginary parts vs field
		fig, ax = plt.subplots(figsize=(8, 5))
		for idx, imag_vals in enumerate(mode_imag, 1):
			ax.plot(fields, imag_vals, marker="o", markersize=4, linewidth=1.5, label=f"Mode {idx}")
		ax.set_xlabel("Field (Oe)")
		ax.set_ylabel("Imaginary Part (GHz)")
		ax.set_title("Own Modes: Imaginary Parts vs Field")
		ax.legend()
		ax.grid(True, alpha=0.3)
		fig.tight_layout()
		
		for fmt in self.plot_formats:
			path = os.path.join(plots_dir, f"own_modes_imag.{fmt}")
			fig.savefig(path, dpi=self.plot_dpi)
		plt.close(fig)

		# Plot 3: Combined plot with subplots
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
		
		# Real parts
		for idx, real_vals in enumerate(mode_real, 1):
			ax1.plot(fields, real_vals, marker="o", markersize=3, linewidth=1.5, label=f"Mode {idx}")
		ax1.set_xlabel("Field (Oe)")
		ax1.set_ylabel("Real Part (GHz)")
		ax1.set_title("Real Parts")
		ax1.legend()
		ax1.grid(True, alpha=0.3)
		
		# Imaginary parts
		for idx, imag_vals in enumerate(mode_imag, 1):
			ax2.plot(fields, imag_vals, marker="o", markersize=3, linewidth=1.5, label=f"Mode {idx}")
		ax2.set_xlabel("Field (Oe)")
		ax2.set_ylabel("Imaginary Part (GHz)")
		ax2.set_title("Imaginary Parts")
		ax2.legend()
		ax2.grid(True, alpha=0.3)
		
		fig.suptitle("Own Modes: Real and Imaginary Parts", fontsize=14)
		fig.tight_layout()
		
		for fmt in self.plot_formats:
			path = os.path.join(plots_dir, f"own_modes_combined.{fmt}")
			fig.savefig(path, dpi=self.plot_dpi)
		plt.close(fig)

	def _format_list(self, values):
		"""Format a list of values as a space-separated string."""
		values_array = np.asarray(values)
		if values_array.size == 0:
			return ""
		return " ".join(str(v) for v in values_array)

	def _format_complex_list(self, values):
		"""Format a list of complex values as '(real,imag)' pairs."""
		values_array = np.asarray(values)
		if values_array.size == 0:
			return ""
		parts = []
		for v in values_array:
			if isinstance(v, complex):
				parts.append(f"({v.real},{v.imag})")
			else:
				# If not complex, treat as real
				parts.append(f"({v},0)")
		return " ".join(parts)

	def _parse_list(self, raw: str):
		"""Parse a space-separated list of numbers."""
		raw = raw.replace(",", " ").strip()
		if not raw:
			return []
		parts = [p for p in raw.split() if p]
		return [float(p) for p in parts]

	def _parse_complex_list(self, raw: str):
		"""Parse a list of complex numbers in format '(real,imag)'."""
		raw = raw.strip()
		if not raw:
			return []
		
		complex_values = []
		# Find all (real,imag) patterns
		import re
		pattern = r'\(([^,]+),([^)]+)\)'
		matches = re.findall(pattern, raw)
		
		for real_str, imag_str in matches:
			real_val = float(real_str.strip())
			imag_val = float(imag_str.strip())
			complex_values.append(complex(real_val, imag_val))
		
		return complex_values
