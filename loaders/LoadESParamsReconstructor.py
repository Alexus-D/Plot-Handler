import os
import numpy as np
import matplotlib.pyplot as plt

from .Loader import Loader


class LoadESParamsReconstructor(Loader):
	def __init__(self, params) -> None:
		super().__init__(params)
		self.data_path = self.params.get("data_path", None)
		self.result_path = self.params.get("result_path", None)
		self.plots_dir = self.params.get("plots_dir", None)
		self.plot_formats = self.params.get("plot_formats", ["png"])
		self.plot_dpi = self.params.get("plot_dpi", 150)

		self.xlabel = self.params.get("xlabel", "X-axis")
		self.ylabel = self.params.get("ylabel", "Y-axis")
		self.zlabel = self.params.get("zlabel", "Z-axis")
		self.title = self.params.get("title", "Contour Plot")

	def load_data(self) -> dict:
		if self.data_path is None:
			raise ValueError("data_path is not set.")

		data = np.loadtxt(self.data_path, delimiter=",", comments="#")
		y = data[0, 1:]
		x = data[1:, 0]
		z = data[1:, 1:]

		return {
			"x": x,
			"y": y,
			"z": z,
			"xlabel": self.xlabel,
			"ylabel": self.ylabel,
			"zlabel": self.zlabel,
			"title": self.title,
		}

	def save_data(self, data: dict):
		if self.result_path is None:
			raise ValueError("result_path is not set.")

		x = data.get("x", None)
		y = data.get("y", None)
		z = data.get("z", None)

		if x is None or y is None or z is None:
			raise ValueError("Data must contain 'x', 'y', and 'z' keys.")

		x = np.asarray(x)
		y = np.asarray(y)
		z = np.asarray(z)

		result = np.zeros((len(x) + 1, len(y) + 1))
		result[0, 1:] = y
		result[1:, 0] = x
		result[1:, 1:] = z

		reconstruction_params = data.get("reconstruction_params", {})

		with open(self.result_path, "w", encoding="utf-8") as f:
			f.write("# ESParamsReconstructor results\n")
			if reconstruction_params:
				for key, value in reconstruction_params.items():
					f.write(f"# {key}: {self._format_value(value)}\n")
			f.write("\n")
			np.savetxt(f, result, delimiter=",")

		self.plot_and_save(data)

	def plot_and_save(self, data: dict, plots_dir: str = None):
		plots_dir = plots_dir or self.plots_dir
		if plots_dir is None:
			if self.result_path is None:
				raise ValueError("plots_dir or result_path must be set to save plots.")
			plots_dir = os.path.dirname(self.result_path)
		os.makedirs(plots_dir, exist_ok=True)

		x = np.asarray(data.get("x", []))
		y = np.asarray(data.get("y", []))
		z = np.asarray(data.get("z", []))

		if x.size == 0 or y.size == 0 or z.size == 0:
			raise ValueError("Data must contain non-empty 'x', 'y', and 'z' for plotting.")

		X, Y = np.meshgrid(x, y)
		Z = np.array(z).transpose()

		title = data.get("title", self.title)
		xlabel = data.get("xlabel", self.xlabel)
		ylabel = data.get("ylabel", self.ylabel)
		zlabel = data.get("zlabel", self.zlabel)

		fig, ax = plt.subplots(figsize=(6, 4))
		contour = ax.contourf(X, Y, Z, cmap="viridis", levels=25)
		cbar = plt.colorbar(contour)
		cbar.set_label(zlabel)
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		fig.tight_layout()

		for fmt in self.plot_formats:
			path = os.path.join(plots_dir, f"es_params_reconstructor.{fmt}")
			fig.savefig(path, dpi=self.plot_dpi)
		plt.close(fig)

	def _format_value(self, value):
		if isinstance(value, np.ndarray):
			value = value.tolist()
		if isinstance(value, (list, tuple)):
			return " ".join(str(v) for v in value)
		return str(value)
