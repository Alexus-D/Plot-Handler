import numpy as np

from .Loader import Loader


class LoadEResonatorExtractor(Loader):
	def __init__(self, params) -> None:
		super().__init__(params)
		self.data_path = self.params.get("data_path", None)
		self.result_path = self.params.get("result_path", None)

	def load_data(self) -> dict:
		if self.data_path is None:
			raise ValueError("data_path is not set.")

		data = {}
		current_section = None

		with open(self.data_path, "r", encoding="utf-8") as f:
			for raw_line in f:
				line = raw_line.strip()
				if not line or line.startswith("#"):
					continue

				lower = line.lower()
				if lower.startswith("[") and lower.endswith("]"):
					current_section = lower.strip("[]")
					if current_section in ("cut", "initial_params", "fitted_params"):
						data.setdefault(current_section, {})
					elif current_section == "fit_curve":
						data.setdefault(current_section, [])
					continue

				if ":" not in line:
					continue

				key, raw_values = line.split(":", 1)
				key = key.strip()
				raw_values = raw_values.strip()

				if current_section in ("cut", "initial_params", "fitted_params"):
					data[current_section][key] = self._parse_value(raw_values)
				elif current_section == "fit_curve":
					if key == "values":
						data[current_section] = self._parse_value(raw_values)
				else:
					data[key] = self._parse_value(raw_values)

		if "fit_enabled" in data:
			data["fit_enabled"] = bool(data["fit_enabled"])

		cut = data.get("cut")
		if cut is not None:
			data["cut"] = {
				"x": cut.get("x", []),
				"y": cut.get("y", [])
			}

		return data

	def save_data(self, data: dict):
		if self.result_path is None:
			raise ValueError("result_path is not set.")

		with open(self.result_path, "w", encoding="utf-8") as f:
			f.write("# EResonatorExtractor results\n\n")

			axis = data.get("axis", "")
			cut_value = data.get("cut_value", "")
			fit_enabled = data.get("fit_enabled", False)
			resonance_freq = data.get("resonance_freq", "")
			res_magnitude = data.get("res_magnitude", "")
			cavity_width = data.get("cavity_width", "")
			plato = data.get("plato", "")

			f.write(f"axis: {axis}\n")
			f.write(f"cut_value: {cut_value}\n")
			f.write(f"fit_enabled: {int(bool(fit_enabled))}\n")
			f.write(f"resonance_freq: {resonance_freq}\n")
			f.write(f"res_magnitude: {res_magnitude}\n")
			f.write(f"cavity_width: {cavity_width}\n")
			f.write(f"plato: {plato}\n\n")

			cut = data.get("cut", {})
			f.write("[cut]\n")
			f.write(f"x: {self._format_list(self._as_list(cut.get('x', [])))}\n")
			f.write(f"y: {self._format_list(self._as_list(cut.get('y', [])))}\n\n")

			initial_params = data.get("initial_params", {})
			f.write("[initial_params]\n")
			for key, value in initial_params.items():
				f.write(f"{key}: {value}\n")
			f.write("\n")

			fitted_params = data.get("fitted_params")
			if isinstance(fitted_params, dict):
				f.write("[fitted_params]\n")
				for key, value in fitted_params.items():
					f.write(f"{key}: {value}\n")
				f.write("\n")

			fit_curve = data.get("fit_curve", [])
			f.write("[fit_curve]\n")
			f.write(f"values: {self._format_list(self._as_list(fit_curve))}\n")

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

	def _parse_value(self, raw: str):
		raw = raw.replace(",", " ").strip()
		if not raw:
			return []
		parts = [p for p in raw.split() if p]
		if len(parts) == 1:
			try:
				return float(parts[0])
			except ValueError:
				return parts[0]
		return [float(p) for p in parts]
