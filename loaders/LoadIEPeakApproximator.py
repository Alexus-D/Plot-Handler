import numpy as np

from .Loader import Loader


class LoadIEPeakApproximator(Loader):
	def __init__(self, params) -> None:
		super().__init__(params)
		self.data_path = self.params.get("data_path", None)
		self.result_path = self.params.get("result_path", None)

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
				key = key.strip()
				raw_values = raw_values.strip()

				if key == "model":
					current[key] = raw_values
				else:
					current[key] = self._parse_list(raw_values, as_float=True)

		if current:
			trajectories.append(current)

		return {"trajectories": trajectories}

	def save_data(self, data: dict):
		if self.result_path is None:
			raise ValueError("result_path is not set.")

		if not isinstance(data, dict) or "trajectories" not in data:
			raise ValueError("Data must contain 'trajectories' key.")

		trajectories = data.get("trajectories", [])

		with open(self.result_path, "w", encoding="utf-8") as f:
			f.write("# IEPeakApproximator results\n")
			f.write(f"# trajectories: {len(trajectories)}\n\n")

			for idx, traj in enumerate(trajectories, start=1):
				f.write(f"[trajectory {idx}]\n")
				model = traj.get("model", "")
				fields = self._as_list(traj.get("fields", []))
				x0 = self._as_list(traj.get("x0", []))
				gamma = self._as_list(traj.get("gamma", []))
				A = self._as_list(traj.get("A", []))
				y0 = self._as_list(traj.get("y0", []))
				q = self._as_list(traj.get("q", []))

				f.write(f"model: {model}\n")
				f.write(f"fields: {self._format_list(fields)}\n")
				f.write(f"x0: {self._format_list(x0)}\n")
				f.write(f"gamma: {self._format_list(gamma)}\n")
				f.write(f"A: {self._format_list(A)}\n")
				f.write(f"y0: {self._format_list(y0)}\n")
				f.write(f"q: {self._format_list(q)}\n\n")

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

	def _parse_list(self, raw: str, as_float: bool = True):
		raw = raw.replace(",", " ").strip()
		if not raw:
			return []
		parts = [p for p in raw.split() if p]
		if as_float:
			return [float(p) for p in parts]
		return parts
