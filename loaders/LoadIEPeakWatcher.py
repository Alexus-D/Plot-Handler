import numpy as np

from .Loader import Loader


class LoadIEPeakWatcher(Loader):
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

				if key == "method":
					current[key] = self._parse_list(raw_values, as_float=False)
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
			f.write("# IEPeakWatcher results\n")
			f.write(f"# trajectories: {len(trajectories)}\n\n")

			for idx, traj in enumerate(trajectories, start=1):
				f.write(f"[trajectory {idx}]\n")
				fields = self._as_list(traj.get("fields", []))
				freq = self._as_list(traj.get("freq", []))
				magnitude = self._as_list(traj.get("magnitude", []))
				prominence = self._as_list(traj.get("prominence", []))
				width = self._as_list(traj.get("width", []))
				method = self._as_list(traj.get("method", []))

				f.write(f"fields: {self._format_list(fields)}\n")
				f.write(f"freq: {self._format_list(freq)}\n")
				f.write(f"magnitude: {self._format_list(magnitude)}\n")
				f.write(f"prominence: {self._format_list(prominence)}\n")
				f.write(f"width: {self._format_list(width)}\n")
				f.write(f"method: {self._format_list(method, as_float=False)}\n\n")

	def _as_list(self, value):
		if isinstance(value, np.ndarray):
			return value.tolist()
		if isinstance(value, np.generic):
			return [value.item()]
		return list(value) if isinstance(value, (list, tuple)) else []

	def _format_list(self, values, as_float: bool = True):
		if not values:
			return ""
		if as_float:
			return " ".join(str(v) for v in values)
		return " ".join(str(v) for v in values)

	def _parse_list(self, raw: str, as_float: bool = True):
		raw = raw.replace(",", " ").strip()
		if not raw:
			return []
		parts = [p for p in raw.split() if p]
		if as_float:
			return [float(p) for p in parts]
		return parts
