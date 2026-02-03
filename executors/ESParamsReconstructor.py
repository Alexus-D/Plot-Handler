import numpy as np

from utils.phys_models import simple_anticrossing_model

from .Executor import Executor


def _align_series(fields: np.ndarray, base_fields: np.ndarray, values, name: str) -> np.ndarray:
	values = np.asarray(values, dtype=float)
	if values.size == 0:
		raise ValueError(f"Missing coupling parameter: {name}")
	if base_fields.size == values.size and np.array_equal(fields, base_fields):
		return values
	if base_fields.size != values.size:
		raise ValueError(
			f"Length mismatch for {name}: field={base_fields.size}, values={values.size}"
		)
	if base_fields.size < 2:
		return np.full_like(fields, values[0], dtype=float)

	sort_idx = np.argsort(base_fields)
	base_sorted = base_fields[sort_idx]
	values_sorted = values[sort_idx]

	return np.interp(fields, base_sorted, values_sorted)


class ESParamsReconstructor(Executor):
	def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
		super().__init__(data, stage_name, initial_params)

	def validate(self):
		return True

	def select_initial_params(self):
		return

	def execute(self):
		fields = np.asarray(self.data.get("x", []), dtype=float)
		freqs = np.asarray(self.data.get("y", []), dtype=float)

		if fields.size == 0 or freqs.size == 0:
			raise ValueError("Input data must contain non-empty 'x' (fields) and 'y' (freqs)")

		resonator_params = self.data.get("fitted_params") or self.data.get("initial_params") or {}
		coupling_params = self.data

		kappa = resonator_params.get("kappa", None)
		beta = resonator_params.get("beta", None)
		resonance_freq = resonator_params.get("resonance_freq", None)
		if kappa is None:
			kappa = coupling_params.get("kappa")
			if isinstance(kappa, (list, tuple, np.ndarray)):
				kappa = np.asarray(kappa, dtype=float)
				kappa = float(kappa[0]) if kappa.size else None
		if beta is None:
			beta = coupling_params.get("beta")
			if isinstance(beta, (list, tuple, np.ndarray)):
				beta = np.asarray(beta, dtype=float)
				beta = float(beta[0]) if beta.size else None
		if resonance_freq is None:
			resonance_freq = coupling_params.get("cavity_freq")
			if isinstance(resonance_freq, (list, tuple, np.ndarray)):
				resonance_freq = np.asarray(resonance_freq, dtype=float)
				resonance_freq = float(resonance_freq[0]) if resonance_freq.size else None

		missing = [
			name
			for name, value in (
				("kappa", kappa),
				("beta", beta),
				("resonance_freq", resonance_freq),
			)
			if value is None
		]
		if missing:
			raise ValueError(f"Missing resonator parameters: {', '.join(missing)}")

		coupling_fields = np.asarray(coupling_params.get("field", fields), dtype=float)
		if coupling_fields.size == 0:
			coupling_fields = fields

		J = _align_series(fields, coupling_fields, coupling_params.get("J", []), "J")
		gamma = _align_series(fields, coupling_fields, coupling_params.get("gamma", []), "gamma")
		magnon_freq = _align_series(
			fields,
			coupling_fields,
			coupling_params.get("magnon_freq", []),
			"magnon_freq",
		)
		alpha = _align_series(fields, coupling_fields, coupling_params.get("alpha", []), "alpha")

		model_params = {
			"kappa": float(kappa),
			"beta": float(beta),
			"resonance_freq": float(resonance_freq),
			"J": J,
			"gamma": gamma,
			"magnon_freq": magnon_freq,
			"alpha": alpha,
		}
		reconstructed = simple_anticrossing_model(freqs, fields, model_params)

		title = self.data.get("title", "Contour Plot")
		reconstructed_data = self.data.copy()
		reconstructed_data.update(
			{
				"x": fields,
				"y": freqs,
				"z": reconstructed,
				"title": f"{title} (reconstructed)",
				"reconstruction_params": model_params,
			}
		)

		self.result = reconstructed_data

	def prepare_for_visualization(self):
		self.data_for_visualization = self.result
