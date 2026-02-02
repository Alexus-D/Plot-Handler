import numpy as np

from .Executor import Executor


def extract_coupling_params(resonator_params: dict, own_modes: dict) -> dict:
	fields = own_modes.get("fields")
	if isinstance(fields, (list, tuple)):
		fields = fields[0] if fields else []
	fields = np.asarray(fields)

	modes = own_modes.get("modes", [])
	if len(modes) < 2:
		raise ValueError("own_modes must contain at least two modes")
	mode_1 = np.asarray(modes[0])
	mode_2 = np.asarray(modes[1])

	kappa = resonator_params["kappa"]
	beta = resonator_params["beta"]
	cavity_freq = resonator_params["resonance_freq"]
	plato = resonator_params["plato"]

	# cavity_complex = cavity_freq - 1j * (beta + kappa)
	cavity_complex = cavity_freq - 1j * beta

	J, Gamma, gamma, magnon_freq, alpha = [], [], [], [], []
	magnon_freq_experimental = []

	for i, field in enumerate(fields):
		mode_1_val = mode_1[i]
		mode_2_val = mode_2[i]
		magnon_complex = mode_1_val + mode_2_val - cavity_complex

		magnon_freq_experimental.append(magnon_complex.real)

		delta = cavity_complex - magnon_complex
		coupling = np.sqrt((mode_1_val - mode_2_val) ** 2 - delta ** 2) / 2

		J_val = coupling.real
		Gamma_val = -coupling.imag
		gamma_val = Gamma_val ** 2 / kappa
		alpha_val = -magnon_complex.imag

		J.append(np.abs(J_val))
		Gamma.append(np.abs(Gamma_val))
		gamma.append(np.abs(gamma_val))
		magnon_freq.append(np.abs(magnon_complex.real))
		alpha.append(np.abs(alpha_val))

	params = {
		"field": fields,
		"J": J,
		"Gamma": Gamma,
		"gamma": gamma,
		"magnon_freq": magnon_freq,
		"magnon_freq_experimental": magnon_freq_experimental,
		"cavity_freq": cavity_freq * np.ones_like(fields),
		"alpha": alpha,
		"kappa": kappa * np.ones_like(fields),
		"beta": beta * np.ones_like(fields),
		"plato": plato,
	}

	return params


class ECouplingExtractor(Executor):
	def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
		super().__init__(data, stage_name, initial_params)

	def validate(self):
		return True

	def select_initial_params(self):
		return

	def execute(self):
		resonator_params = (
			self.initial_params.get("resonator_params")
			or self.data.get("resonator_params")
			or self.data.get("fitted_params")
			or self.data.get("initial_params")
		)
		own_modes = self.initial_params.get("own_modes") or self.data.get("own_modes")

		missing = [
			name
			for name, value in (
				("resonator_params", resonator_params),
				("own_modes", own_modes),
			)
			if value is None
		]
		if missing:
			raise ValueError(f"Missing parameters: {', '.join(missing)}")

		self.result = extract_coupling_params(resonator_params, own_modes)

	def prepare_for_visualization(self):
		self.data_for_visualization = self.result