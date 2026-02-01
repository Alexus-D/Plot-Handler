import matplotlib.pyplot as plt

from ui_selectors import SPeakParams

from .Executor import Executor


class EPeakParams(Executor):
	def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
		super().__init__(data, stage_name, initial_params)

	def validate(self):
		return True

	def execute(self):
		peak_freq = self.initial_params.get("peak_freq")
		peak_value = self.initial_params.get("peak_value")
		plateau = self.initial_params.get("plateau")
		peak_width = self.initial_params.get("peak_width")
		peak_type = self.initial_params.get("peak_type")

		missing = [
			name
			for name, value in (
				("peak_freq", peak_freq),
				("peak_value", peak_value),
				("plateau", plateau),
				("peak_width", peak_width),
				("peak_type", peak_type),
			)
			if value is None
		]
		if missing:
			raise ValueError(f"Missing peak parameters: {', '.join(missing)}")

		prominence = abs(peak_value - plateau)

		self.result = {
			"peak_freq": peak_freq,
			"peak_value": peak_value,
			"plateau": plateau,
			"prominence": prominence,
			"peak_width": peak_width,
			"peak_type": peak_type,
		}

	def select_initial_params(self):
		required = {"peak_freq", "peak_value", "plateau", "peak_width", "peak_type"}
		if required.issubset(self.initial_params.keys()):
			return

		selector = SPeakParams(self.stage_name, self.data)
		plt.show(block=False)
		while not selector.finished:
			plt.pause(0.1)
		params = selector.get_params()
		self.initial_params = dict(params)

	def prepare_for_visualization(self):
		self.data_for_visualization = self.result
