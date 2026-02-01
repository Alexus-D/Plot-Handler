import matplotlib.pyplot as plt

from ui_selectors import SPeakTrajectories

from .Executor import Executor


class ETrajectories(Executor):
	def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
		super().__init__(data, stage_name, initial_params)

	def validate(self):
		return True

	def execute(self):
		trajectories = self.initial_params.get("trajectories")
		if trajectories is None:
			raise ValueError("trajectories must be set before execution")
		self.result = {"trajectories": trajectories}

	def select_initial_params(self):
		if self.initial_params.get("trajectories") is not None:
			return

		selector = SPeakTrajectories(self.stage_name, self.data)
		plt.show()
		params = selector.get_params()
		self.initial_params = {"trajectories": params.get("trajectories", [])}

	def prepare_for_visualization(self):
		self.data_for_visualization = self.result
