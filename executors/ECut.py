import matplotlib.pyplot as plt

from ui_selectors import SNPoints
from utils.algorithms import make_cut

from .Executor import Executor


class ECut(Executor):
	def __init__(self, data: dict, stage_name: str, axis: str = "x", initial_params: dict = None):
		super().__init__(data, stage_name, initial_params)
		if axis not in ("x", "y"):
			raise ValueError("axis must be 'x' or 'y'")
		self.axis = axis

	def validate(self):
		return True

	def execute(self):
		axis = self.axis
		cut_value = self.initial_params.get("cut_value")
		if cut_value is None:
			raise ValueError("cut_value must be set before execution")

		cut = make_cut(self.data, cut_value=cut_value, axis=axis)

		if axis == "x":
			xlabel = self.data.get("ylabel", "Y-axis")
		else:
			xlabel = self.data.get("xlabel", "X-axis")
		ylabel = self.data.get("zlabel", "Z-axis")

		title = self.data.get("title", "Contour Cut")
		title = f"{title} (cut {axis}={cut_value:.4g})"

		self.result = {
			**cut,
			"xlabel": xlabel,
			"ylabel": ylabel,
			"title": title,
			"axis": axis,
			"cut_value": cut_value,
		}

	def select_initial_params(self):
		if self.initial_params.get("cut_value") is not None:
			return
		axis = self.axis

		label = "Select X cut" if axis == "x" else "Select Y cut"
		selector = SNPoints(
			self.stage_name,
			self.data,
			num_points=1,
			buttons=[("select_points", label)],
		)
		plt.show()
		points = selector.get_params().get("select_points", [])
		if not points:
			raise ValueError("No point selected for cut")

		x_sel, y_sel = points[0]
		cut_value = x_sel if axis == "x" else y_sel
		self.initial_params = {"cut_value": cut_value}

	def prepare_for_visualization(self):
		self.data_for_visualization = self.result
