from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt


class PCut(Plotter):
	def __init__(self, plot_data):
		super().__init__(plot_data)

	def create_figure(self):
		x = self.plot_data.get("x")
		y = self.plot_data.get("y")

		if x is None or y is None:
			raise ValueError("plot_data must contain 'x' and 'y' keys.")

		x = np.array(x)
		y = np.array(y)

		self.figure, ax = plt.subplots()
		ax.plot(x, y)

		ax.set_title(self.plot_data.get("title", "Cut Plot"))
		ax.set_xlabel(self.plot_data.get("xlabel", "X-axis"))
		ax.set_ylabel(self.plot_data.get("ylabel", "Y-axis"))

		return self.figure
    
	def redraw(self):
		if self.figure is None:
			raise ValueError("No plot has been created to redraw.")
		plt.clf()
		self.create_figure()

	def get_axis_for_marker(self):
		if self.figure is None:
			raise ValueError("No plot has been created to get axis from.")
		return self.figure.axes[0]
