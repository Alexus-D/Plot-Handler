from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt

class PContour(Plotter):
    def __init__(self, plot_data):
        super().__init__(plot_data)

    def create_figure(self):
        x = self.plot_data.get("x")
        y = self.plot_data.get("y")
        z = self.plot_data.get("z")

        if x is None or y is None or z is None:
            raise ValueError("plot_data must contain 'x', 'y', and 'z' keys.")

        # Our data convention: z has shape (len(x), len(y))
        # matplotlib contourf expects: if using meshgrid default (xy indexing),
        # then z should have shape (len(y), len(x))
        X, Y = np.meshgrid(x, y)  # Shape: (len(y), len(x))
        Z = np.array(z).T  # Transpose to match: (len(x), len(y)) -> (len(y), len(x))

        self.figure, ax = plt.subplots()
        contour = ax.contourf(X, Y, Z, cmap='viridis', levels=25)
        cbar = plt.colorbar(contour)
        cbar.set_label(self.plot_data.get("zlabel", "Z-axis"), fontsize=10)
        ax.set_title(self.plot_data.get("title", "Contour Plot"), fontsize=11)
        ax.set_xlabel(self.plot_data.get("xlabel", "X-axis"), fontsize=10)
        ax.set_ylabel(self.plot_data.get("ylabel", "Y-axis"), fontsize=10)
        ax.tick_params(labelsize=9)

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