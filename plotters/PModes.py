from .Plotter import Plotter
import numpy as np
import matplotlib.pyplot as plt

class PModes(Plotter):
    def __init__(self, plot_data):
        super().__init__(plot_data)

    def create_figure(self):
        modes = self.plot_data.get("modes")
        if modes is None:
            raise ValueError("plot_data must contain 'modes' key.")
        
        mode1 = modes[0]
        mode2 = modes[1]

        fields = mode1.get("fields")
        vals1 = mode1.get("values")
        vals2 = mode2.get("values")

        re_vals1 = np.real(vals1)
        im_vals1 = -np.imag(vals1)

        re_vals2 = np.real(vals2)
        im_vals2 = -np.imag(vals2)

        self.figure, ax = plt.subplots(2, 1, figsize=(8, 10))
        ax[0].plot(fields, re_vals1, label='Mode 1 - Real', color='blue')
        ax[0].plot(fields, re_vals2, label='Mode 2 - Real', color='orange')
        ax[0].set_title('Real Parts of Modes')

        ax[1].plot(fields, im_vals1, label='Mode 1 - Imaginary', color='blue')
        ax[1].plot(fields, im_vals2, label='Mode 2 - Imaginary', color='orange')
        ax[1].set_title('Imaginary Parts of Modes')

        for a in ax:
            a.set_xlabel("H(Oe)")
            a.set_ylabel("f(GHz)")
            a.legend()

        return self.figure

    def redraw(self):
        if self.figure is None:
            raise ValueError("No plot has been created to redraw.")
        plt.clf()
        self.create_figure()

    def get_axis_for_marker(self):
        if self.figure is None:
            raise ValueError("No plot has been created to get axis from.")
        return self.figure.axes[1]