import numpy as np

from .Loader import Loader


class LoadContourTXT(Loader):
    def __init__(self, params) -> None:
        super().__init__(params)
        self.data_path = self.params.get("data_path", None)
        self.result_path = self.params.get("result_path", None)
        self.last_line_err = self.params.get("last_line_err", True)

        self.xlabel = self.params.get("xlabel", "X-axis")
        self.ylabel = self.params.get("ylabel", "Y-axis")
        self.zlabel = self.params.get("zlabel", "Z-axis")
        self.title = self.params.get("title", "Contour Plot")

    def load_data(self) -> dict:
        if self.data_path is None:
            raise ValueError("data_path is not set.")
        data = np.loadtxt(self.data_path, delimiter='\t')
        y = data[0, 1:]
        x = data[1:, 0]
        z = data[1:, 1:]

        if self.last_line_err:
            z = z[:-1, :]
            x = x[:-1]

        return {
            "x": x,
            "y": y,
            "z": z,
            "xlabel": self.xlabel,
            "ylabel": self.ylabel,
            "zlabel": self.zlabel,
            "title": self.title
        }

    def save_data(self, data: dict):
        if self.result_path is None:
            raise ValueError("result_path is not set.")
        x = data.get("x", None)
        y = data.get("y", None)
        z = data.get("z", None)

        if x is None or y is None or z is None:
            raise ValueError("Data must contain 'x', 'y', and 'z' keys.")

        z = np.array(z)

        result = np.zeros((len(x) + 1, len(y) + 1))
        result[0, 1:] = y
        result[1:, 0] = x
        result[1:, 1:] = z

        np.savetxt(self.result_path, result, delimiter=',')