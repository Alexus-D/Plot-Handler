import numpy as np
import matplotlib.pyplot as plt

from .Executor import Executor
from ui_selectors import SModeFlipper
import utils.algorithms as alg

class EModesMaker(Executor):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
    
    def select_initial_params(self):
        return
    
    def validate(self):
        return True

    def execute(self):
        trajectories = self.data.get("trajectories")
        if trajectories is None or len(trajectories) != 2:
            raise ValueError("Two trajectories must be provided in initial_params under 'trajectories' key.")
        mode_1 = trajectories[0]
        mode_2 = trajectories[1]
        
        freqs1 = mode_1.get("freq", [])
        freqs2 = mode_2.get("freq", [])

        widths1 = mode_1.get("width", [])
        widths2 = mode_2.get("width", [])

        common_fields, freqs1, freqs2 = alg.allighn_arrays(
            mode_1.get("fields", []), freqs1,
            mode_2.get("fields", []), freqs2
        )

        _, widths1, widths2 = alg.allighn_arrays(
            mode_1.get("fields", []), widths1,
            mode_2.get("fields", []), widths2
        )

        output = {}
        mode_1 = {
            "fields": common_fields,
            'values': np.array(freqs1)  - 1j * np.array(widths1)}
        mode_2 = {
            "fields": common_fields,
            'values': np.array(freqs2)  - 1j * np.array(widths2)}
        output["modes"] = [mode_1, mode_2]

        selector = SModeFlipper(self.stage_name, output, 2)
        plt.show()
        fliping_points = selector.get_params().get("select_points", [])
        fliping_points = np.array(fliping_points)
        point_indexes = np.argsort(fliping_points[:, 0])
        low_field = fliping_points[point_indexes[0], 0]
        high_field = fliping_points[point_indexes[1], 0]
        low_width = fliping_points[point_indexes[0], 1]
        high_width = fliping_points[point_indexes[1], 1]


        fliping_indexes = np.where((common_fields <= high_field))[0]
        widths2[fliping_indexes] = 2*high_width - widths2[fliping_indexes]
        fliping_indexes = np.where((common_fields <= low_field))[0]
        widths2[fliping_indexes] = 2*low_width - widths2[fliping_indexes]


        output["modes"] = [
            {
                "fields": common_fields,
                'values': np.array(freqs1)  - 1j * np.array(widths1)
            },
            {
                "fields": common_fields,
                'values': np.array(freqs2)  - 1j * np.array(widths2)
            }
        ]

        self.result = output


    def prepare_for_visualization(self):
        self.data_for_visualization = self.result