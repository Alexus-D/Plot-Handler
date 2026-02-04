from .Executor import Executor
import utils.unit_transformations as ut

import numpy as np

class ESParamsRecipricator(Executor):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)

    def validate(self):
        # No validation needed for this executor
        return True

    def select_initial_params(self):
        return # No initial parameter selection needed
    
    def prepare_for_visualization(self):
        self.data_for_visualization = self.result
    
    def execute(self):
        z = self.data.get("z")
        z = ut.convert_dB_to_linear(z)
        reciprocal_z = 1 / z
        reciprocal_z = ut.convert_linear_to_dB(reciprocal_z)
        self.data.update({"z": reciprocal_z})
        self.result = self.data