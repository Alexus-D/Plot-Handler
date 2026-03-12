import numpy as np

import utils.algorithms as alg

from .Executor import Executor


class EOwnModesBuilder(Executor):
    """
    Builds own_modes from trajectories by synchronizing fields and creating complex modes.
    
    Input data:
        - trajectories: list of trajectory dicts with 'fields', 'freq', 'width'
    
    Output result:
        - own_modes: dict with 'fields' and 'modes' (complex values)
    """
    
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
    
    def validate(self):
        """Auto-validation: always returns True."""
        return True
    
    def select_initial_params(self):
        """No initial parameters needed."""
        pass
    
    def execute(self):
        """Build own_modes from trajectories."""
        trajectories = self.data.get("trajectories", [])
        
        if len(trajectories) < 2:
            raise ValueError(f"At least 2 trajectories required, got {len(trajectories)}")
        
        # Extract data from first two trajectories
        fields0 = np.array(trajectories[0].get("fields", []))
        fields1 = np.array(trajectories[1].get("fields", []))
        freq0 = np.array(trajectories[0].get("freq", []))
        freq1 = np.array(trajectories[1].get("freq", []))
        width0 = np.array(trajectories[0].get("width", []))
        width1 = np.array(trajectories[1].get("width", []))

        # Sort all arrays by fields0
        sort_indices = np.argsort(fields0)
        fields0 = fields0[sort_indices]
        freq0 = freq0[sort_indices]
        width0 = width0[sort_indices]

        sort_indices1 = np.argsort(fields1)
        fields1 = fields1[sort_indices1]
        freq1 = freq1[sort_indices1]
        width1 = width1[sort_indices1]
        
        # Synchronize arrays by field values
        common_fields, freq0, freq1 = alg.allighn_arrays(fields0, freq0, fields1, freq1)
        _, width0, width1 = alg.allighn_arrays(fields0, width0, fields1, width1)
        
        fields0 = np.array(common_fields)
        fields1 = np.array(common_fields)
        
        # Correct width1 sign based on damping sum consistency
        sum_damping = np.max(width0[-10:]) + np.min(width1[-10:])
        
        for i, w in enumerate(width1):
            cur_sum_damping = width0[i] + width1[i]
            cur_sub_damping = width0[i] - width1[i]
            if abs(sum_damping - cur_sum_damping) > abs(sum_damping - cur_sub_damping):
                width1[i] = -width1[i]
        
        # Find common fields with tolerance for floating point comparison
        common_fields_list = []
        mode1_sync = []
        mode2_sync = []
        
        for i, field in enumerate(fields0):
            # Find this field in trajectory 1
            idx = np.where(np.abs(fields1 - field) < 0.01)[0]
            if len(idx) > 0:
                j = idx[0]
                common_fields_list.append(field)
                mode1_sync.append(complex(freq0[i], width0[i]))
                mode2_sync.append(complex(freq1[j], width1[j]))
        
        if len(common_fields_list) == 0:
            raise ValueError("No common fields found between trajectories")
        
        # Build own_modes structure
        own_modes = {
            "fields": [common_fields_list],
            "modes": [mode1_sync, mode2_sync]
        }
        
        self.result = {"own_modes": own_modes}
    
    def prepare_for_visualization(self):
        """Prepare data for visualization."""
        self.data_for_visualization = self.result
