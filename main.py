import os

import loaders
import interruptible_executors

from sugar import make_step, make_time_directory

data_path = "data/CoherentCoupling_S12.txt" 
result_path = os.path.join(make_time_directory("data"), "filtered_result.txt")

data_params = {
    "data_path": data_path,
    "result_path": result_path,
    "xlabel": "Fields (Oe)",
    "ylabel": "Frequency (GHz)",
    "zlabel": "S11 (dB)",
    "title": "S11 Contour Plot",
    "unit": "dB"
}

loader = loaders.LoadContourTXT(data_params)

data = loader.load_data()

result = make_step(data, interruptible_executors.ETestInterruptible, "Test Interruptible Step")