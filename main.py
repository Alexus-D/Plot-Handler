import os

import loaders, executors, interruptible_executors

from sugar import make_step, make_time_directory


data_path = "data/DEN#3 bulk resonator.txt"
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

data = make_step(data, executors.EFilter, "Filter Step")

result = make_step(data, interruptible_executors.IEPeakWatcher, "Peak Watcher")

data.update(result)

# Аппроксимация пиков моделью Fano (или Lorentzian)
approx_params = {
    "model": "Fano",              # или "Lorentzian"
    "fit_windows": [
        # (0.10, 0.25),
        (0.015, 0.010)
    ] # окно фита = width * multiplier
}
approx_result = make_step(data, interruptible_executors.IEPeakApproximator, "Peak Approximator", approx_params)

data.update(approx_result)


