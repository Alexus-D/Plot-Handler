import os

import loaders, executors, interruptible_executors

from sugar import make_step, make_next_numbered_directory, save_with_loader


data_path = "data/mease2026S21.txt"
result_path = os.path.join('results', os.path.basename(data_path).split('.')[0])
result_path = make_next_numbered_directory(result_path)

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

result = make_step(data, interruptible_executors.IEPeakWatcherTwoDir, "Peak Watcher")

data_params = {
    "result_path": os.path.join(result_path, "ie_peak_watcher_result.txt")
}

save_with_loader(loaders.LoadIEPeakWatcher, data_params, result)

data.update(result)

# Аппроксимация пиков моделью Fano (или Lorentzian)
approx_params = {
    "model": "Fano",              # или "Lorentzian"
    "fit_windows": [
        (0.010, 0.025),
        (0.015, 0.010)
    ] # окно фита = width * multiplier
}
approx_result = make_step(data, interruptible_executors.IEPeakApproximatorTwoDir, "Peak Approximator", approx_params)

data_params["result_path"] = os.path.join(result_path, "ie_peak_approximator_result.txt")
save_with_loader(loaders.LoadIEPeakApproximator, data_params, approx_result)

data.update(approx_result)


