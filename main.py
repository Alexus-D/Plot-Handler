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

resonator_data = make_step(data, executors.EResonatorExtractor, "Resonator Data Extraction")
data_params = {
    "result_path": os.path.join(result_path, "resonator_extractor_result.txt")
}
save_with_loader(loaders.LoadEResonatorExtractor, data_params, resonator_data)
data.update(resonator_data)


data = make_step(data, executors.EFilter, "Filter Step")

peak_watcher = interruptible_executors.IEPeakWatcherTwoDir(data, "Peak Watcher")
peak_watcher.execute_with_validation()
result = peak_watcher.get_result()

# Save figure before closing
peak_watcher.save_figure(os.path.join(result_path, "peak_watcher_figure.png"))

data_params["result_path"] = os.path.join(result_path, "ie_peak_watcher_result.txt")
save_with_loader(loaders.LoadIEPeakWatcher, data_params, result)

data.update(result)

# Prepare own_modes from trajectories for coupling extraction
trajectories = result.get("trajectories", [])
if len(trajectories) >= 2:
    own_modes = {
        "fields": [trajectories[0].get("fields", [])],
        "modes": [
            [complex(f, m) for f, m in zip(trajectories[0].get("freq", []), trajectories[0].get("magnitude", []))],
            [complex(f, m) for f, m in zip(trajectories[1].get("freq", []), trajectories[1].get("magnitude", []))]
        ]
    }
    data["own_modes"] = own_modes

coupling_data = make_step(data, executors.ECouplingExtractor, "Coupling Calculator")
data_params = {
    "result_path": os.path.join(result_path, "coupling_extractor_result.txt")
}
save_with_loader(loaders.LoadECouplingExtractor, data_params, coupling_data)
data.update(coupling_data)
loaders.LoadECouplingExtractor(data_params).plot_and_save(coupling_data, plots_dir=result_path)

# # Аппроксимация пиков моделью Fano (или Lorentzian)
# approx_params = {
#     "model": "Fano",              # или "Lorentzian"
#     "fit_windows": [
#         (0.007, 0.025),
#         (0.015, 0.007)
#     ] # окно фита = width * multiplier
# }
# approx_result = make_step(data, interruptible_executors.IEPeakApproximatorTwoDir, "Peak Approximator", approx_params)

# data_params["result_path"] = os.path.join(result_path, "ie_peak_approximator_result.txt")
# save_with_loader(loaders.LoadIEPeakApproximator, data_params, approx_result)
# data.update(approx_result)


