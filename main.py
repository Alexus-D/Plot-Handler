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
peak_watcher.save_figure_path = os.path.join(result_path, "peak_watcher_figure.png")
peak_watcher.execute_with_validation()
result = peak_watcher.get_result()

data_params["result_path"] = os.path.join(result_path, "ie_peak_watcher_result.txt")
save_with_loader(loaders.LoadIEPeakWatcher, data_params, result)

data.update(result)

# Prepare own_modes from trajectories for coupling extraction
trajectories = result.get("trajectories", [])
if len(trajectories) >= 2:
    # Synchronize trajectories by field values
    import numpy as np
    fields0 = np.array(trajectories[0].get("fields", []))
    fields1 = np.array(trajectories[1].get("fields", []))
    freq0 = np.array(trajectories[0].get("freq", []))
    freq1 = np.array(trajectories[1].get("freq", []))
    width0 = np.array(trajectories[0].get("width", []))
    width1 = np.array(trajectories[1].get("width", []))
    
    # Find common fields (with small tolerance for floating point comparison)
    common_fields = []
    mode1_sync = []
    mode2_sync = []
    
    for i, field in enumerate(fields0):
        # Find this field in trajectory 1
        idx = np.where(np.abs(fields1 - field) < 0.01)[0]
        if len(idx) > 0:
            j = idx[0]
            common_fields.append(field)
            mode1_sync.append(complex(freq0[i], width0[i]))
            mode2_sync.append(complex(freq1[j], width1[j]))
    
    if len(common_fields) == 0:
        raise ValueError("No common fields found between trajectories")
    
    own_modes = {
        "fields": [common_fields],
        "modes": [mode1_sync, mode2_sync]
    }
    data["own_modes"] = own_modes

coupling_data = make_step(data, executors.ECouplingExtractor, "Coupling Calculator")
data_params = {
    "result_path": os.path.join(result_path, "coupling_extractor_result.txt")
}
save_with_loader(loaders.LoadECouplingExtractor, data_params, coupling_data)
data.update(coupling_data)
loaders.LoadECouplingExtractor(data_params).plot_and_save(coupling_data, plots_dir=result_path)

reconstructed_params = make_step(data, executors.ESParamsReconstructor, "ES Params Reconstruction")
data_params = {
    "result_path": os.path.join(result_path, "es_params_reconstructor_result.txt")
}
save_with_loader(loaders.LoadESParamsReconstructor, data_params, reconstructed_params)
data.update(reconstructed_params)
loaders.LoadESParamsReconstructor(data_params).plot_and_save(reconstructed_params, plots_dir=result_path)

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


