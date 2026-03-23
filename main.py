import os

import loaders, executors, interruptible_executors

from sugar import make_step, make_next_numbered_directory, save_with_loader


data_path = "data/mease2026.txt"
result_path = data_path.replace("data", "results").rsplit(".", 1)[0]
# result_path = os.path.join('results', os.path.basename(data_path).split('.')[0])
result_path = make_next_numbered_directory(result_path)

plot_type = "S11" if "S11" in data_path else "S21" if "S21" in data_path else "S12" if "S12" in data_path else "S22" if "S22" in data_path else "Unknown"
title = data_path.split("data/")[-1].split(".txt")[0]
data_params = {
    "data_path": data_path,
    "result_path": result_path,
    "xlabel": "Fields (Oe)",
    "ylabel": "Frequency (GHz)",
    "zlabel": f"{plot_type} (dB)",
    "title": title,
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

magnon_calib_data = make_step(data, executors.EMagnonFreqCalibrator, "Magnon Frequency Calibration")
data_params = {
    "result_path": os.path.join(result_path, "magnon_freq_calibrator_result.txt")
}
save_with_loader(loaders.LoadEMagnonFreqCalibrator, data_params, magnon_calib_data)
data.update(magnon_calib_data)


data = make_step(data, executors.EFilter, "Filter Step")

# Global 2D fit - choose approximator:
# executors.ETotalApproximator - coupled-resonator model (fits: g, kappa_m, gamma_tot)
# executors.EAnticrossingApproximator - anticrossing model (fits: J, alpha, gamma)
total_fit_result = make_step(data, executors.EAnticrossingApproximator, "Anticrossing Approximation")
data_params = {
    "result_path": os.path.join(result_path, "total_approximator_result.txt")
}
# Universal loader - works with any 2D approximator
save_with_loader(loaders.Load2DApproximator, data_params, total_fit_result)
loaders.Load2DApproximator(data_params).plot_and_save(total_fit_result, plots_dir=result_path)
data.update(total_fit_result)

# result = make_step(
#     data,
#     interruptible_executors.IEPeakWatcherPolyline,
#     "Peak Watcher",
#     save_figure_path=os.path.join(result_path, "peak_watcher_figure.png")
# )

# data_params["result_path"] = os.path.join(result_path, "ie_peak_watcher_result.txt")
# save_with_loader(loaders.LoadIEPeakWatcher, data_params, result)

# data.update(result)

# lorentz_result = make_step(
#     data,
#     interruptible_executors.IELorentzianApproximator,
#     "Lorentzian Approximator",
#     save_figure_path=os.path.join(result_path, "lorentzian_approximator_figure.png"),
# )

# data_params["result_path"] = os.path.join(result_path, "ie_lorentzian_approximator_result.txt")
# save_with_loader(loaders.LoadIELorentzianApproximator, data_params, lorentz_result)
# loaders.LoadIELorentzianApproximator(data_params).plot_and_save(
#     lorentz_result, plots_dir=result_path
# )

# data.update(lorentz_result)

# # Build own_modes from trajectories for coupling extraction
# own_modes_result = make_step(data, executors.EOwnModesBuilder, "Own Modes Builder")
# data_params = {
#     "result_path": os.path.join(result_path, "own_modes_result.txt")
# }
# save_with_loader(loaders.LoadEOwnModesBuilder, data_params, own_modes_result)
# data.update(own_modes_result)
# loaders.LoadEOwnModesBuilder(data_params).plot_and_save(own_modes_result, plots_dir=result_path)

# coupling_data = make_step(data, executors.ECouplingExtractor, "Coupling Calculator")
# data_params = {
#     "result_path": os.path.join(result_path, "coupling_extractor_result.txt")
# }
# save_with_loader(loaders.LoadECouplingExtractor, data_params, coupling_data)
# data.update(coupling_data)
# loaders.LoadECouplingExtractor(data_params).plot_and_save(coupling_data, plots_dir=result_path)

# reconstructed_params = make_step(data, executors.ESParamsReconstructor, "ES Params Reconstruction")
# data_params = {
#     "result_path": os.path.join(result_path, "es_params_reconstructor_result.txt")
# }
# save_with_loader(loaders.LoadESParamsReconstructor, data_params, reconstructed_params)
# data.update(reconstructed_params)
# loaders.LoadESParamsReconstructor(data_params).plot_and_save(reconstructed_params, plots_dir=result_path)

# # # # Аппроксимация пиков моделью Fano (или Lorentzian)
# # # approx_params = {
# # #     "model": "Fano",              # или "Lorentzian"
# # #     "fit_windows": [
# # #         (0.007, 0.025),
# # #         (0.015, 0.007)
# # #     ] # окно фита = width * multiplier
# # # }
# # # approx_result = make_step(data, interruptible_executors.IEPeakApproximatorTwoDir, "Peak Approximator", approx_params)

# # # data_params["result_path"] = os.path.join(result_path, "ie_peak_approximator_result.txt")
# # # save_with_loader(loaders.LoadIEPeakApproximator, data_params, approx_result)
# # # data.update(approx_result)


