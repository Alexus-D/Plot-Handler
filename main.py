import os
import numpy as np
import matplotlib.pyplot as plt

import loaders, executors, interruptible_executors, plotters

from sugar import make_step, make_next_numbered_directory, save_with_loader
import utils.algorithms as alg


data_path = "data/DEN#3 bulk resonator.txt"
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

resonator_data_params = {
    'data_path': r'results/DEN#3 bulk resonator/4/resonator_extractor_result.txt',
    'result_path': os.path.join(r'results/DEN#3 bulk resonator/testing', 'resonator_extractor_result.txt')
}
trajectories_data_params = {
    'data_path': r'results/DEN#3 bulk resonator/4/ie_peak_watcher_result.txt',
    'result_path': os.path.join(r'results/DEN#3 bulk resonator/testing', 'ie_peak_watcher_result.txt')
}


loader_data = loaders.LoadContourTXT(data_params)
data = loader_data.load_data()
loader_resonator = loaders.LoadEResonatorExtractor(resonator_data_params)
resonator_data = loader_resonator.load_data()

loader_trajectories = loaders.LoadIEPeakWatcher(trajectories_data_params)
trajectories_data = loader_trajectories.load_data()

fields1 = trajectories_data.get("trajectories", [])[0].get("fields", [])
freqs1 = trajectories_data.get("trajectories", [])[0].get("freq", [])
widths1 = trajectories_data.get("trajectories", [])[0].get("width", [])

fields2 = trajectories_data.get("trajectories", [])[1].get("fields", [])
freqs2 = trajectories_data.get("trajectories", [])[1].get("freq", [])
widths2 = trajectories_data.get("trajectories", [])[1].get("width", [])

common_fields, freqs1, freqs2 = alg.allighn_arrays(fields1, freqs1, fields2, freqs2)
_, width1, widths2 = alg.allighn_arrays(fields1, widths1, fields2, widths2)
fields1 = np.array(common_fields)
fields2 = np.array(common_fields)

filter_params = {
    "x_range": (np.min(common_fields), np.max(common_fields)),
    "y_range": (min(min(freqs1), min(freqs2)), max(max(freqs1), max(freqs2))),
}

data = make_step(data, executors.EFilter, "Data Filter", filter_params)

modes = make_step(trajectories_data, executors.EModesMaker, "Make Modes")
modes["file_path"] = os.path.join(r'results/DEN#3 bulk resonator/testing', 'modes.png')

plotter = plotters.PModes(modes)
plotter.save_figure()
# plt.show()

mode_dict = {
    "fields": modes["modes"][0]["fields"],
    "modes": [modes["modes"][0]["values"], modes["modes"][1]["values"]]
    }

coupling_params = {
    "resonator_params": resonator_data["initial_params"],
    "own_modes": mode_dict,
}
coupling_data = make_step(_, executors.ECouplingExtractor, "Coupling Extractor", coupling_params)

data_params = {
    "result_path": os.path.join(r'results/DEN#3 bulk resonator/testing', "coupling_extractor_result.txt")
}
save_with_loader(loaders.LoadECouplingExtractor, data_params, coupling_data)

loaders.LoadECouplingExtractor(data_params).plot_and_save(coupling_data, plots_dir=result_path)

print("Done.")



# ranges = {
#     "x_range": (20, 4000),
#     "y_range": (5, 15),
# }
# data = make_step(data, executors.EFilter, "Data Filter", {"threshold": -80})


# trajectories = trajectories_data.get("trajectories", [])



# # Prepare own_modes from trajectories for coupling extraction
# trajectories = result.get("trajectories", [])
# if len(trajectories) >= 2:
#     # Synchronize trajectories by field values
#     import numpy as np
#     fields0 = np.array(trajectories[0].get("fields", []))
#     fields1 = np.array(trajectories[1].get("fields", []))
#     freq0 = np.array(trajectories[0].get("freq", []))
#     freq1 = np.array(trajectories[1].get("freq", []))
#     width0 = np.array(trajectories[0].get("width", []))
#     width1 = np.array(trajectories[1].get("width", []))

#     common_fields, freq0, freq1 = alg.allighn_arrays(fields0, freq0, fields1, freq1)
#     _, width0, width1 = alg.allighn_arrays(fields0, width0, fields1, width1)

#     fields0 = np.array(common_fields)
#     fields1 = np.array(common_fields)

#     sum_damping = np.max(width0[-10:]) + np.min(width1[-10:])

#     for i, w in enumerate(width1):
#         cur_sum_damping = width0[i] + width1[i]
#         cur_sub_damping = width0[i] - width1[i]
#         if abs(sum_damping - cur_sum_damping) > abs(sum_damping - cur_sub_damping):
#             width1[i] = -width1[i]
    
#     # Find common fields (with small tolerance for floating point comparison)
#     common_fields = []
#     mode1_sync = []
#     mode2_sync = []
    
#     for i, field in enumerate(fields0):
#         # Find this field in trajectory 1
#         idx = np.where(np.abs(fields1 - field) < 0.01)[0]
#         if len(idx) > 0:
#             j = idx[0]
#             common_fields.append(field)
#             mode1_sync.append(complex(freq0[i], width0[i]))
#             mode2_sync.append(complex(freq1[j], width1[j]))
    
#     if len(common_fields) == 0:
#         raise ValueError("No common fields found between trajectories")
    
#     own_modes = {
#         "fields": [common_fields],
#         "modes": [mode1_sync, mode2_sync]
#     }
#     data["own_modes"] = own_modes

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

# # # Аппроксимация пиков моделью Fano (или Lorentzian)
# # approx_params = {
# #     "model": "Fano",              # или "Lorentzian"
# #     "fit_windows": [
# #         (0.007, 0.025),
# #         (0.015, 0.007)
# #     ] # окно фита = width * multiplier
# # }
# # approx_result = make_step(data, interruptible_executors.IEPeakApproximatorTwoDir, "Peak Approximator", approx_params)

# # data_params["result_path"] = os.path.join(result_path, "ie_peak_approximator_result.txt")
# # save_with_loader(loaders.LoadIEPeakApproximator, data_params, approx_result)
# # data.update(approx_result)


