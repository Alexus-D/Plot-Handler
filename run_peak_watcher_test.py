import os
import traceback

import loaders
import executors
import interruptible_executors

from sugar import make_step, make_time_directory


def main():
    data_path = "data/DEN#3 bulk resonator.txt"
    result_path = os.path.join(make_time_directory("data"), "filtered_result.txt")

    data_params = {
        "data_path": data_path,
        "result_path": result_path,
        "xlabel": "Fields (Oe)",
        "ylabel": "Frequency (GHz)",
        "zlabel": "S11 (dB)",
        "title": "S11 Contour Plot",
        "unit": "dB",
    }

    loader = loaders.LoadContourTXT(data_params)
    data = loader.load_data()

    result = make_step(data, executors.EFilter, "Filter Step")
    result = make_step(result, interruptible_executors.IEPeakWatcher, "Peak Watcher")

    print("Done. Trajectories:", len(result.get("trajectories", [])))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
