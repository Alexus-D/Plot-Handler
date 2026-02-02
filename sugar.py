from datetime import datetime
import os


def make_step(data, executor, step_name, initial_params=None):
    step_executor = executor(data, stage_name=step_name, initial_params=initial_params)
    step_executor.execute_with_validation()
    results = step_executor.get_result()
    return results


def make_time_directory(base_path: str) -> str:
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_directory = os.path.join(base_path, time_stamp)
    os.makedirs(time_directory, exist_ok=True)
    return time_directory


def make_next_numbered_directory(base_path: str) -> str:
    os.makedirs(base_path, exist_ok=True)

    max_index = 0
    for name in os.listdir(base_path):
        path = os.path.join(base_path, name)
        if not os.path.isdir(path):
            continue
        try:
            idx = int(name)
        except ValueError:
            continue
        if idx > max_index:
            max_index = idx

    next_index = max_index + 1
    next_dir = os.path.join(base_path, str(next_index))
    os.makedirs(next_dir, exist_ok=True)
    return next_dir


def save_with_loader(loader_cls, loader_params: dict, data: dict):
    loader = loader_cls(loader_params)
    loader.save_data(data)