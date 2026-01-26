from datetime import datetime
import os

def make_step(data, executor, step_name, initial_params=None):
    step_executor = executor(data, stage_name=step_name, initial_params=initial_params)
    step_executor.execute_with_validation()
    return step_executor.get_result()

def make_time_directory(base_path: str) -> str:
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    time_directory = os.path.join(base_path, time_stamp)
    os.makedirs(time_directory, exist_ok=True)
    return time_directory