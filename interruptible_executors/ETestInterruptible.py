import time

import matplotlib.pyplot as plt

from .InterruptibleExecutor import InterruptibleExecutor
from ui_selectors import SFreq


TOTAL_PROCESSING_TIME = 2.0


class ETestInterruptible(InterruptibleExecutor):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)

    def select_initial_params(self):
        print("Select initial height: click on graph after pressing 'Select Frequency', then close window")
        selector = SFreq(self.stage_name, self.data)
        plt.show()
        params = selector.get_params()
        height = params.get('select_freq', 0)
        print(f"Selected height: {height}")
        return {'height': height}

    def select_correcting_params(self):
        print("Select new height for remaining points")
        plt.ioff()
        selector = SFreq(f"{self.stage_name} - Correction", self.data)
        plt.show()
        params = selector.get_params()
        height = params.get('select_freq', 0)
        print(f"Selected new height: {height}")
        return {'height': height}

    def interruptible_function(self, remain_data, params, current_result):
        height = params.get('height', 0)
        x_values = sorted(remain_data.get('x', []))
        result = []
        processed_x = []

        fig = self.plotter.get_figure()
        total = len(x_values)
        target_time = TOTAL_PROCESSING_TIME * (len(x_values) / len(self.data.get('x', [1])))
        print(f"Processing {total} points at height={height:.2f} (target: {target_time:.1f}s)")

        start_time = time.time()
        for i, x in enumerate(x_values):
            if self.interrupted:
                break

            result.append((x, height))
            processed_x.append(x)
            
            self._update_line(current_result + result)
            fig.canvas.flush_events()
            
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{total}")
            
            elapsed = time.time() - start_time
            expected = target_time * (i + 1) / total
            if elapsed < expected:
                time.sleep(expected - elapsed)

        for x in processed_x:
            if x in remain_data['x']:
                remain_data['x'].remove(x)

        actual_time = time.time() - start_time
        print(f"Done: {len(processed_x)} points in {actual_time:.1f}s" + (" (interrupted)" if self.interrupted else ""))
        return result
