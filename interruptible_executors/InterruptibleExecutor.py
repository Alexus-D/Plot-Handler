from abc import abstractmethod

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector

from executors import Executor
from plotters import PContour
from validators import VContourf


class InterruptibleExecutor(Executor):
    def __init__(self, data: dict, stage_name: str, initial_params: dict = None):
        super().__init__(data, stage_name, initial_params)
        self.remain_data = {}
        self.interrupted = False
        
        self.plotter = None
        self.line = None

    def execute(self, initial_params):
        self.remain_data = self._init_remain_data()
        params = initial_params
        result = []

        while len(self.remain_data.get('x', [])) > 0:
            self.interrupted = False
            self._setup_execution_plot(result)
            
            partial_result = self.interruptible_function(self.remain_data, params, result)
            result.extend(partial_result)

            if self.interrupted:
                result, deleted = self.delete_wrong_results(result)
                
                deleted_x = [p[0] for p in deleted]
                current_remain_x = list(self.remain_data.get('x', []))
                current_remain_x.extend(deleted_x)
                current_remain_x.sort()
                self.remain_data['x'] = current_remain_x
                
                if len(self.remain_data.get('x', [])) > 0:
                    self._close_execution_plot()
                    params = self.select_correcting_params()
            else:
                break

        self._close_execution_plot()
        return result

    def _close_execution_plot(self):
        if self.plotter is not None:
            plt.close(self.plotter.get_figure())
            self.plotter = None
            self.line = None
            plt.ioff()

    def _init_remain_data(self):
        return {'x': list(self.data.get('x', []))}

    def _setup_execution_plot(self, result):
        if self.plotter is None:
            self.plotter = PContour(self.data)
            ax = self.plotter.get_axis_for_marker()
            self.line, = ax.plot([], [], 'r-', linewidth=2)
            self._create_interrupt_button()
            plt.show(block=False)
            plt.pause(0.1)
        plt.ion()
        self._update_line(result)

    def _update_line(self, points):
        if points:
            sorted_points = sorted(points, key=lambda p: p[0])
            xs = []
            ys = []
            all_x = sorted(self.data.get('x', []))
            x_set = {p[0] for p in sorted_points}
            
            for i, p in enumerate(sorted_points):
                xs.append(p[0])
                ys.append(p[1])
                
                if i < len(sorted_points) - 1:
                    next_p = sorted_points[i + 1]
                    curr_idx = all_x.index(p[0]) if p[0] in all_x else -1
                    next_idx = all_x.index(next_p[0]) if next_p[0] in all_x else -1
                    if curr_idx >= 0 and next_idx >= 0 and next_idx - curr_idx > 1:
                        xs.append(float('nan'))
                        ys.append(float('nan'))
            
            self.line.set_data(xs, ys)
        else:
            self.line.set_data([], [])
        self.plotter.get_figure().canvas.draw_idle()

    def _create_interrupt_button(self):
        fig = self.plotter.get_figure()
        ax_btn = fig.add_axes([0.02, 0.02, 0.12, 0.04])
        btn = Button(ax_btn, 'Interrupt')
        btn.on_clicked(lambda e: setattr(self, 'interrupted', True))
        self._interrupt_btn = btn

    def delete_wrong_results(self, result):
        selected_points = []
        all_deleted = []
        confirmed = [False]
        
        print(f"Delete mode: {len(result)} points. Select area and press Delete, then Continue.")
        self._update_line(result)
        
        fig = self.plotter.get_figure()
        ax = self.plotter.get_axis_for_marker()
        
        def on_rect_select(eclick, erelease):
            x1, x2 = sorted([eclick.xdata, erelease.xdata])
            y1, y2 = sorted([eclick.ydata, erelease.ydata])
            selected_points.clear()
            selected_points.extend([
                p for p in result
                if x1 <= p[0] <= x2 and y1 <= p[1] <= y2
            ])
            print(f"  Selected {len(selected_points)} points")
        
        def on_delete(event):
            nonlocal result
            if not selected_points:
                print("  Nothing selected")
                return
            all_deleted.extend(selected_points)
            result = [p for p in result if p not in selected_points]
            print(f"  Deleted {len(selected_points)} points, {len(result)} remaining")
            self._update_line(result)
            selected_points.clear()
        
        def on_continue(event):
            confirmed[0] = True
        
        rect_selector = RectangleSelector(ax, on_rect_select, useblit=True, button=[1], interactive=True)
        
        ax_delete = fig.add_axes([0.02, 0.08, 0.12, 0.04])
        delete_btn = Button(ax_delete, 'Delete')
        delete_btn.on_clicked(on_delete)
        
        ax_continue = fig.add_axes([0.16, 0.08, 0.12, 0.04])
        continue_btn = Button(ax_continue, 'Continue')
        continue_btn.on_clicked(on_continue)
        
        fig.canvas.draw_idle()
        
        while not confirmed[0]:
            plt.pause(0.1)
        
        ax_delete.remove()
        ax_continue.remove()
        rect_selector.set_active(False)
        fig.canvas.draw_idle()
        
        print(f"Continuing with {len(result)} points, {len(all_deleted)} deleted")
        return result, all_deleted

    def validate(self, data_for_visualization):
        print("Validation: Accept or Deny the result")
        validator = VContourf(self.stage_name, data_for_visualization)
        
        ax = validator.plotter.get_axis_for_marker()
        points = data_for_visualization.get('points', [])
        if points:
            sorted_points = sorted(points, key=lambda p: p[0])
            all_x = sorted(self.data.get('x', []))
            xs = []
            ys = []
            
            for i, p in enumerate(sorted_points):
                xs.append(p[0])
                ys.append(p[1])
                
                if i < len(sorted_points) - 1:
                    next_p = sorted_points[i + 1]
                    curr_idx = all_x.index(p[0]) if p[0] in all_x else -1
                    next_idx = all_x.index(next_p[0]) if next_p[0] in all_x else -1
                    if curr_idx >= 0 and next_idx >= 0 and next_idx - curr_idx > 1:
                        xs.append(float('nan'))
                        ys.append(float('nan'))
            
            ax.plot(xs, ys, 'r-', linewidth=2)
        
        plt.show()
        result = validator.get_params().get("Validation")
        print(f"Validation result: {'Accepted' if result else 'Denied (will restart)'}")
        return result

    def prepare_for_visualization(self, result):
        return {
            'points': result,
            'x': self.data.get('x'),
            'y': self.data.get('y'),
            'z': self.data.get('z'),
            'xlabel': self.data.get('xlabel'),
            'ylabel': self.data.get('ylabel'),
            'zlabel': self.data.get('zlabel'),
            'title': self.data.get('title')
        }

    @abstractmethod
    def interruptible_function(self, remain_data, params, current_result):
        pass

    @abstractmethod
    def select_correcting_params(self):
        pass
