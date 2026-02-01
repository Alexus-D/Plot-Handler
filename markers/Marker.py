from abc import ABC
from abc import abstractmethod


class Marker(ABC):
    def __init__(self, axes):
        if axes is None:
            raise ValueError("axes cannot be None")
        self.axes = axes
        self.ticks = []
    
    def delete_ticks(self, indices=None):
        if indices is None:
            indices = list(range(len(self.ticks)))
        elif not isinstance(indices, (list, tuple, set)):
            raise ValueError("indices must be a list/tuple/set of integers or None")

        unique_indices = sorted(set(indices), reverse=True)

        for idx in unique_indices:
            if not isinstance(idx, int):
                raise ValueError("indices must contain only integers")
            if idx < 0 or idx >= len(self.ticks):
                continue

            tick = self.ticks[idx]
            if tick is not None and hasattr(tick, "remove"):
                try:
                    tick.remove()
                except Exception:
                    pass
            self.ticks.pop(idx)
        
    @abstractmethod
    def set_ticks(self, ticks):
        pass

    def update_ticks(self, ticks, clear: bool = True):
        if clear:
            self.delete_ticks()
        self.set_ticks(ticks)
        self.redraw()

    def redraw(self):
        if hasattr(self.axes, "figure") and hasattr(self.axes.figure, "canvas"):
            self.axes.figure.canvas.draw_idle()
            return
        raise ValueError("axes does not have a figure canvas to redraw")