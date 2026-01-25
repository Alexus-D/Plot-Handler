from abc import ABC
from abc import abstractmethod


class Marker(ABC):
    def __init__(self, figure):
        if figure is None:
            raise ValueError("figure cannot be None")
        self.figure = figure
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

    def redraw(self):
        self.figure.canvas.draw_idle()