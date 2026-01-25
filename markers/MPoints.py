from .Marker import Marker


class MPoints(Marker):
    def __init__(self, axes):
        super().__init__(axes)

    def set_ticks(self, ticks):
        for tick in ticks:
            point, = self.axes.plot(tick[0], tick[1], marker='o', linestyle='None')
            self.ticks.append(point)
