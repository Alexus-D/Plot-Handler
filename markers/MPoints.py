from .Marker import Marker


class MPoints(Marker):
    def __init__(self, figure):
        super().__init__(figure)

    def set_ticks(self, ticks):
        for tick in ticks:
            point, = self.figure.plot(tick[0], tick[1], marker='o', linestyle='None')
            self.ticks.append(point)
