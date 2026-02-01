import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import config_physics
import utils.algorithms as alg
from utils.phys_models import Lorentzian, Fano
from InterruptibleExecutor import InterruptibleExecutor


class IEPeakWatcher(InterruptibleExecutor):