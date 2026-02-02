import numpy as np

import utils.unit_transformations as ut

def Lorentzian(x: np.ndarray, x0: float, gamma: float, A: float, y0: float) -> np.ndarray:
    """
    Computes the Lorentzian function.

    Parameters:
    x (np.ndarray): Input array of x values.
    x0 (float): Center of the peak.
    gamma (float): Half-width at half-maximum (HWHM).
    A (float): Amplitude of the peak.
    y0 (float): Baseline offset.

    Returns:
    np.ndarray: Computed Lorentzian values.
    """
    return A * (gamma**2 / ((x - x0)**2 + gamma**2)) + y0


def Fano(x: np.ndarray, x0: float, gamma: float, q: float, A: float, y0: float) -> np.ndarray:
    """
    Computes the Fano resonance function.

    Parameters:
    x (np.ndarray): Input array of x values.
    x0 (float): Resonance energy (frequency).
    gamma (float): Width of the resonance.
    q (float): Fano parameter.
    A (float): Amplitude of the resonance.
    y0 (float): Baseline offset.

    Returns:
    np.ndarray: Computed Fano resonance values.
    """    
    epsilon = (x - x0) / (gamma / 2)
    fano_normalized = ((q + epsilon)**2) / (1 + epsilon**2) / (1 + q**2)
    y0_linear = ut.convert_dB_to_linear(y0)
    result_linear = A * fano_normalized + y0_linear
    return ut.convert_linear_to_dB(result_linear)