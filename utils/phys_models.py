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


def simple_anticrossing_model(freqs, fields, params):
    """
    Simple anticrossing model to describe the interaction between two modes.

    Parameters:
    freq (np.ndarray): Frequency array.
    field (np.ndarray): Magnetic field array.
    params (dict): Dictionary containing model parameters:
        - 'alpha': Array of magnon damping rates corresponding to the fields.
        - 'beta': Magnon decay rate.
        - 'kappa': Cavity decay rate.
        - 'gamma': Additional decay rate.
        - 'J': Coupling strength.
        - 'magnon_freq_slope': Magnon dispersion slope (GHz/Oe).
        - 'magnon_freq_intercept': Magnon dispersion intercept (GHz).
        - 'resonance_freq': Resonance frequency of the cavity.

    Returns:
    np.ndarray: Computed response based on the anticrossing model.
    """
    alpha = params["alpha"]
    beta = params["beta"]
    kappa = params["kappa"]
    gamma = params["gamma"]
    J = params["J"]
    G = kappa * gamma
    resonance_freq = params["resonance_freq"]
    magnon_slope = params["magnon_freq_slope"]
    magnon_intercept = params["magnon_freq_intercept"]

    magnon_freqs = magnon_slope * fields + magnon_intercept

    if len(fields) != len(magnon_freqs):
        raise ValueError("Length of field array must match length of magnon_freq array")

    linear_response = np.zeros((len(fields), len(freqs)), dtype=complex)

    for i, field in enumerate(fields):
        alpha = params['alpha'][i]
        gamma = params['gamma'][i]
        J = params['J'][i]
        Gamma =kappa * gamma
        magnon_freq = magnon_freqs[i]
        coupling = 1j * J + Gamma

        cavity_term = 1j * (freqs - resonance_freq) - (kappa + beta)
        magnon_term = 1j * (freqs - magnon_freq) - (alpha + gamma)

        denominator = cavity_term  - coupling**2 / magnon_term

        linear_response[i, :] = 1 + kappa / denominator
    
    return linear_response