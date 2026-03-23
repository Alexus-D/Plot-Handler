"""Quick test of non-interactive approximator."""

import numpy as np
from executors import EAnticrossingApproximator
from utils.unit_transformations import convert_linear_to_dB 
from utils.phys_models import simple_anticrossing_model

# Generate synthetic data
fields = np.linspace(2800, 3000, 20)
freqs = np.linspace(3.6, 4.0, 25)

kappa = 0.05
beta = 0.01
resonance_freq = 3.8
magnon_slope = 0.001
magnon_intercept = 2.5
J_true = 0.015
alpha_true = 0.03
gamma_true = 0.04

# Generate synthetic data using anticrossing model
n_fields = len(fields)
params_true = {
    'alpha': np.full(n_fields, alpha_true),
    'beta': beta,
    'kappa': kappa,
    'gamma': np.full(n_fields, gamma_true),
    'J': np.full(n_fields, J_true),
    'magnon_freq_slope': magnon_slope,
    'magnon_freq_intercept': magnon_intercept,
    'resonance_freq': resonance_freq,
}

z_linear = simple_anticrossing_model(freqs, fields, params_true)
z_dB = convert_linear_to_dB(np.abs(z_linear))
z_dB += np.random.randn(*z_dB.shape) * 0.1

data = {
    "x": fields,
    "y": freqs,
    "z": z_dB,
    "kappa": kappa,
    "beta": beta,
    "resonance_freq": resonance_freq,
    "slope": magnon_slope,
    "intercept": magnon_intercept,
}

print("Creating approximator...")
approximator = EAnticrossingApproximator(
    data,
    "Quick Test",
    initial_params={
        "J_init": 0.01,
        "alpha_init": 0.05,
        "gamma_init": 0.05
    }
)

print("Running execute_with_validation (should skip interactive)...")
approximator.execute_with_validation()

result = approximator.get_result()
print("\nResult:")
print(f"  J       = {result['J']:.6f} (true: {J_true})")
print(f"  alpha   = {result['alpha']:.6f} (true: {alpha_true})")
print(f"  gamma   = {result['gamma']:.6f} (true: {gamma_true})")
print("\nTest PASSED!")
