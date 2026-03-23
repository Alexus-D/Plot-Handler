"""Test script for interactive 2D approximator parameter selection."""

import numpy as np
import matplotlib.pyplot as plt
from executors import EAnticrossingApproximator
from utils.phys_models import simple_anticrossing_model
from utils.unit_transformations import convert_linear_to_dB

# Generate synthetic 2D data using anticrossing model
print("Generating synthetic anticrossing data...")
fields = np.linspace(2800, 3200, 30)
freqs = np.linspace(3.6, 4.2, 40)

# Real model parameters (ground truth)
true_J = 0.056440      # GHz - coupling strength
true_alpha = 0.001000   # GHz - magnon damping
true_gamma = 0.001000   # GHz - additional decay
kappa = 0.1000        # GHz - cavity decay
beta = 0.0100         # GHz - magnon decay
resonance_freq = 3.9  # GHz - cavity resonance
magnon_slope = 0.002433  # GHz/Oe - magnon dispersion
magnon_intercept = -3.2712  # GHz - magnon intercept

# Generate synthetic data using anticrossing model
n_fields = len(fields)
params_true = {
    'alpha': np.full(n_fields, true_alpha),
    'beta': beta,
    'kappa': kappa,
    'gamma': np.full(n_fields, true_gamma),
    'J': np.full(n_fields, true_J),
    'magnon_freq_slope': magnon_slope,
    'magnon_freq_intercept': magnon_intercept,
    'resonance_freq': resonance_freq,
}

print(f"  True parameters: J={true_J:.4f}, alpha={true_alpha:.4f}, gamma={true_gamma:.4f}")

# Generate clean response
z_linear = simple_anticrossing_model(freqs, fields, params_true)
z_dB = convert_linear_to_dB(np.abs(z_linear))

# Add realistic noise
noise_level = 0.1  # dB
z_dB += np.random.randn(*z_dB.shape) * noise_level

# Prepare data dict
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

print("Creating EAnticrossingApproximator executor...")
executor = EAnticrossingApproximator(data, "test_anticrossing", initial_params={})

print("\n" + "="*60)
print("STEP 1: Interactive parameter selection")
print("="*60)
print("Instructions:")
print("1. Click on the contour plot to define polygon vertices")
print("2. Press Enter or Esc to complete the polygon")
print("3. Adjust sliders to set initial parameter guesses")
print("4. Watch the preview plots update in real-time")
print("5. Click 'Done' when satisfied")
print("="*60)

# This will show the interactive selector
executor.select_initial_params()

print("\nSelected initial parameters:")
print(f"  J_init: {executor.initial_params.get('J_init', 'N/A')}")
print(f"  alpha_init: {executor.initial_params.get('alpha_init', 'N/A')}")
print(f"  gamma_init: {executor.initial_params.get('gamma_init', 'N/A')}")

if 'polygon_mask' in executor.initial_params:
    mask = executor.initial_params['polygon_mask']
    print(f"  Polygon region: {np.sum(mask)} / {mask.size} points ({100*np.sum(mask)/mask.size:.1f}%)")
else:
    print("  Polygon region: Not defined (will use all data)")

print("\n" + "="*60)
print("STEP 2: Running optimization...")
print("="*60)

# Run the fitting
executor.execute()

print("\n" + "="*60)
print("STEP 3: Interactive validation")
print("="*60)
print("Instructions:")
print("1. Inspect the three-panel plot (Experiment, Fit, Residual)")
print("2. Click 'Accept' to keep the fit, or 'Deny' to reject")
print("="*60)

# This will show the validation plot
validation_result = executor.validate()

print(f"\nValidation result: {'ACCEPTED' if validation_result else 'REJECTED'}")

if validation_result:
    result = executor.get_result()
    print("\nFitted parameters:")
    print(f"  J       = {result.get('J', 'N/A'):.6f} GHz  (true: {true_J:.6f})")
    print(f"  alpha   = {result.get('alpha', 'N/A'):.6f} GHz  (true: {true_alpha:.6f})")
    print(f"  gamma   = {result.get('gamma', 'N/A'):.6f} GHz  (true: {true_gamma:.6f})")
    
    print("\nFitting errors:")
    print(f"  ΔJ       = {abs(result.get('J', 0) - true_J):.6f} GHz")
    print(f"  Δalpha   = {abs(result.get('alpha', 0) - true_alpha):.6f} GHz")
    print(f"  Δgamma   = {abs(result.get('gamma', 0) - true_gamma):.6f} GHz")

print("\nTest complete!")
