"""Demo script to test ETotalApproximator visualization."""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from executors.ETotalApproximator import ETotalApproximator, S21_coupled_resonators
from loaders.LoadETotalApproximator import LoadETotalApproximator
from utils.unit_transformations import convert_linear_to_dB


def main():
    print("=" * 70)
    print("ETotalApproximator Visualization Demo")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    fields = np.linspace(2800, 3000, 30)
    freqs = np.linspace(3.6, 4.0, 40)

    # Ground truth parameters
    kappa_c = 0.05
    beta = 0.01
    kappa_tot = kappa_c + beta
    f_c = 3.8
    magnon_slope = 0.001
    magnon_intercept = 2.5
    g_true = 0.015
    kappa_m_true = 0.04
    gamma_tot_true = 0.03

    print(f"   Ground truth: g = {g_true:.4f} GHz, kappa_m = {kappa_m_true:.4f} GHz, gamma_tot = {gamma_tot_true:.4f} GHz")

    field_grid, freq_grid = np.meshgrid(fields, freqs, indexing='ij')

    # Generate ground truth S21
    S21_true = S21_coupled_resonators(
        freq_grid, field_grid,
        kappa_c, kappa_tot, f_c,
        magnon_slope, magnon_intercept,
        g_true, kappa_m_true, gamma_tot_true
    )

    # Convert to dB and add realistic noise
    z_dB = convert_linear_to_dB(np.abs(S21_true))
    np.random.seed(42)
    z_dB += np.random.randn(*z_dB.shape) * 0.2  # Realistic noise level

    # Prepare data dict
    data = {
        "x": fields,
        "y": freqs,
        "z": z_dB,
        "kappa": kappa_c,
        "beta": beta,
        "resonance_freq": f_c,
        "slope": magnon_slope,
        "intercept": magnon_intercept,
    }

    # Run approximator
    print("\n2. Running ETotalApproximator...")
    approximator = ETotalApproximator(
        data,
        "Visualization Demo",
        initial_params={"g_init": 0.01, "kappa_m_init": 0.05, "gamma_tot_init": 0.05}
    )
    approximator.execute_with_validation()
    result = approximator.get_result()

    print("\n3. Fit results:")
    print(f"   g         = {result['g']:.6f} GHz (true: {g_true:.4f})")
    print(f"   kappa_m   = {result['kappa_m']:.6f} GHz (true: {kappa_m_true:.4f})")
    print(f"   gamma_tot = {result['gamma_tot']:.6f} GHz (true: {gamma_tot_true:.4f})")
    print(f"   Error g:         {abs(result['g'] - g_true)/g_true * 100:.2f}%")
    print(f"   Error kappa_m:   {abs(result['kappa_m'] - kappa_m_true)/kappa_m_true * 100:.2f}%")
    print(f"   Error gamma_tot: {abs(result['gamma_tot'] - gamma_tot_true)/gamma_tot_true * 100:.2f}%")

    # Compute residual statistics
    residual = result['z'] - result['z_fit']
    rms_error = np.sqrt(np.mean(residual**2))
    max_error = np.max(np.abs(residual))
    
    print(f"\n4. Residual statistics:")
    print(f"   RMS error: {rms_error:.4f} dB")
    print(f"   Max error: {max_error:.4f} dB")

    # The figure was already created by prepare_for_visualization
    print("\n5. Checking visualization...")
    print("   (3-panel plot: Original | Fit | Residual)")
    
    if hasattr(approximator, 'figure') and approximator.figure is not None:
        print("   ✓ Figure created successfully!")
        print(f"   ✓ Figure has {len(approximator.figure.axes)} axes (expected: 6)")
        # In interactive mode, you would call plt.show() here
        plt.close('all')  # Close for testing
    else:
        print("   ✗ ERROR: No figure was created!")

    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
