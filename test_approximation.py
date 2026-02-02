import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.gridspec import GridSpec

from loaders import LoadContourTXT, LoadIEPeakWatcher
from utils.phys_models import Fano
import utils.unit_transformations as ut

# ========== PATHS ==========
trajectories_path = r"results/mease2026S21/13/ie_peak_watcher_result.txt"
data_path = "data/mease2026S21.txt"

# ========== FANO MODEL FOR FITTING (works in dB) ==========
def fano_model_dB(x, x0, gamma, q, A, y0):
    """
    Fano resonance model that works directly with dB data.
    Returns values in dB.
    """
    epsilon = (x - x0) / (gamma / 2)
    fano_normalized = ((q + epsilon)**2) / (1 + epsilon**2) / (1 + q**2)
    y0_linear = ut.convert_dB_to_linear(y0)
    result_linear = A * fano_normalized + y0_linear
    return ut.convert_linear_to_dB(result_linear)

def weighted_fit_fano(freqs, magnitudes_dB, x0_init, gamma_init, peak_focus_factor=5.0):
    """
    Fits Fano model with emphasis on peak region.
    
    Parameters:
    -----------
    freqs : array - frequency array
    magnitudes_dB : array - S21 magnitude in dB
    x0_init : float - initial guess for resonance frequency
    gamma_init : float - initial guess for width
    peak_focus_factor : float - how much more weight to give to peak region
    
    Returns:
    --------
    popt : optimized parameters [x0, gamma, q, A, y0]
    pcov : covariance matrix
    """
    # Create weights: higher near peak, lower in tails
    distance_from_peak = np.abs(freqs - x0_init)
    sigma = gamma_init * 2  # Weight drops off at ~2*gamma
    weights = 1 + (peak_focus_factor - 1) * np.exp(-distance_from_peak**2 / (2 * sigma**2))
    
    # Initial guesses
    y0_init = np.max(magnitudes_dB)  # baseline (away from peak)
    A_init = ut.convert_dB_to_linear(y0_init) - ut.convert_dB_to_linear(np.min(magnitudes_dB))
    q_init = 0.1  # small asymmetry initially
    
    p0 = [x0_init, gamma_init, q_init, A_init, y0_init]
    
    # Bounds to prevent divergence to neighboring peaks
    freq_range = freqs[-1] - freqs[0]
    bounds = (
        [x0_init - gamma_init * 2, gamma_init * 0.1, -10, 0, y0_init - 20],  # lower
        [x0_init + gamma_init * 2, gamma_init * 10, 10, A_init * 10, y0_init + 5]  # upper
    )
    
    try:
        popt, pcov = curve_fit(
            fano_model_dB, freqs, magnitudes_dB, 
            p0=p0, 
            bounds=bounds,
            sigma=1/weights,  # Lower sigma = higher weight
            absolute_sigma=False,
            maxfev=5000
        )
        return popt, pcov
    except Exception as e:
        print(f"Fitting failed: {e}")
        return p0, None

# ========== LOAD DATA ==========
print("Loading data...")
contour_loader = LoadContourTXT({
    "data_path": data_path,
    "last_line_err": True
})
contour_data = contour_loader.load_data()
fields = contour_data["x"]  # magnetic field values
freqs = contour_data["y"]   # frequencies
z_matrix = contour_data["z"]  # S21 magnitude matrix (field x freq)

print(f"Data loaded: fields {fields.shape}, freqs {freqs.shape}, z_matrix {z_matrix.shape}")

# ========== LOAD TRAJECTORIES ==========
print("Loading trajectories...")
traj_loader = LoadIEPeakWatcher({"data_path": trajectories_path})
traj_data = traj_loader.load_data()
trajectories = traj_data["trajectories"]

print(f"Loaded {len(trajectories)} trajectories")

# ========== SETUP REAL-TIME PLOTTING ==========
plt.ion()  # Interactive mode for real-time updates

# Create figure with subplots for parameters
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig)

# Subplots for fitted parameters
ax_x0 = fig.add_subplot(gs[0, 0])
ax_gamma = fig.add_subplot(gs[0, 1])
ax_q = fig.add_subplot(gs[0, 2])
ax_A = fig.add_subplot(gs[1, 0])
ax_y0 = fig.add_subplot(gs[1, 1])
ax_fit_quality = fig.add_subplot(gs[1, 2])
ax_current_fit = fig.add_subplot(gs[2, :])

ax_x0.set_xlabel('Field')
ax_x0.set_ylabel('x0 (GHz)')
ax_x0.set_title('Resonance Frequency x0')

ax_gamma.set_xlabel('Field')
ax_gamma.set_ylabel('γ (GHz)')
ax_gamma.set_title('Width γ')

ax_q.set_xlabel('Field')
ax_q.set_ylabel('q')
ax_q.set_title('Fano Parameter q')

ax_A.set_xlabel('Field')
ax_A.set_ylabel('A')
ax_A.set_title('Amplitude A')

ax_y0.set_xlabel('Field')
ax_y0.set_ylabel('y0 (dB)')
ax_y0.set_title('Baseline y0')

ax_fit_quality.set_xlabel('Field')
ax_fit_quality.set_ylabel('Peak MSE (dB²)')
ax_fit_quality.set_title('Fit Quality (Peak Region)')

ax_current_fit.set_xlabel('Frequency (GHz)')
ax_current_fit.set_ylabel('S21 (dB)')
ax_current_fit.set_title('Current Fit')

plt.tight_layout()

# ========== FIT EACH TRAJECTORY ==========
colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))

all_results = []

for traj_idx, traj in enumerate(trajectories):
    print(f"\n=== Processing Trajectory {traj_idx + 1} ===")
    
    traj_fields = np.array(traj["fields"])
    traj_freqs = np.array(traj["freq"])
    traj_widths = np.array(traj["width"])
    traj_magnitudes = np.array(traj["magnitude"])
    
    # Results storage for this trajectory
    results = {
        "fields": [],
        "x0": [],
        "gamma": [],
        "q": [],
        "A": [],
        "y0": [],
        "peak_mse": []
    }
    
    for i, (field, expected_freq, expected_width) in enumerate(zip(traj_fields, traj_freqs, traj_widths)):
        # Find field index in data
        field_idx = np.argmin(np.abs(fields - field))
        
        # Extract frequency slice at this field
        s21_slice = z_matrix[field_idx, :]
        
        # Define fitting window around expected peak
        window_half = max(expected_width * 5, 0.02)  # At least 20 MHz window
        freq_mask = (freqs >= expected_freq - window_half) & (freqs <= expected_freq + window_half)
        
        if np.sum(freq_mask) < 10:
            print(f"  Field {field:.2f}: Not enough points in window, skipping")
            continue
        
        fit_freqs = freqs[freq_mask]
        fit_magnitudes = s21_slice[freq_mask]
        
        # Perform weighted fit
        popt, pcov = weighted_fit_fano(
            fit_freqs, fit_magnitudes, 
            expected_freq, expected_width,
            peak_focus_factor=10.0  # Strong focus on peak
        )
        
        x0_fit, gamma_fit, q_fit, A_fit, y0_fit = popt
        
        # Calculate fit quality in peak region only
        peak_mask = np.abs(fit_freqs - x0_fit) < gamma_fit
        if np.sum(peak_mask) > 0:
            fitted_values = fano_model_dB(fit_freqs[peak_mask], *popt)
            peak_mse = np.mean((fit_magnitudes[peak_mask] - fitted_values)**2)
        else:
            peak_mse = np.nan
        
        # Store results
        results["fields"].append(field)
        results["x0"].append(x0_fit)
        results["gamma"].append(gamma_fit)
        results["q"].append(q_fit)
        results["A"].append(A_fit)
        results["y0"].append(y0_fit)
        results["peak_mse"].append(peak_mse)
        
        # Update plots every 10 points or at the end
        if i % 10 == 0 or i == len(traj_fields) - 1:
            color = colors[traj_idx]
            
            # Clear and replot
            ax_x0.clear()
            ax_gamma.clear()
            ax_q.clear()
            ax_A.clear()
            ax_y0.clear()
            ax_fit_quality.clear()
            ax_current_fit.clear()
            
            # Plot all previous trajectories
            for prev_idx, prev_res in enumerate(all_results):
                prev_color = colors[prev_idx]
                ax_x0.plot(prev_res["fields"], prev_res["x0"], 'o-', color=prev_color, alpha=0.5, markersize=2)
                ax_gamma.plot(prev_res["fields"], prev_res["gamma"], 'o-', color=prev_color, alpha=0.5, markersize=2)
                ax_q.plot(prev_res["fields"], prev_res["q"], 'o-', color=prev_color, alpha=0.5, markersize=2)
                ax_A.plot(prev_res["fields"], prev_res["A"], 'o-', color=prev_color, alpha=0.5, markersize=2)
                ax_y0.plot(prev_res["fields"], prev_res["y0"], 'o-', color=prev_color, alpha=0.5, markersize=2)
                ax_fit_quality.plot(prev_res["fields"], prev_res["peak_mse"], 'o-', color=prev_color, alpha=0.5, markersize=2)
            
            # Plot current trajectory progress
            ax_x0.plot(results["fields"], results["x0"], 'o-', color=color, markersize=3, label=f'Traj {traj_idx+1}')
            ax_gamma.plot(results["fields"], results["gamma"], 'o-', color=color, markersize=3)
            ax_q.plot(results["fields"], results["q"], 'o-', color=color, markersize=3)
            ax_A.plot(results["fields"], results["A"], 'o-', color=color, markersize=3)
            ax_y0.plot(results["fields"], results["y0"], 'o-', color=color, markersize=3)
            ax_fit_quality.plot(results["fields"], results["peak_mse"], 'o-', color=color, markersize=3)
            
            # Plot current fit
            ax_current_fit.plot(fit_freqs, fit_magnitudes, 'b.', markersize=4, label='Data')
            fitted_curve = fano_model_dB(fit_freqs, *popt)
            ax_current_fit.plot(fit_freqs, fitted_curve, 'r-', linewidth=2, label='Fano fit')
            ax_current_fit.axvline(x0_fit, color='g', linestyle='--', alpha=0.5, label=f'x0={x0_fit:.6f}')
            ax_current_fit.axvline(expected_freq, color='orange', linestyle=':', alpha=0.5, label=f'expected={expected_freq:.6f}')
            
            # Labels
            ax_x0.set_xlabel('Field')
            ax_x0.set_ylabel('x0 (GHz)')
            ax_x0.set_title('Resonance Frequency x0')
            ax_x0.legend(loc='best', fontsize=8)
            
            ax_gamma.set_xlabel('Field')
            ax_gamma.set_ylabel('γ (GHz)')
            ax_gamma.set_title('Width γ')
            
            ax_q.set_xlabel('Field')
            ax_q.set_ylabel('q')
            ax_q.set_title('Fano Parameter q')
            
            ax_A.set_xlabel('Field')
            ax_A.set_ylabel('A')
            ax_A.set_title('Amplitude A')
            
            ax_y0.set_xlabel('Field')
            ax_y0.set_ylabel('y0 (dB)')
            ax_y0.set_title('Baseline y0')
            
            ax_fit_quality.set_xlabel('Field')
            ax_fit_quality.set_ylabel('Peak MSE (dB²)')
            ax_fit_quality.set_title('Fit Quality (Peak Region)')
            
            ax_current_fit.set_xlabel('Frequency (GHz)')
            ax_current_fit.set_ylabel('S21 (dB)')
            ax_current_fit.set_title(f'Field={field:.2f}, Traj {traj_idx+1}, Point {i+1}/{len(traj_fields)}')
            ax_current_fit.legend(loc='best', fontsize=8)
            
            plt.tight_layout()
            plt.pause(0.01)
        
        # Progress output
        if i % 20 == 0:
            print(f"  Field {field:.2f}: x0={x0_fit:.6f}, γ={gamma_fit:.6f}, q={q_fit:.3f}, MSE={peak_mse:.4f}")
    
    all_results.append(results)
    print(f"Trajectory {traj_idx + 1} complete: {len(results['fields'])} points fitted")

# ========== FINAL SUMMARY PLOT ==========
plt.ioff()
fig2, axes = plt.subplots(2, 3, figsize=(14, 8))
ax_x0_final, ax_gamma_final, ax_q_final = axes[0]
ax_A_final, ax_y0_final, ax_mse_final = axes[1]

for traj_idx, res in enumerate(all_results):
    color = colors[traj_idx]
    label = f'Trajectory {traj_idx + 1}'
    ax_x0_final.plot(res["fields"], res["x0"], 'o-', color=color, markersize=3, label=label)
    ax_gamma_final.plot(res["fields"], res["gamma"], 'o-', color=color, markersize=3)
    ax_q_final.plot(res["fields"], res["q"], 'o-', color=color, markersize=3)
    ax_A_final.plot(res["fields"], res["A"], 'o-', color=color, markersize=3)
    ax_y0_final.plot(res["fields"], res["y0"], 'o-', color=color, markersize=3)
    ax_mse_final.plot(res["fields"], res["peak_mse"], 'o-', color=color, markersize=3)

ax_x0_final.set_xlabel('Field')
ax_x0_final.set_ylabel('x0 (GHz)')
ax_x0_final.set_title('Resonance Frequency x0')
ax_x0_final.legend(fontsize=8)

ax_gamma_final.set_xlabel('Field')
ax_gamma_final.set_ylabel('γ (GHz)')
ax_gamma_final.set_title('Width γ')

ax_q_final.set_xlabel('Field')
ax_q_final.set_ylabel('q')
ax_q_final.set_title('Fano Parameter q')

ax_A_final.set_xlabel('Field')
ax_A_final.set_ylabel('A')
ax_A_final.set_title('Amplitude A')

ax_y0_final.set_xlabel('Field')
ax_y0_final.set_ylabel('y0 (dB)')
ax_y0_final.set_title('Baseline y0')

ax_mse_final.set_xlabel('Field')
ax_mse_final.set_ylabel('Peak MSE (dB²)')
ax_mse_final.set_title('Fit Quality (Peak Region)')

plt.tight_layout()
plt.suptitle('Fano Approximation Results', y=1.02, fontsize=14)
plt.show()

print("\n=== All trajectories processed ===")
for traj_idx, res in enumerate(all_results):
    print(f"Trajectory {traj_idx + 1}: {len(res['fields'])} points")
    if res['fields']:
        print(f"  Fields: {min(res['fields']):.2f} - {max(res['fields']):.2f}")
        print(f"  x0 range: {min(res['x0']):.6f} - {max(res['x0']):.6f} GHz")
        print(f"  Mean Peak MSE: {np.nanmean(res['peak_mse']):.4f} dB²")