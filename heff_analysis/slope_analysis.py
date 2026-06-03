import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from TransmonCouplerSimulation.historic.simulator import Simulator
from tqdm import tqdm

# Configuration similar to heff_theory.py
config = {
    'tgate': 200,
    'pulse_shape': 'cos',
    'pulse_args': [0.5],
    'order': 3,
    'base_amplitude': 0.1, # in GHz
    'base_wd':1,
    'wd_detune':1
}

def linear_fit(x, a, b):
    """Linear function for fitting."""
    return a * x + b

def compute_slope_for_amplitude(simulator, amplitude_ghz, freq_range=None):
    """
    Compute the slope of |da - db| vs frequency for a given amplitude.
    
    Args:
        simulator: Simulator object
        amplitude_ghz: Amplitude in GHz
        freq_range: tuple (f_min, f_max, num_points) in GHz or None to use default
    
    Returns:
        slope: Slope of the linear fit
        freq_ghz: Frequency values (GHz)
        delta_diff_mag: Magnitude |da - db|
    """
    if freq_range is None:
        f_min, f_max, num_points = 0.7, 0.712, 10
    else:
        f_min, f_max, num_points = freq_range
    
    # Create frequency sweep
    lwd = np.linspace(f_min, f_max, num_points) * 2 * np.pi  # convert to rad/ns
    
    # Get state indices
    state_a = simulator.state_a  # |100>
    state_b = simulator.state_b  # |001>
    
    # Sweep frequency to get da and db
    da = simulator.sweep_frequency(lwd, amplitude_ghz * 2 * np.pi, state_a, state_a)
    db = simulator.sweep_frequency(lwd, amplitude_ghz * 2 * np.pi, state_b, state_b)
    
    # Compute magnitude |da - db|
    delta_diff = da - db
    
    # Convert frequency back to GHz for fitting
    freq_ghz = lwd / (2 * np.pi)
    
    # Fit linear function: |da - db| = a * f + b
    try:
        popt, _ = curve_fit(linear_fit, freq_ghz, delta_diff)
        slope = popt[0]
        intercept = popt[1]
        print("slope: ", slope, " intercept: ", intercept)
        print("data:" , delta_diff)
    except:
        slope = np.nan
    
    return slope, intercept

if __name__ == "__main__":
    # Initialize simulator
    simulator = Simulator()
    simulator.config(config)
    
    # Amplitude sweep
    amplitudes_ghz = np.linspace(0, 0.2, 4)  # sweep from 0.01 to 0.2 GHz
    lslope = []
    lintercept = []
    
    print("Computing slopes for different amplitudes...")
    for amp in tqdm(amplitudes_ghz, desc="Amplitude sweep"):
        slope, intercept = compute_slope_for_amplitude(simulator, amp)
        lslope.append(slope)
        lintercept.append(intercept)
    
    lslope = np.array(lslope)
    lintercept = np.array(lintercept)
    lroot = -lintercept/lslope
    
    # Plot: Slopes vs Amplitude with separate y-axes for each quantity
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Primary axis: |Slope|
    color1 = 'tab:red'
    ax1.set_xlabel('Amplitude (GHz)', fontsize=12)
    ax1.set_ylabel('|Slope|', fontsize=11, color=color1)
    ax1.plot(amplitudes_ghz, np.abs(lslope), 'o-', linewidth=2, markersize=8, label='|Slope|', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Secondary axis: Slope
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Slope', fontsize=11, color=color2)
    ax2.plot(amplitudes_ghz, lslope, 's-', linewidth=2, markersize=7, label='Slope', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Tertiary axis: intercept
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    color3 = 'tab:green'
    ax3.set_ylabel('Intercept', fontsize=11, color=color3)
    ax3.plot(amplitudes_ghz, lintercept, '^-', linewidth=2, markersize=7, label='Intercept', color=color3)
    ax3.tick_params(axis='y', labelcolor=color3)
    
    # Quaternary axis: root
    ax4 = ax1.twinx()
    ax4.spines['right'].set_position(('outward', 120))
    color4 = 'tab:orange'
    ax4.set_ylabel('Root', fontsize=11, color=color4)
    ax4.plot(amplitudes_ghz, lroot, 'D-', linewidth=2, markersize=7, label='Root', color=color4)
    ax4.tick_params(axis='y', labelcolor=color4)
    
    ax1.set_title('Linear Fit Parameters vs Drive Amplitude', fontsize=13, fontweight='bold')
    
    # Combine legends from all axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    
    ax1.legend(lines1 + lines2 + lines3 + lines4, labels1 + labels2 + labels3 + labels4, 
               fontsize=10, loc='upper left')
    
    fig.tight_layout()
    plt.show()
    
    # Print some results
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    for amp, slope in zip(amplitudes_ghz, lslope):
        print(f"Amplitude: {amp:.4f} GHz  ->  Slope: {slope:.6e}")
