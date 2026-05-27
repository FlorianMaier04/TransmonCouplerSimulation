import sys
from helper_function import *
from utils import *
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import LogFormatterMathtext, SymmetricalLogLocator
import matplotlib.ticker as ticker

# ============================================================================
# MODE 1: Sweep gate times and detunings around resonance frequency
# MODE 2: Sweep gate times and amplitudes
# MODE 3: Sweep gate times and absolute frequencies
# ============================================================================

MODE = 3  # Set to 1, 2, or 3

if MODE == 1:
    config = {
        'amplitude_value': 0.2,  # Fixed amplitude in GHz
        'gate_time_range': np.linspace(80, 130, 10),  # Gate times in ns
        'detuning_range': np.linspace(-0.05, 0.05, 10),  # Frequency detuning in GHz
    }
elif MODE == 2:
    config = {
        'gate_time_range': np.linspace(100, 250, 10),  # Gate times in ns
        'amplitude_range': np.linspace(0.05, 0.4, 10),  # Amplitudes in GHz
    }
elif MODE == 3:
    config = {
        'amplitude_value': 0.2,  # Fixed amplitude in GHz
        'gate_time_range': np.linspace(100, 130, 100),  # Gate times in ns
        'frequency_range': np.linspace(0.708, 0.716, 100),  # Absolute drive frequencies in GHz
    }

def get_fidelity(s, tg, wd, amp):
    """
    Calculate gate fidelity for given parameters.
    
    Parameters:
    - simulator: Simulation instance
    - tg: Gate time in ns
    - wd: Drive frequency in rad/ns
    - amp: Drive amplitude in rad/ns
    
    Returns:
    - fidelity: Fidelity value
    """
    heff = s.heff(wd, amp)            
    tlist = np.arange(0, tg, 0.01)
    time_transformed_h = compute_t_dependency(s, heff, tg, use_drag=True, use_gauss=True)
    fidelity = s.extract_fidelity(time_transformed_h, tlist)
    
    return fidelity

# Manual tick positions (powers of 10, focused on lower values)
MANUAL_TICKS = np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])

def get_ticks_in_range(vmin, vmax):
    valid_ticks = MANUAL_TICKS[(MANUAL_TICKS >= vmin * 0.99) & (MANUAL_TICKS <= vmax * 1.01)]
    if len(valid_ticks) > 0:
        return valid_ticks
    else:
        # Fallback: generate logarithmic ticks
        log_min = np.log10(vmin)
        log_max = np.log10(vmax)
        log_ticks = np.linspace(log_min, log_max, 5)
        return 10 ** log_ticks

def run_mode_1():
    """
    MODE 1: Sweep gate times and frequency detunings around resonance.
    """
    simulator = Simulation()
    
    amp_gHz = config['amplitude_value']
    amp = amp_gHz * 2 * np.pi
    
    gate_times = config['gate_time_range']
    detune_gHz = config['detuning_range']
    
    # Find resonance frequency for the fixed amplitude
    wd_res = simulator.find_resonance(amp)
    
    fidelity_map = np.zeros((len(detune_gHz), len(gate_times)))
    
    print(f"MODE 1: Sweeping {len(gate_times)} gate times × {len(detune_gHz)} detunings")
    print(f"Fixed amplitude: {amp_gHz} GHz")
    print(f"Resonance frequency: {wd_res / (2 * np.pi):.6f} GHz\n")
    
    for i, tg in enumerate(tqdm(gate_times, desc="Gate Time")):
        for j, detune_val in enumerate(detune_gHz):
            detune = detune_val * 2 * np.pi
            wd = wd_res + detune
            
            fidelity = get_fidelity(simulator, tg, wd, amp)
            fidelity_map[j, i] = fidelity if not np.isnan(fidelity) else 0.0
    
    infidelity_map = (1 - fidelity_map) * 100
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get actual minimum value (smallest non-zero infidelity)
    infidelity_nonzero = infidelity_map[infidelity_map > 0]
    vmin = np.min(infidelity_nonzero) if len(infidelity_nonzero) > 0 else 1e-6
    vmax = np.nanmax(infidelity_map) if np.nanmax(infidelity_map) > 0 else 1
    
    print(f"Infidelity range: {vmin:.2e} to {vmax:.2e}")
    
    im = ax.imshow(
        infidelity_map,
        cmap='viridis_r',
        aspect='auto',
        origin='lower',
        extent=[gate_times[0], gate_times[-1], detune_gHz[0], detune_gHz[-1]],
        norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    )
    
    ax.set_xlabel('Gate Time (ns)', fontsize=12)
    ax.set_ylabel('Frequency Detuning (GHz)', fontsize=12)
    ax.set_title(f'Gate Infidelity (MODE 1) - Amplitude: {amp_gHz} GHz', fontsize=13, fontweight='bold')
    
    # Get ticks in range [vmin, vmax]
    ticks = get_ticks_in_range(vmin, vmax)
    
    cbar = plt.colorbar(im, ticks=ticks)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cbar.set_label('1 - Fidelity', fontsize=11)
    
    plt.tight_layout()
    plt.show()

def run_mode_2():
    """
    MODE 2: Sweep gate times and amplitudes.
    """
    simulator = Simulation()
    
    psi0 = simulator.state_b if config['initial_state'] == 'b' else simulator.state_a
    
    gate_times = config['gate_time_range']
    amp_gHz = config['amplitude_range']
    
    fidelity_map = np.zeros((len(amp_gHz), len(gate_times)))
    
    print(f"MODE 2: Sweeping {len(gate_times)} gate times × {len(amp_gHz)} amplitudes\n")
    
    for i, tg in enumerate(tqdm(gate_times, desc="Gate Time")):
        for j, amp_val in enumerate(amp_gHz):
            amp = amp_val * 2 * np.pi
            wd_res = simulator.find_resonance(amp)
            
            # Use resonance frequency (zero detuning)
            fidelity = get_fidelity(simulator, tg, wd_res, amp)
            fidelity_map[j, i] = fidelity if not np.isnan(fidelity) else 0.0
    
    infidelity_map = 1 - fidelity_map
    
    # Plot results
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get actual minimum value (smallest non-zero infidelity)
    infidelity_nonzero = infidelity_map[infidelity_map > 0]
    vmin = np.min(infidelity_nonzero) if len(infidelity_nonzero) > 0 else 1e-6
    vmax = np.nanmax(infidelity_map) if np.nanmax(infidelity_map) > 0 else 1
    
    print(f"Infidelity range: {vmin:.2e} to {vmax:.2e}")
    
    im = ax.imshow(
        infidelity_map,
        cmap='viridis_r',
        aspect='auto',
        origin='lower',
        extent=[gate_times[0], gate_times[-1], amp_gHz[0], amp_gHz[-1]],
        norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    )
    
    ax.set_xlabel('Gate Time (ns)', fontsize=12)
    ax.set_ylabel('Amplitude (GHz)', fontsize=12)
    ax.set_title('Gate Infidelity (MODE 2)', fontsize=13, fontweight='bold')
    
    # Get ticks in range [vmin, vmax]
    ticks = get_ticks_in_range(vmin, vmax)
    
    cbar = plt.colorbar(im, ticks=ticks)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cbar.set_label('1 - Fidelity', fontsize=11)
    
    plt.tight_layout()
    plt.show()

def run_mode_3():
    """
    MODE 3: Sweep gate times and absolute drive frequencies.
    """
    simulator = Simulation()
    
    amp_gHz = config['amplitude_value']
    amp = amp_gHz * 2 * np.pi
    
    gate_times = config['gate_time_range']
    freq_gHz = config['frequency_range']
    
    fidelity_map = np.zeros((len(freq_gHz), len(gate_times)))
    
    print(f"MODE 3: Sweeping {len(gate_times)} gate times × {len(freq_gHz)} frequencies")
    print(f"Fixed amplitude: {amp_gHz} GHz\n")
    
    for i, tg in enumerate(tqdm(gate_times, desc="Gate Time")):
        for j, freq_val in enumerate(freq_gHz):
            wd = freq_val * 2 * np.pi
            
            fidelity = get_fidelity(simulator, tg, wd, amp)
            fidelity_map[j, i] = fidelity if not np.isnan(fidelity) else 0.0
    
    infidelity_map = 1 - fidelity_map
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    infidelity_nonzero = infidelity_map[infidelity_map > 0]
    vmin = np.min(infidelity_nonzero) if len(infidelity_nonzero) > 0 else 1e-6
    vmax = np.nanmax(infidelity_map) if np.nanmax(infidelity_map) > 0 else 1
    
    print(f"Infidelity range: {vmin:.2e} to {vmax:.2e}")
    
    im = ax.imshow(
        infidelity_map,
        cmap='viridis_r',
        aspect='auto',
        origin='lower',
        extent=[gate_times[0], gate_times[-1], freq_gHz[0], freq_gHz[-1]],
        norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)
    )
    
    ax.set_xlabel('Gate Time (ns)', fontsize=12)
    ax.set_ylabel('Drive Frequency (GHz)', fontsize=12)
    ax.set_title(f'Gate Infidelity (MODE 3) - Amplitude: {amp_gHz} GHz', fontsize=13, fontweight='bold')
    
    ticks = get_ticks_in_range(vmin, vmax)
    
    cbar = plt.colorbar(im, ticks=ticks)
    cbar.ax.yaxis.set_major_formatter(LogFormatterMathtext())
    cbar.set_label('1 - Fidelity', fontsize=11)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if MODE == 1:
        run_mode_1()
    elif MODE == 2:
        run_mode_2()
    elif MODE == 3:
        run_mode_3()
    else:
        print("Error: MODE must be 1, 2, or 3")