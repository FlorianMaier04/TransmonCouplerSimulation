import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from Floquet_perturbation_theory import *
from helper_function import *
from TransmonCouplerSimulation.historic.simulator import Simulator

config = {
    'lA': [0.001, 0.2, 5],
    'lwd': [0.7, 0.712, 5],
    'hr': False,
    'fast': False,
    'wd_detune': 0.0001, # in GHz
    'base_amplitude': 0.1,
    'base_wd': 'res',
    'plotA': True,
    'plotF': False,
    'tgate': 200, # in nanoseconds
    'pulse_shape':'cos', # 'gauss', 'cos', 'cossin'
    'pulse_args': [0.5], # gauss: sigma=value*tg
    'order': 3,
    'state_i': 'a',  # 'a', 'b', or 'c' - state index i for sweeps
    'state_j': 'b',  # 'a', 'b', or 'c' - state index j for sweeps (if i==j, sweeps delta_ii instead of omega_ij)
    'delta_diff': False,  # if True, calculates delta_ii - delta_jj
}

def plot_sweeps(simulator, config):
    """Plot sweeps für delta_ii, delta_ii - delta_jj, oder Omega_ij."""
    hr_factor = 8 if config['hr'] else 1
    if(config['fast']): hr_factor = 0.1
    lA = np.linspace(config['lA'][0], config['lA'][1]*hr_factor, int(config['lA'][2]*hr_factor)) * 2 * np.pi
    lwd = np.linspace(config['lwd'][0]/hr_factor, config['lwd'][1]*hr_factor, int(config['lwd'][2]*hr_factor)) * 2 * np.pi
    
    # Map state labels to state indices
    state_map = {'a': simulator.state_a, 'b': simulator.state_b, 'c': simulator.state_c}
    i = state_map[config['state_i']]
    j = state_map[config['state_j']]
    
    state_labels = {simulator.state_a: 'a', simulator.state_b: 'b', simulator.state_c: 'c'}
    label_i = state_labels[i]
    label_j = state_labels[j]
    
    # Determine if sweeping delta, delta_diff, or omega
    if config['delta_diff']:
        is_delta_diff = True
        is_delta = False
    else:
        is_delta = (i == j)
        is_delta_diff = False
    
    if is_delta:
        observable_label = f'δ_{{{label_i},{label_i}}}'
        observable_name = f'δ_{label_i}_{label_i}'
    elif is_delta_diff:
        observable_label = f'δ_{{{label_i},{label_i}}} - δ_{{{label_j},{label_j}}}'
        observable_name = f'δ_{label_i}_{label_i} - δ_{label_j}_{label_j}'
    else:
        observable_label = f'Ω_{{{label_i},{label_j}}}'
        observable_name = f'Omega_{label_i}_{label_j}'
    
    num_plots = int(config['plotA']) + int(config['plotF'])
    fig, axes = plt.subplots(1, num_plots, figsize=(7*num_plots, 6))
    if num_plots == 1:
        axes = [axes]
    
    pulse_args_str = ', '.join([f'{arg:.3f}' for arg in config['pulse_args']]) if config['pulse_shape']!='cos' else ''
    
    config_text = (f"Observable: {observable_name}  |  "
        f"Configuration: Pulse Shape: {config['pulse_shape']} "
        f"{pulse_args_str}  |  "
        f"Base Amplitude: {(simulator.base_amplitude/(2*np.pi)):.4f} GHz  |  "
        f"Base Frequency: {(simulator.base_wd/(2*np.pi)):.4f} GHz  |  "
        f"Tgate: {config['tgate']} ns  |  "
        f"Order: {config['order']}"
    )
    
    ax_idx = 0
    
    if config['plotA']:
        print("\n" + "="*70)
        if is_delta_diff:
            results_amp_i = simulator.sweep_amplitude(lA, simulator.base_wd, i, i)
            results_amp_j = simulator.sweep_amplitude(lA, simulator.base_wd, j, j)
            results_amp = results_amp_i - results_amp_j
        else:
            results_amp = simulator.sweep_amplitude(lA, simulator.base_wd, i, j)
        ax = axes[ax_idx]
        ax.plot(lA / (2 * np.pi), np.real(results_amp), 'o-', 
                linewidth=2, markersize=6, color='#217dbf', label=observable_label + " (GHz)")
        ax.set_xlabel('Amplitude (GHz)', fontsize=11)
        ax.set_ylabel(observable_label, fontsize=11)
        ax.set_title(f'Frequency Response vs Amplitude - {observable_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax_idx += 1
    
    if config['plotF']:
        print("\n" + "="*70)
        if is_delta_diff:
            results_freq_i = simulator.sweep_frequency(lwd, simulator.base_amplitude, i, i)
            results_freq_j = simulator.sweep_frequency(lwd, simulator.base_amplitude, j, j)
            results_freq = results_freq_i - results_freq_j
        else:
            results_freq = simulator.sweep_frequency(lwd, simulator.base_amplitude, i, j)
        ax = axes[ax_idx]
        ax.plot(lwd / (2 * np.pi), np.real(results_freq), 's-', 
                linewidth=2, markersize=6, color='#ff7f0e', label=observable_label)
        ax.set_xlabel('Frequency (GHz)', fontsize=11)
        ax.set_ylabel(observable_label, fontsize=11)
        ax.set_title(f'Frequency Response vs Frequency - {observable_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Füge Config-Text unter den Plots hinzu
    fig.text(0.5, 0.02, config_text, ha='center', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='monospace')
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig, axes


if __name__ == "__main__":
    simulator = Simulator()
    simulator.config(config)
    # base parameters
    if config['plotA'] or config['plotF']:
        fig, axes = plot_sweeps(simulator, config)
        plt.show()

