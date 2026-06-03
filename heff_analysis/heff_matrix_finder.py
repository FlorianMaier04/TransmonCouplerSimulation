import numpy as np
import matplotlib.pyplot as plt
from qutip import *
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Floquet_perturbation_theory import *
from helper_function import *
from utils import *

config = {
    'lA': [0.001, 0.4, 5],
    'lwd': [0.7, 0.712, 5],
    'hr': False,
    'fast': False,
    'base_amplitude': 0.2,
    'base_freq': 0.7102,
    'plotA': False,
    'plotF': True,
    'order': 3,
    'state_i': 'a',  # 'a', 'b', or 'c' - state index i for sweeps
    'state_j': 'b',  # 'a', 'b', or 'c' - state index j for sweeps (if i==j, sweeps delta_ii instead of omega_ij)
    'delta_diff': False,   # if True, calculates delta_ii - delta_jj
    'img': False,
    'img_amp': False,
}# 1,074524

def fit_data(x, y, x_label, y_label):
    r2_linear = 1 - np.sum((y - np.polyval(np.polyfit(x, y, 1), x))**2) / np.sum((y - np.mean(y))**2)
    
    if r2_linear > 0.99:
        coeffs = np.polyfit(x, y, 1)
        fit_y = np.polyval(coeffs, x)
        print(f"  Linear Fit ({y_label} vs {x_label}): y = {coeffs[0]:.6e}*x + {coeffs[1]:.6e}  (R² = {r2_linear:.6f})")
        print(f"[{coeffs[0]:.6f},{coeffs[1]:.6f}]")
        return coeffs, fit_y
    else:
        coeffs = np.polyfit(x, y, 2)
        fit_y = np.polyval(coeffs, x)
        r2_poly = 1 - np.sum((y - fit_y)**2) / np.sum((y - np.mean(y))**2)
        print(f"  Poly Fit ({y_label} vs {x_label}): y = {coeffs[0]:.6e}*x² + {coeffs[1]:.6e}*x + {coeffs[2]:.6e}  (R² = {r2_poly:.6f})")
        print(f"[{coeffs[0]:.6f},{coeffs[1]:.6f},{coeffs[2]:.6f}]")
        return coeffs, fit_y


def plot_sweeps(s, config):
    """Plot sweeps für delta_ii, delta_ii - delta_jj, oder Omega_ij."""
    hr_factor = 8 if config['hr'] else 1
    if(config['fast']): hr_factor = 0.1
    lA = np.linspace(config['lA'][0], config['lA'][1]*hr_factor, int(config['lA'][2]*hr_factor)) * 2 * np.pi
    lwd = np.linspace(config['lwd'][0]/hr_factor, config['lwd'][1]*hr_factor, int(config['lwd'][2]*hr_factor)) * 2 * np.pi
    base_amp = config['base_amplitude'] * 2*np.pi
    base_wd = config['base_freq'] * 2*np.pi

    # Map state labels to state indices
    state_map = {'a': s.state_a, 'b': s.state_b, 'c': s.state_c}
    state_map = {'a': 0, 'b':1, 'c':2}
    i = state_map[config['state_i']]
    j = state_map[config['state_j']]
    state_labels = {s.state_a: 'a', s.state_b: 'b', s.state_c: 'c'}
    state_labels = {0: 'a', 1: 'b', 2: 'c'}
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
        
    config_text = (f"Observable: {observable_name}  |  "
        f"Configuration: "
        f"Base Amplitude: {(base_amp/(2*np.pi)):.4f} GHz  |  "
        f"Base Frequency: {(base_wd/(2*np.pi)):.4f} GHz  |  "
        f"Order: {config['order']}"
    )
    print("\n" + "="*70)
    resA, resF = None, None
    ax_idx = 0
    if config['plotA']:
        factor = 1j if config['img_amp'] else 1
        if is_delta_diff:
            res_d1 = [s.heff(base_wd, amp*factor)[i][i] for amp in lA]
            res_d2 = [s.heff(base_wd, amp*factor)[j][j] for amp in lA]
            resA = [res_d1[i] - res_d2[i] for i in range(0, len(res_d1))]
        else: 
            resA = [s.heff(base_wd, amp*factor)[i][j] for amp in lA]
    if config['plotF']:
        if is_delta_diff:
            res_d1 = [s.heff(wd, base_amp)[i][i] for wd in lwd]
            res_d2 = [s.heff(wd, base_amp)[j][j] for wd in lwd]
            resF = res_d1 - res_d2
        else: 
            resF = [s.heff(wd, base_amp)[i][j] for wd in lwd]
    if resA:
        resA = np.real(resA) if not config['img'] else np.imag(resA)
        coeffs_A, fit_A = fit_data(lA / (2 * np.pi), resA / (2*np.pi), 'Amplitude', observable_name)
        ax = axes[ax_idx]
        ax.plot(lA / (2 * np.pi), resA / (2*np.pi), 'o-', 
                linewidth=2, markersize=6, color='#217dbf', label=observable_label + " (GHz)")
        ax.plot(lA / (2 * np.pi), fit_A, '--', linewidth=2, color='#d9534f', label='Fit')
        ax.set_xlabel('Amplitude (GHz)', fontsize=11)
        ax.set_ylabel(observable_label, fontsize=11)
        ax.set_title(f'Heff response vs Amplitude - {observable_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax_idx += 1
    if resF:
        resF = np.real(resF) if not config['img'] else np.imag(resF)
        coeffs_F, fit_F = fit_data(lwd / (2 * np.pi), resF / (2*np.pi), 'Frequenz', observable_name)
        ax = axes[ax_idx]
        ax.plot(lwd / (2 * np.pi), resF / (2*np.pi), 'o-', 
                linewidth=2, markersize=6, color='#217dbf', label=observable_label + " (GHz)")
        ax.plot(lwd / (2 * np.pi), fit_F, '--', linewidth=2, color='#d9534f', label='Fit')
        ax.set_xlabel('Frequenz (GHz)', fontsize=11)
        ax.set_ylabel(observable_label, fontsize=11)
        ax.set_title(f'Heff Response vs Frequenz - {observable_name}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax_idx += 1
        
    # Füge Config-Text unter den Plots hinzu
    fig.text(0.5, 0.02, config_text, ha='center', fontsize=9, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), family='monospace')
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig, axes


if __name__ == "__main__":
    simulator = Simulation()
    if config['plotA'] or config['plotF']:
        fig, axes = plot_sweeps(simulator, config)
        plt.show()

