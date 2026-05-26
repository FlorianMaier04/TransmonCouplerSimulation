import sys
from helper_function import *
from utils import *
from qutip import *
import numpy as np
import matplotlib.pyplot as plt

config = {
    'amplitude': 0.2,
    'order': 3,
    'pulse_shape': 'cos',
    'pulse_args': [0.5],
    'freq_range': 0.01,
    'freq_points': 50,
}

def get_max_fidelity_freq(simulator, wd_res, amp, psi0, tg, freq_range=0.01):
    def calc_fidelity(wd):
        heff = simulator.heff(wd, amp)
        phi0, ops = compute_phi0(simulator, wd, amp, psi0)
        time_transformed_h = compute_t_dependency(simulator, heff, wd, tg)
        result = mesolve(time_transformed_h, phi0, np.arange(0, tg+1, 0.01), c_ops=[], e_ops=ops, options={'nsteps': 50000})
        state_a_idx = list(simulator.resonances.keys()).index(simulator.state_a)
        return result.expect[state_a_idx][int(tg / 0.01)]
    
    freq_det = np.linspace(-freq_range, freq_range, 15)
    fids = [calc_fidelity(wd_res + d*2*np.pi) for d in freq_det]
    
    best_det = freq_det[np.argmax(fids)]
    fine_range = freq_range / 7
    freq_det_fine = np.linspace(best_det - fine_range, best_det + fine_range, 15)
    fids_fine = [calc_fidelity(wd_res + d*2*np.pi) for d in freq_det_fine]
    
    return freq_det_fine[np.argmax(fids_fine)]

if __name__ == "__main__":
    s = Simulation()
    s.config(config)
    
    amp = config['amplitude'] * 2 * np.pi
    wd = s.find_resonance(amp)

    heff = (s.heff(wd, amp))
    Omega_ab = heff[0][1]
    # find the optimal gate time
    tg, optimal_fidelity, _ = find_optimal_time(wd,amp,s.order,
        s.state_b,s.state_a,s.state_c,s.E_array,s.V1_dressed_array, int(np.pi/(2*Omega_ab)),)
    tlist = np.arange(0, tg, 0.04)
    
    psi0 = s.state_b
    
    freq_detunings = np.linspace(-config['freq_range'], config['freq_range'], config['freq_points'])
    frequencies = wd + freq_detunings * 2 * np.pi
    
    fidelities = []
    
    for wd in frequencies:
        heff = s.heff(wd, amp)
        Omega_ab = heff[0][1]
        
        phi0, ops = compute_phi0(s, wd, amp, psi0)
        # phi0 = qt.basis(simulator.sd, 1)
        
        time_transformed_h = compute_t_dependency(s, heff, wd, tg)
        n_steps = 50000
        
        result = mesolve(
            time_transformed_h,
            phi0,
            tlist,
            c_ops=[],
            e_ops=ops,
            options={'nsteps': n_steps},
        )
        
        state_a_idx = list(s.resonances.keys()).index(s.state_a)
        pop_a_at_tg = result.expect[state_a_idx][-1]
        fidelities.append(pop_a_at_tg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_detunings, fidelities, 'o-', linewidth=2, markersize=6)
    ax.set_xlabel('Frequency Detuning (GHz)', fontsize=12)
    ax.set_ylabel('Population of State a', fontsize=12)
    ax.set_title(f'Fidelity Sweep - Amplitude: {config["amplitude"]} GHz, Gate Time: {tg} ns', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    max_freq_det = get_max_fidelity_freq(s, wd, amp, psi0, tg, config['freq_range'])
    print(f"Max fidelity at frequency detuning: {max_freq_det:.6f} GHz")
    
    plt.tight_layout()
    plt.show()
