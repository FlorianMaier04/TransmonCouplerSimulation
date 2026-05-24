import sys
from helper_function import *
from utils import Simulator
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

config = {
    'show_normal': True, 
    'initial_state': 'a',  # 'a' or 'b' for initial state selection
    'amplitude': 0.1,  # in GHz
    'frequency': 0.7063365266894976 ,  # in GHz (res: 0.7063365266894976) (5.41521)
    'order': 3,
    'tgate': 300,  # in ns
    'pulse_shape': 'cos',  # 'cos', 'cossin', or 'gauss'
    'pulse_args': [0.5],  # for gauss: [sigma_fraction]
}

def simulate_normal(simulator, wd, amp, psi0, tg=200):
    tlist = np.linspace(0, tg, 1000)
    psi0_state = simulator.E_states[psi0]
    ops = [simulator.E_states[state] @ simulator.E_states[state].dag() for state in simulator.resonances.keys()]
    result = mesolve([simulator.H0_dressed, [simulator.V1_dressed, lambda t,args: amp*np.cos(wd*t)]], 
                     psi0_state,tlist, [], ops)
    return result, tlist

def compute_phi0(simulator, wd, amp, psi0): 
    rW = 2
    W = W_Floquet_elements(rW, wd, simulator.resonances, simulator.E_array, amp/2 * simulator.V1_dressed_array,
                        None, ref_state = None, analytics = False)
    # express phi0 in the resonant subspace, assuming psi0 is no superposition and lies in the res. subspace
    phi0 = qt.zero_ket(simulator.sd)
    projections = [None] * simulator.sd
    for idx, a in enumerate(simulator.resonances.keys()):
        dic = W[rW, psi0, a]
        n_psi0 = simulator.resonances[psi0] 
        if n_psi0 in dic: c = np.conj(dic[n_psi0])
        else: c = 0
        phi0 += c * qt.basis(simulator.sd, idx)
        projections[idx] = qt.basis(simulator.sd, idx).proj()
        # projections[idx] = c * np.conj(c) * qt.basis(simulator.sd, idx).proj()
    phi0 = phi0.unit()
    print("phi0: ", phi0)
    return phi0, projections

def simulate_sambe(simulator, wd, amp, psi0, tg=200):
    """Simulate population dynamics using mesolve."""
    # TODO: check Hermicity, Complex description and Dimension
    ops = [qt.basis(simulator.sd, idx).proj() for idx, _ in enumerate(simulator.resonances)]
    heff_array = simulator.heff(wd, amp)
    # heff_array[1][1] = 29.6
    heff = Qobj(heff_array)
    
    tlist = np.linspace(0, tg, 1000)
    
    phi0, ops = compute_phi0(simulator, wd, amp, psi0)
    display(heff)
    print(type(ops[1]))
    # Simulate using mesolve
    result = mesolve(
        [heff],
        phi0,
        tlist,
        [],
        ops,
    )
    return result, tlist

def show_result(simulator, r, tlist, ax_normal, ax_log, sambe=True):
    linestyle = 'dashed' if sambe else '-'

    n_states = len(simulator.resonances)

    # Basis-Farben für Theorie
    theory_colors = plt.cm.tab10(np.linspace(0, 1, n_states))

    # Simulationsfarben: leicht verschoben in derselben Palette
    sim_colors = plt.cm.Set2(np.linspace(0, 1, n_states))

    # Transparenz: Simulation blasser
    alpha = 1.0 if sambe else 0.4
    linewidth = 2.5 if sambe else 2

    colors = theory_colors if sambe else sim_colors

    for i, state in enumerate(simulator.resonances):
        name = simulator.get_state_name(state).upper()

        label = f"State {name}"
        if sambe:
            label += " theo."

        log_scale = max(r.expect[i]) < 3e-1
        if log_scale:
            label += " (log.)"

        ax = ax_log if log_scale else ax_normal

        ax.plot(
            tlist,
            r.expect[i],
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            color=colors[i],
            alpha=alpha,
        )

if __name__ == "__main__":
    simulator = Simulator()
    simulator.config(config)
    
    # Build e_ops based on config
    labels = []
    wd = config['frequency'] * 2 * np.pi
    amp = config['amplitude'] * 2 * np.pi

    print("amp: ", amp, " freq: ", wd/(2*np.pi), " resonance: ", simulator.res_freq_static)
    tg = config['tgate']
    psi0 = None
    if config['initial_state']=='a':
        psi0 = simulator.state_a
    elif config['initial_state']=='b':
        psi0 = simulator.state_b
    # Compute population
    result, tlist = simulate_sambe(simulator, wd, amp, psi0, tg)
    if(config['show_normal']):
        result_normal, tlist = simulate_normal(simulator, wd, amp, psi0, tg)
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    # Separate c-population from other populations
    c_index = None
    ax_log = ax.twinx()

    if(config['show_normal']): show_result(simulator, result_normal, tlist, ax, ax_log, sambe=False) 
    show_result(simulator, result, tlist, ax, ax_log)
    # Create title with parameters
    title = f'Population Dynamics\n'
    title += f'Amplitude: {config["amplitude"]} GHz, '
    title += f'Frequency: {config["frequency"]} GHz, '
    title += f'Gate Time: {tg} ns, '
    title += f'Initial State: {config['initial_state'].upper()}'
    
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax_log.set_ylabel('Population (log scale)', fontsize=12)
    ax_log.set_ylim([1e-5, 1.0])
    ax_log.set_yscale('log')

    # Combine legends from both axes
    lines_ax, labels_ax = ax.get_legend_handles_labels()
    lines_c, labels_c = ax_log.get_legend_handles_labels()
    ax.legend(lines_ax + lines_c, labels_ax + labels_c, fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.show()