import sys
from helper_function import *
from utils import *
from qutip import *
import numpy as np
import matplotlib.pyplot as plt


config = {
    'show_direct': True,
    'plot_epsilon': False,  # if True, plot epsilon_x and epsilon_y against t and exit
    'initial_state': 'a',  # 'a' or 'b' for initial state selection
    'amplitude': 0.23440161220302155,  # in GHz
    'detune': 0.0, # in GHz # 223 # 0.00389 # 00145
    'freq': 0.7061212073258497, 
    'tgate': 200, # in ns
}

def simulate_direct(simulator, wd, amp, psi0, tlist):
    psi0_state = simulator.E_states[psi0]
    cos_amp, cos_wd = compute_cos_params(simulator, config['tgate']) 
    # ops = [simulator.E_states[state] @ simulator.E_states[state].dag() for state in simulator.resonances.keys()]
    result = mesolve([simulator.H0_dressed, [simulator.V1_dressed, lambda t,args: cos_amp*np.cos(cos_wd*t)]], 
                     psi0_state,tlist, c_ops=[], e_ops=[])
    return result

def simulate_sambe(s, wd, amp, psi0):
    heff = (s.heff(wd, amp))#
    print("heff")
    display(Qobj(heff))
    # find the optimal gate time
    tg = config['tgate']
    tlist = np.arange(0, tg, 0.01)
    phi0, ops = compute_phi0(s, wd, amp, psi0)
    time_transformed_h = compute_t_dependency(heff, tg)
    result = mesolve(
        time_transformed_h,
        phi0,
        tlist,
        c_ops=[],e_ops=[],)
    return result, tlist

def show_result(simulator, result, tlist, ax_normal, ax_log, sambe=True):
    linestyle = 'dashed' if sambe else '-'  
    n_states = len(simulator.resonances)
    theory_colors = plt.cm.tab10(np.linspace(0, 1, n_states))
    sim_colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    alpha = 1.0 if sambe else 0.4
    linewidth = 2.5 if sambe else 2

    colors = theory_colors if sambe else sim_colors
    final_state = result.states[-1].full()
    print("final_state: ", final_state)
    for i, state in enumerate(simulator.resonances):
        name = simulator.get_state_name(state).upper()
        label = f"State {name}"
        if sambe:
            label += " theo."
        pop = [abs(simulator.E_states[state].dag() * result.states[idx_t])**2 
               for idx_t in range(len(result.states))]
        log_scale = max(pop) < 3e-1
        if log_scale:
            label += " (log.)"
        ax = ax_log if log_scale else ax_normal
        ax.plot(
            tlist,
            pop,
            label=label,
            linewidth=linewidth,
            linestyle=linestyle,
            color=colors[i],
            alpha=alpha,
        )

if __name__ == "__main__":
    simulator = Simulation()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax_log = ax.twinx()

    amp = config['amplitude'] * 2 * np.pi
    detune = config['detune'] * 2 * np.pi
    detune = 0
    wd = config['freq'] * 2 * np.pi
    psi0 = None
    if config['initial_state']=='a':
        psi0 = simulator.state_a
    elif config['initial_state']=='b':
        psi0 = simulator.state_b

    tlist = np.linspace(0, config['tgate'], 4000)
    # result, tlist = simulate_sambe(simulator, wd, amp, psi0)
    if(config['show_direct']): 
        result_normal = simulate_direct(simulator, wd, amp, psi0, tlist)
        show_result(simulator, result_normal, tlist, ax, ax_log, sambe=False) 
    # show_result(simulator, result, tlist, ax, ax_log)
    # Create title with parameters
    title = f'Population Dynamics\n'
    title += f'Amplitude: {config["amplitude"]} GHz, '
    title += f'Frequency: {wd/(2*np.pi):2f} GHz, '
    title += f'Gate Time: {tlist[-1]} ns, '
    title += f'Initial State: {config['initial_state'].upper()}'
    
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    ax_log.set_ylabel('Population (log scale)', fontsize=12)
    ax_log.set_ylim([1e-6, 1.0])
    ax_log.set_yscale('log')

    # Combine legends from both axes
    lines_ax, labels_ax = ax.get_legend_handles_labels()
    lines_c, labels_c = ax_log.get_legend_handles_labels()
    ax.legend(lines_ax + lines_c, labels_ax + labels_c, fontsize=11, loc='lower center')
    
    plt.tight_layout()
    plt.show()