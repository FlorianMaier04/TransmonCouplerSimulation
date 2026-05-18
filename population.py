
from helper_function import *
from utils import Simulator
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

config = {
    'showA': True, 
    'showB': True, 
    'showC': True,
    'initial_state': 'a',  # 'a' or 'b' for initial state selection
    'amplitude': 1,  # in GHz
    'frequency': 0.706,  # in GHz
    'order': 3,
    'tgate': 200,  # in ns
    'pulse_shape': 'cos',  # 'cos', 'cossin', or 'gauss'
    'pulse_args': [0.5],  # for gauss: [sigma_fraction]
}

def compute_population(simulator, wd, amp, e_ops, initial_state, tg=200):
    """Simulate population dynamics using mesolve."""
    # TODO: check Hermicity, Complex description and Dimension
    heff = Qobj(simulator.heff(wd, amp))
    
    tlist = np.linspace(0, tg, 1000)
    
    # Set initial state based on config
    if initial_state == 'a':
        psi0 = basis(3, 0)  # state |a>
    else:  # 'b'
        psi0 = basis(3, 1)  # state |b>
    display(heff)
    # Simulate using mesolve
    result = mesolve(
        [heff],
        psi0,
        tlist,
        [],
        e_ops=e_ops,
    )
    
    return result, tlist

if __name__ == "__main__":
    simulator = Simulator()
    simulator.config(config)
    
    # Build e_ops based on config
    e_ops = []
    labels = []
    from qutip import basis

    a = basis(3, 0)
    b = basis(3, 1)
    c = basis(3, 2)
    P_a = a * a.dag()
    P_b = b * b.dag()
    P_c = c * c.dag()
    if config['showA']:
        e_ops.append(P_a)
        labels.append('State A')
    if config['showB']:
        e_ops.append(P_b)
        labels.append('State B')
    if config['showC']:
        e_ops.append(P_c)
        labels.append('State C')
    
    # Extract parameters from config
    wd = config['frequency'] * 2 * np.pi
    amp = config['amplitude'] * 2 * np.pi
    print("amp: ", amp, " freq: ", wd, " resonance: ", simulator.res_freq_static)
    tg = config['tgate']
    initial_state = config['initial_state']
    # Compute population
    result, tlist = compute_population(simulator, wd, amp, e_ops, initial_state, tg)
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, label in enumerate(labels):
        ax.plot(tlist, result.expect[i], label=label, linewidth=2)
        # print("expect: ", result.expect[i])
        
    # Create title with parameters
    title = f'Population Dynamics\n'
    title += f'Amplitude: {config["amplitude"]} GHz, '
    title += f'Frequency: {config["frequency"]} GHz, '
    title += f'Gate Time: {tg} ns, '
    title += f'Initial State: {initial_state.upper()}'
    
    ax.set_xlabel('Time (ns)', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.show()