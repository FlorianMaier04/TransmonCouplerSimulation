import sys
from helper_function import *
from utils import *
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

config = {
    'initial_state': 'b',
    'amplitude_range': np.linspace(0.05, 0.4, 15),
    'detune_range': np.linspace(-0.002, 0.02, 15),
}

def get_fidelity(s, wd, amp, psi0, target_state):
    heff = s.heff(wd, amp)
    Omega_ab = heff[0][1]
    
    try:
        tg, _, _ = find_optimal_time(wd, amp, s.order,
            s.state_b, s.state_a, s.state_c,
            s.E_array, s.V1_dressed_array, int(np.pi/(2*np.abs(Omega_ab))),)
    except:
        return np.nan
    
    tlist = np.arange(0, tg, 0.01)    
    time_transformed_h = compute_t_dependency(s, heff, wd, tg)    
    fidelity = s.extract_fidelity(time_transformed_h, tlist)  

    return fidelity

if __name__ == "__main__":
    simulator = Simulation()
    
    psi0 = simulator.state_b if config['initial_state'] == 'b' else simulator.state_a
    target_state = simulator.state_a if config['initial_state'] == 'b' else simulator.state_b
    
    amp_gHz = config['amplitude_range']
    detune_gHz = config['detune_range']
    
    fidelity_map = np.zeros((len(detune_gHz), len(amp_gHz)))
    
    print(f"Sweeping {len(amp_gHz)} amplitudes × {len(detune_gHz)} detunings...")
    
    for i, amp_val in enumerate(tqdm(amp_gHz, desc="Amplitude")):
        amp = amp_val * 2 * np.pi
        wd_res = simulator.find_resonance(amp)
        
        for j, detune_val in enumerate(detune_gHz):
            detune = detune_val * 2 * np.pi
            wd = wd_res + detune
            
            fidelity = get_fidelity(simulator, wd, amp, psi0, target_state)
            fidelity_map[j, i] = fidelity if not np.isnan(fidelity) else 0.0
    
    infidelity_map = 1 - fidelity_map
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(infidelity_map, cmap='viridis_r', aspect='auto', origin='lower',
                   extent=[amp_gHz[0], amp_gHz[-1], detune_gHz[0], detune_gHz[-1]],
                   norm=plt.matplotlib.colors.LogNorm(vmin=np.nanmin(infidelity_map[infidelity_map > 0]) or 1e-4, 
                                                       vmax=np.nanmax(infidelity_map)))
    
    ax.set_xlabel('Amplitude (GHz)', fontsize=12)
    ax.set_ylabel('Frequency Detuning (GHz)', fontsize=12)
    ax.set_title('Gate Infidelity (1 - Fidelity)', fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='1 - Fidelity (log scale)')
    
    plt.tight_layout()
    plt.show()
