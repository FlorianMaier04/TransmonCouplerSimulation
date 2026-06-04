from qutip import *
import numpy as np
from tqdm import tqdm
from qutip import *

def fourier_coeffs(f, wd, N=100, M=10000):
    T = 2*np.pi / wd
    t = np.linspace(0, T, M, endpoint=False)
    dt = t[1] - t[0]
    ft = f(t)  # shape: (M, D, D)
    coeffs = []
    for n in range(0, N):
        # Berechne V_n = (1/T) * integral_0^T V(t) * exp(-i*n*wd*t) dt
        exp_term = np.exp(-1j*n*wd*t)  # shape: (M,)
        integrand = ft * exp_term[:, None, None]  # shape: (M, D, D)
        coeff = (1/T) * np.sum(integrand, axis=0) * dt  # shape: (D, D)
        coeffs.append(coeff.real) # we only insert real pulse shapes
    return np.array(coeffs)

def sigma_x_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return ej*ei.dag() + ei*ej.dag()

def sigma_y_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return -1j*ei*ej.dag() + 1j*ej*ei.dag()

def extract_fidelity(lh, tlist, sd=3, sim=None):
    """
    Extract fidelity for iSWAP-like gate.
    
    Parameters:
    -----------
    lh : list or Qobj
        Hamiltonian for time evolution
    tlist : array
        Time list for evolution
    sd : int
        System dimension (default: 3)
    sim : Simulation object, optional
        If provided, uses full 27D Hilbert space with states from sim.state_a, sim.state_b, sim.state_c
    
    Returns:
    --------
    float : Process fidelity
    """
    U_ideal = np.array([[0, 1j, 0], [1j, 0, 0], [0, 0, 1]], dtype=complex)
    U_subsys = np.zeros((sd, sd), dtype=complex)
    
    if sim is None:
        # Original 3D implementation
        for col_idx in range(0, sd): 
            phi0 = basis(sd, col_idx)
            result = mesolve(lh, phi0, tlist, e_ops=[])
            final_state = result.states[-1]
            for row_idx in range(0, sd):
                U_subsys[row_idx, col_idx] = final_state.full()[row_idx, 0]
    else:
        # Full 27D implementation using state_a, state_b, state_c
        state_indices = [sim.state_a, sim.state_b, sim.state_c]
        initial_states = [sim.E_states[idx] for idx in state_indices]
        
        for col_idx, phi0_full in enumerate(initial_states):
            result = mesolve(lh, phi0_full, tlist, e_ops=[])
            final_state = result.states[-1]
            
            # Project onto the 3D subspace spanned by state_a, state_b, state_c
            for row_idx, target_state_idx in enumerate(state_indices):
                target_state = sim.E_states[target_state_idx]
                amplitude = (target_state.dag() @ final_state).tr()
                U_subsys[row_idx, col_idx] = amplitude
    
    fidelity = qutip.process_fidelity(Qobj(U_subsys), Qobj(U_ideal))
    return fidelity

def extract_pop_fid(lh, tlist, plot=False, sim=None):
    """
    Extract population fidelity by checking if population is exchanged for important initial states.
    For iSWAP-like behavior: |0⟩ ↔ |1⟩ (lowest two states)
    
    Parameters:
    -----------
    sim : Simulation object, optional
        If provided, uses full 27D Hilbert space with states from sim.state_a, sim.state_b, sim.state_c    
    Returns:
    --------
    dict : Contains population exchanges for each initial state
        - 'pop_0_to_1': Population in state 1 after starting from state 0
        - 'pop_1_to_0': Population in state 0 after starting from state 1
        - 'pop_2_final': Population in state 2 for additional test case (if sd > 2)
        - 'iswap_fidelity': Average fidelity of population exchange
    """
    results = {}
    sd = 3
    if sim is None:
        # Original 3D implementation
        initial_states = {
            'state_0': basis(sd, 0),  # |0⟩
            'state_1': basis(sd, 1),  # |1⟩
            'state_2': basis(sd, 2)
        }
        
        items = initial_states.items() if not plot else tqdm(initial_states.items())
        for state_name, initial_state in items:
            result = mesolve(lh, initial_state, tlist, e_ops=[])
            final_state = result.states[-1]
            final_state_vec = final_state.full().flatten()
            populations = np.abs(final_state_vec)**2
            
            results[f'initial_{state_name}'] = state_name
            for i in range(min(sd, len(populations))):
                results[f'pop_{i}_{state_name}'] = populations[i]
    else:
        # Full 27D implementation using state_a, state_b, state_c
        state_indices = [sim.state_a, sim.state_b, sim.state_c]
        state_names = ['state_a', 'state_b', 'state_c']
        initial_states = {
            state_names[i]: sim.E_states[idx] for i, idx in enumerate(state_indices)
        }
        
        items = initial_states.items() if not plot else tqdm(initial_states.items())
        for state_name, initial_state_full in items:
            result = mesolve(lh, initial_state_full, tlist, e_ops=[])
            final_state = result.states[-1]
            
            # Project onto the 3D subspace spanned by state_a, state_b, state_c
            results[f'initial_{state_name}'] = state_name
            for i, target_state_idx in enumerate(state_indices):
                target_state = sim.E_states[target_state_idx]
                amplitude = (target_state.dag() @ final_state).tr()
                population = np.abs(amplitude)**2
                results[f'pop_{i}_{state_name}'] = population
    
    # Check iSWAP exchange conditions for the two lowest states
    pop_0_to_1 = results.get('pop_1_state_0', 0)  # Should be close to 1 for perfect iSWAP
    pop_1_to_0 = results.get('pop_0_state_1', 0)  # Should be close to 1 for perfect iSWAP
    
    # Additional test case check (if sd > 2): state |2⟩ should remain in |2⟩
    pop_2_final = results.get('pop_2_state_2', 0) if sd > 2 else None
    
    # Calculate fidelity based on iSWAP conditions
    iswap_fidelity = (pop_0_to_1 + pop_1_to_0) / 2.0
    
    # Store results
    results['pop_0_to_1'] = pop_0_to_1
    results['pop_1_to_0'] = pop_1_to_0
    if pop_2_final is not None:
        results['pop_2_final'] = pop_2_final
    results['iswap_fidelity'] = iswap_fidelity
    
    return results


# heff fitting

import numpy as np

def fit_heff_element(s, i, j,
                     wd_range, amp_real_range, amp_imag_range,
                     n_wd=100, n_amp_real=100, n_amp_imag=100):
    si = s.get_associated_index(i)
    sj = s.get_associated_index(j)
    wd_vals = np.linspace(*wd_range, n_wd)
    amp_real_vals = np.linspace(*amp_real_range, n_amp_real)
    amp_imag_vals = np.linspace(*amp_imag_range, n_amp_imag)
    X = []
    Y = []
    for wd in tqdm(wd_vals):
        for amp_real in amp_real_vals:
            for amp_imag in amp_imag_vals:
                amp = amp_real + 1j * amp_imag
                h = s.heff_element(si, sj, wd, amp)
                X.append([
                    1,
                    wd,
                    amp_real,
                    amp_imag,
                    wd**2,
                    amp_real**2,
                    amp_imag**2,
                    wd*amp_real,
                    wd*amp_imag,
                    amp_real*amp_imag
                ])
                Y.append(h)
    X = np.asarray(X)
    Y = np.asarray(Y)
    coeffs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return coeffs