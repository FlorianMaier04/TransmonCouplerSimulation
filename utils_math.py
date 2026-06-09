from qutip import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from qutip import *
from helper_function import * 
from Floquet_perturbation_theory import * 
from scipy.optimize import root_scalar

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

def extract_fidelity(lh, tlist, sd=3, sim=None, plot=False, debug=False):
    """
    lh : list or Qobj
        Hamiltonian for time evolution
    tlist : array
        Time list for evolution
    sd : int
        System dimension (default: 3)
    sim : Simulation object, optional
        If provided, uses full 27D Hilbert space with states from sim.state_a, sim.state_b, sim.state_c
    """
    U_ideal = np.array([[0, 1j, 0], [1j, 0, 0], [0, 0, 1]], dtype=complex)
    U_subsys = np.zeros((sd, sd), dtype=complex)
    options={"progress_bar": "tqdm"} if debug else None
    if sim is None:
        # Original 3D implementation
        enum = range(0,sd) if not plot else tqdm(range(0,sd))
        for col_idx in enum: 
            phi0 = basis(sd, col_idx)
            result = mesolve(lh, phi0, tlist, e_ops=[], options=options)
            final_state = result.states[-1]
            for row_idx in range(0, sd):
                U_subsys[row_idx, col_idx] = final_state.full()[row_idx, 0]
    else:
        # Full 27D implementation using state_a, state_b, state_c
        state_indices = [sim.state_a, sim.state_b, sim.state_c]
        initial_states = [sim.E_states[idx] for idx in state_indices]
        enum = enumerate(initial_states) if not plot else tqdm(enumerate(initial_states))
        for col_idx, phi0_full in enumerate(initial_states):
            result = mesolve(lh, phi0_full, tlist, e_ops=[], options=options)
            final_state = result.states[-1]
            
            # Project onto the 3D subspace spanned by state_a, state_b, state_c
            for row_idx, target_state_idx in enumerate(state_indices):
                target_state = sim.E_states[target_state_idx]
                amplitude = target_state.overlap(final_state)
                U_subsys[row_idx, col_idx] = amplitude
    # print("U_subsys")
    # print_matrix(U_subsys)
    fidelity = qutip.process_fidelity(Qobj(U_subsys), Qobj(U_ideal))
    return fidelity

def extract_pop_fid(lh, tlist, plot=False, sim=None, debug=False):
    """
    Extract population fidelity by checking if population is exchanged for important initial states.
    For iSWAP-like behavior: |0⟩ ↔ |1⟩ (lowest two states)
    
    Parameters:
    -----------
    sim : Simulation object, optional
        If provided, uses full 27D Hilbert space with states from sim.state_a, sim.state_b, sim.state_c
    plot : bool, optional
        If True, plots the population of states 0 and 1 vs time
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
    options={"progress_bar": "tqdm"} if debug else None
    if sim is None:
        # Original 3D implementation
        states = {
            'state_0': basis(sd, 0),  # |0⟩
            'state_1': basis(sd, 1),  # |1⟩
        }
        
        pop_0_timeseries = {}
        pop_1_timeseries = {}
        items = states.items()
        for state_name, initial_state in items:
            if plot:
                # Compute population expectation values at all times
                result = mesolve(lh, initial_state, tlist, e_ops=[], options=options)
                final_state = result.states[-1]
                pop_0_timeseries[state_name] = [np.abs(states['state_0'].overlap(result.states[i]))**2
                                                for i in range(len(tlist))]
                pop_1_timeseries[state_name] = [np.abs(states['state_1'].overlap(result.states[i]))**2
                                                for i in range(len(tlist))]
            else:
                # Only compute final state
                result = mesolve(lh, initial_state, tlist, e_ops=[], options=options)
                final_state = result.states[-1]
            
            final_state_vec = final_state.full().flatten()
            populations = np.abs(final_state_vec)**2
            
            results[f'initial_{state_name}'] = state_name
            for i in range(min(sd, len(populations))):
                results[f'pop_{i}_{state_name}'] = populations[i]
    else:
        # Full 27D implementation using state_a, state_b, state_c
        states = {
            'state_0':sim.state_a, 'state_1':sim.state_b
        }
        pop_0_timeseries = {}
        pop_1_timeseries = {}
        items = states.items()
        for state_name, in_state in items:
            if plot:
                # For 27D system, we need to project the expectation value
                result = mesolve(lh, sim.E_states[in_state], tlist, e_ops=[], options=options)
                
                # Calculate projection onto states 0 and 1 for each time step
                pop_0_list = []
                pop_1_list = []
                for state_t in result.states:
                    amplitude_0 = sim.E_states[states['state_0']].overlap(state_t)
                    amplitude_1 = sim.E_states[states['state_1']].overlap(state_t)
                    pop_0_list.append(np.abs(amplitude_0)**2)
                    pop_1_list.append(np.abs(amplitude_1)**2)
                pop_0_timeseries[state_name] = np.array(pop_0_list)
                pop_1_timeseries[state_name] = np.array(pop_1_list)
                final_state = result.states[-1]
            else:
                result = mesolve(lh, sim.E_states[in_state], tlist, e_ops=[], options=options)
                final_state = result.states[-1]
            
            # Project onto the 3D subspace spanned by state_a, state_b, state_c
            results[f'initial_{state_name}'] = state_name
            for i, target_state_idx in enumerate(states.values()):
                target_state = sim.E_states[target_state_idx]
                amplitude = target_state.overlap(final_state)
                population = np.abs(amplitude)**2
                results[f'pop_{i}_{state_name}'] = population
    
    # Check iSWAP exchange conditions for the two lowest states
    pop_0_to_1 = results.get('pop_1_state_0', 0)  # Should be close to 1 for perfect iSWAP
    pop_1_to_0 = results.get('pop_0_state_1', 0)  # Should be close to 1 for perfect iSWAP
    iswap_fidelity = (pop_0_to_1 + pop_1_to_0) / 2.0
    results['iswap_fidelity'] = iswap_fidelity
    
    # Plot if requested
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot populations for state_0 initial state
        axes[0].plot(tlist, pop_0_timeseries['state_0'], 'b-', linewidth=2, label='Population in state 0')
        axes[0].plot(tlist, pop_1_timeseries['state_0'], 'r-', linewidth=2, label='Population in state 1')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('Population')
        axes[0].set_title('Initial state: |0⟩')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot populations for state_1 initial state
        axes[1].plot(tlist, pop_0_timeseries['state_1'], 'b-', linewidth=2, label='Population in state 0')
        axes[1].plot(tlist, pop_1_timeseries['state_1'], 'r-', linewidth=2, label='Population in state 1')
        axes[1].set_xlabel('Time')
        axes[1].set_ylabel('Population')
        axes[1].set_title('Initial state: |1⟩')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
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





def print_matrix(A, d=3, th=1e-4):
    f = lambda x: "0" if abs(x) < th else (f"{x:.{d}e}" if abs(x) < 10**(-d) or abs(x) >= 1e4 else f"{x:.{d}f}")
    s = lambda z: (
        f(z.real) if abs(z.imag) < th else
        f"{f(z.imag)}j" if abs(z.real) < th else
        f"{f(z.real)} {'+' if z.imag >= 0 else '-'} {f(abs(z.imag))}j"
    )
    rows = [[s(z) for z in row] for row in A]
    w = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    print('\n'.join('[ ' + '  '.join(x.rjust(wi) for x, wi in zip(r, w)) + ' ]' for r in rows))

def find_optimal_amplitude(s, tg, amp_guess=0.3):
    """
    For a given gate-tie, find the optimal cos amplitude to optimaze gate fidelity with iSWAP gate.

    Parameters:
    -----------
    fre : float
        Drive frequency
    amp : float
        Drive amplitude
    Returns:
    --------
    optimal_amp : complex
    optimal_fidelity : float
        Maximum achieved fidelity with iSWAP gate
    optimal_U : array (3x3)
        Subsystem unitary at optimal time
    """
    la = np.arange(0.2, 1.5 * amp_guess, 0.001) * 2*np.pi
    E0 = s.E_array
    V1 = s.V1_dressed_array
    # Define initial states for subsystem basis {i, f, l}
    initial_state_a = np.zeros(len(E0))
    initial_state_a[s.state_a] = 1
    initial_state_b = np.zeros(len(E0))
    initial_state_b[s.state_b] = 1
    initial_state_c = np.zeros(len(E0))
    initial_state_c[s.state_c] = 1
    initial_state_list = [initial_state_a, initial_state_b, initial_state_c]

    # Define ideal iSWAP gate for subsystem {i, f, l}
    # iSWAP: |i⟩→i|f⟩, |f⟩→i|i⟩, |l⟩→|l⟩ (spectator)
    U_ideal = np.array([[0, 1j, 0],
                        [1j, 0, 0],
                        [0, 0, 1]], dtype=complex)
    tlist = np.arange(0, tg, 0.01)
    fidelities = np.zeros_like(la)
    lU = np.zeros((len(la), 3, 3), dtype=complex)
    # Evolve each basis state and construct unitary matrix
    for k, amp in enumerate(la):
        # Storage for time-evolved unitaries
        for j, initial_state in enumerate(initial_state_list):
            psi_t, _ = Psi_t_FloquetPerturb(
                s.order, 0, s.find_resonance(amp), amp,
                s.resonances,
                E0, V1, initial_state, tlist,
            )
            # Extract subsystem components at each time
            # psi_t[k, :] is the evolved state at time tlist[k]
            lU[k, 0, j] = psi_t[s.state_a, -1]  # Project onto state i
            lU[k, 1, j] = psi_t[s.state_b, -1]  # Project onto state f
            lU[k, 2, j] = psi_t[s.state_c, -1]  # Project onto state l
            trace_overlap = np.trace(U_ideal.conj().T @ lU[k])
            fidelities[k] = np.abs(trace_overlap)**2 / 3**2
            # print("fidelity: ", np.abs(trace_overlap)**2/9)
    max_idx = np.argmax(fidelities)
    max_fidelity = fidelities[max_idx]
    amp_max = la[max_idx]
    U_max = lU[max_idx]
    return amp_max, max_fidelity, U_max

from scipy.optimize import fsolve

def find_amplitude_for_gate_time(s, target_time):
    def objective(amp):
        amp = np.asarray(amp).item()
        wd = s.find_resonance(amp)
        rabi_rate = s.heff_element(s.state_a, s.state_b, wd, amp)
        tguess = int(2*np.pi/np.abs(rabi_rate))/4
        gate_time, optimal_f, _, _, _ = find_optimal_time(
            wd, amp, 3,
            s.state_a, s.state_b, s.state_c,
            s.E_array, s.V1_dressed_array,
            tguess
        )
        return gate_time - target_time

    res = root_scalar(objective,bracket=[0.001*2*np.pi, 0.3*2*np.pi],method="brentq")
    print("target: ", target_time, " real: ", objective(res.root))

    return res.root
