from qutip import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from qutip import *
from helper_function import * 
from Floquet_perturbation_theory import * 
from scipy.optimize import root_scalar

def sigma_x_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return ej*ei.dag() + ei*ej.dag()

def sigma_y_ij(i, j, d):
    ei = basis(d, i)
    ej = basis(d, j)
    return -1j*ei*ej.dag() + 1j*ej*ei.dag()

def extract_fidelity(lh, tlist, sd=3, s=None, plot=False, debug=False):
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
    if s is None:
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
        state_indices = [s.state_a, s.state_b, s.state_c]
        initial_states = [s.E_states[idx] for idx in state_indices]
        enum = enumerate(initial_states) if not plot else tqdm(enumerate(initial_states))
        for col_idx, phi0_full in enumerate(initial_states):
            result = mesolve(lh, phi0_full, tlist, e_ops=[], options=options)
            final_state = result.states[-1]
            
            # Project onto the 3D subspace spanned by state_a, state_b, state_c
            for row_idx, target_state_idx in enumerate(state_indices):
                target_state = s.E_states[target_state_idx]
                amplitude = target_state.overlap(final_state)
                U_subsys[row_idx, col_idx] = amplitude
    # print("U_subsys")
    # print_matrix(U_subsys)
    fidelity = qutip.process_fidelity(Qobj(U_subsys), Qobj(U_ideal))
    return fidelity

def resonant_subspace_column_evolution(lh, tlist, j, debug=False, s=None):
    """Compute the time evolution of the j-th column of the resonant 3x3 subspace unitary."""
    sd = 3
    if j not in (0, 1, 2):
        raise ValueError(f"Column index j must be 0, 1, or 2, got {j}.")

    options = {"progress_bar": "tqdm", "nsteps":100000} if debug else None
    if s is None:
        resonant_states = [basis(sd, 0), basis(sd, 1), basis(sd, 2)]
        initial_state = resonant_states[j]
    else:
        state_indices = [s.state_a, s.state_b, s.state_c]
        resonant_states = [s.E_states[idx] for idx in state_indices]
        initial_state = resonant_states[j]

    result = mesolve(lh, initial_state, tlist, e_ops=[], options=options)
    amplitudes = np.array([
        [state.overlap(psi_t) for psi_t in result.states]
        for state in resonant_states
    ])
    return amplitudes

def extract_pop_fid(lh, tlist, plot=False, debug=False, plot_c=False, s=None, cut=False):
    """
    Extract populations and fidelity from simulations.
    
    Parameters:
    -----------
    lh : list or Qobj
        Hamiltonian for time evolution
    tlist : array
        Time list for evolution
    plot : bool
        Whether to plot populations over time
    debug : bool
        Whether to show progress bar
    plot_c : bool
        Whether to include state c population in plots (with log scale on secondary axis)
    s : Simulation object, optional
        For 27D Hilbert space; if None uses 3D implementation
    """
    results = {}
    pop_a, pop_b, pop_c = {}, {}, {}

    amp_col_a = resonant_subspace_column_evolution(lh, tlist, 0, debug=debug, s=s)
    pop_a['state_a'] = np.abs(amp_col_a[0])**2
    pop_b['state_a'] = np.abs(amp_col_a[1])**2
    pop_c['state_a'] = np.abs(amp_col_a[2])**2
    if not cut:
        amp_col_b = resonant_subspace_column_evolution(lh, tlist, 1, debug=debug, s=s)
        pop_a['state_b'] = np.abs(amp_col_b[0])**2
        pop_b['state_b'] = np.abs(amp_col_b[1])**2
        pop_c['state_b'] = np.abs(amp_col_b[2])**2

    if not cut:
        results['iswap_fidelity'] = (pop_a['state_b'][-1] + pop_b['state_a'][-1]) / 2.0

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        alpha_ab = 0.3 if plot_c else 1.0
        for ax_idx, state_name in enumerate(['state_a', 'state_b']):
            if cut and ax_idx==1:
                continue
            ax = axes[ax_idx]
            ax.grid(True, alpha=0.3)
            ax.plot(tlist, pop_a[state_name], 'b-', linewidth=2, alpha=alpha_ab, label='Population in state a')
            ax.plot(tlist, pop_b[state_name], 'r-', linewidth=2, alpha=alpha_ab, label='Population in state b')
            ax.set_xlabel('Time')
            ax.set_ylabel('Population')
            ax.set_title(f'Initial state: |{state_name.split("_")[1]}⟩')
            if plot_c:
                ax2 = ax.twinx()
                c_style = '-' if state_name == 'state_a' else '--'
                ax2.semilogy(tlist, pop_c[state_name], f'g{c_style}', linewidth=2, label='Population in state c')
                ax2.set_ylabel('Population in state c (log scale)', color='g')
                ax2.tick_params(axis='y', labelcolor='g')
                ax2.set_ylim([1e-20, 1e-2])
                lines = ax.get_lines() + ax2.get_lines()
            else:
                lines = ax.get_lines()
            ax.legend(lines, [l.get_label() for l in lines], loc='best')    
        plt.tight_layout()
        plt.show()

    return results


# heff fitting

import numpy as np

def fit_heff_element(s, i, j, wd_range, amp_real_range, amp_imag_range,
                     n_wd=100, n_amp_real=100, n_amp_imag=100, cubic=False):
    si = s.get_associated_index(i)
    sj = s.get_associated_index(j)
    wd_vals = np.linspace(*wd_range, n_wd)
    amp_real_vals = np.linspace(*amp_real_range, n_amp_real)
    amp_imag_vals = np.linspace(*amp_imag_range, n_amp_imag)

    X, Y = [], []

    for wd in tqdm(wd_vals):
        for amp_real in amp_real_vals:
            for amp_imag in amp_imag_vals:
                amp = amp_real + 1j*amp_imag
                h = s.heff_element(si, sj, wd, amp)
                row = [
                    1,
                    wd, amp_real, amp_imag,
                    wd**2, amp_real**2, amp_imag**2,
                    wd*amp_real, wd*amp_imag, amp_real*amp_imag
                ]
                if cubic:
                    row.extend([
                        wd**3, amp_real**3, amp_imag**3,
                        wd**2*amp_real, wd**2*amp_imag,
                        wd*amp_real**2, wd*amp_imag**2,
                        amp_real**2*amp_imag, amp_real*amp_imag**2,
                        wd*amp_real*amp_imag
                    ])

                X.append(row)
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