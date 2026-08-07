from qutip import *
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from qutip import *
from helper_function import * 
from Floquet_perturbation_theory import * 
from scipy.optimize import root_scalar
import string

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
    options = {"progress_bar": "tqdm", "nsteps":100000} if debug else None

    if s is None:
        resonant_states = [basis(3, i) for i in range(3)]
    else:
        # Order resonant state indices by their assigned order/value
        state_indices = [k for k, _ in sorted(s.resonances.items(), key=lambda x: x[1])]

        # Try to determine the Hamiltonian dimension by probing the provided `lh`.
        Hdim = None
        try:
            # `lh` may be a time-dependent list like [H_func] or a callable, or a Qobj
            if isinstance(lh, (list, tuple)):
                h0 = lh[0](tlist[0]) if callable(lh[0]) else lh[0]
            elif callable(lh):
                h0 = lh(tlist[0])
            else:
                h0 = lh

            # Qobj or ndarray: get first dimension
            Hdim = h0.shape[0] if hasattr(h0, 'shape') else None
        except Exception:
            Hdim = None

        # If the Hamiltonian lives in the reduced resonant subspace (Hdim == number of resonant states)
        # then use basis vectors of that reduced space. Otherwise fall back to full eigenstates.
        if Hdim is not None and Hdim == len(state_indices):
            resonant_states = [basis(Hdim, idx) for idx in range(Hdim)]
        else:
            resonant_states = [s.E_states[k] for k in state_indices]

    if j >= len(resonant_states):
        raise ValueError(f"Column index j={j} out of range.")

    result = mesolve(lh, resonant_states[j], tlist, e_ops=[], options=options)

    return np.array([
        [state.overlap(psi_t) for psi_t in result.states]
        for state in resonant_states
    ])

def extract_pop_fid(lh, tlist, plot=False, debug=False, plot_c=False, s=None, cut=False):

    results = {}

    labels = list(string.ascii_lowercase[:len(s.resonances)]) if s is not None else ["a", "b", "c"]
    pops = {l: {} for l in labels}

    amp_col_a = resonant_subspace_column_evolution(lh, tlist, 0, debug=debug, s=s)
    for i, l in enumerate(labels):
        pops[l]["state_a"] = np.abs(amp_col_a[i])**2

    if not cut:
        amp_col_b = resonant_subspace_column_evolution(lh, tlist, 1, debug=debug, s=s)
        for i, l in enumerate(labels):
            pops[l]["state_b"] = np.abs(amp_col_b[i])**2

    if not cut:
        results["iswap_fidelity"] = (pops["a"]["state_b"][-1] + pops["b"]["state_a"][-1]) / 2.0
    else:
        results["iswap_fidelity"] = pops["b"]["state_a"][-1]

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        alpha_ab = 0.3 if plot_c else 1.0

        for ax_idx, state_name in enumerate(["state_a", "state_b"]):
            if cut and ax_idx == 1:
                continue

            ax = axes[ax_idx]
            ax.grid(True, alpha=0.3)

            # alle Populationen außer der letzten (wenn plot_c=True)
            for l in labels[:-1] if plot_c else labels:
                ax.plot(
                    tlist,
                    pops[l][state_name],
                    linewidth=2,
                    alpha=alpha_ab,
                    label=f"Population in state {l}",
                )

            if plot_c:
                ax2 = ax.twinx()
                c_style = "-" if state_name == "state_a" else "--"
                ax2.semilogy(
                    tlist,
                    pops[labels[-1]][state_name],
                    f"g{c_style}",
                    linewidth=2,
                    label=f"Population in state {labels[-1]}",
                )
                ax2.set_ylabel(f"Population in state {labels[-1]} (log scale)", color="g")
                ax2.tick_params(axis="y", labelcolor="g")
                ax2.set_ylim([1e-20, 1e-2])
                lines = ax.get_lines() + ax2.get_lines()
            else:
                lines = ax.get_lines()

            ax.set_xlabel("Time")
            ax.set_ylabel("Population")
            ax.set_title(f"Initial state: |{state_name.split('_')[1]}⟩")
            ax.legend(lines, [l.get_label() for l in lines], loc="best")

        plt.tight_layout()
        plt.show()
    return results


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