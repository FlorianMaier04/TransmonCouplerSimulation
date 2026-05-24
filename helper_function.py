import numpy as np
import qutip as qt
import sympy as sp
from sympy.physics.quantum import Dagger
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.operatorordering import normal_ordered_form
from Floquet_perturbation_theory import *

### --- For Transmon-Coupler-Transmon system Substitution of phase and charge operators into creation and annihilation operators --- ###

def SecondQuantizedForm(Hamiltonian):
  """
  Substitute phase and charge operators in terms of creation and annihilation operator and expand the resulting expression
  """
  EJSigma = sp.symbols("E_{J\\Sigma}")
  n1, n2, nc = sp.symbols("n_1, n_2, n_c")
  phi1, phi2, phic = sp.symbols("\\phi_1, \\phi_2, \\phi_c")
  a1 = BosonOp("a_1")
  ad1 = Dagger(a1)
  a2 = BosonOp("a_2")
  ad2 = Dagger(a2)
  ac = BosonOp("a_c")
  adc = Dagger(ac)
  EC1, EC2, ECc = sp.symbols("E_{C1}, E_{C2}, E_{Cc}")
  EJ1, EJ2, EJc = sp.symbols("E_{J1}, E_{J2}, E_{Jc}")


  Hamiltonian_substituted = Hamiltonian.subs(
      {
        phi1: (2 * EC1 / EJ1) ** sp.Rational(1, 4) * (a1 + ad1),
        n1: sp.I * (EJ1 / (32 * EC1)) ** sp.Rational(1, 4) * (ad1 - a1),
        phi2: (2 * EC2 / EJ2) ** sp.Rational(1, 4) * (a2 + ad2),
        n2: sp.I * (EJ2 / (32 * EC2)) ** sp.Rational(1, 4) * (ad2 - a2),
        phic: (2 * ECc / EJSigma) ** sp.Rational(1, 4) * (ac + adc),
        nc: sp.I * (EJSigma / (32 * ECc)) ** sp.Rational(1, 4) * (adc - ac),
      }
  )

  #expand higher power terms
  Hamiltonian_substituted = Hamiltonian_substituted.expand()
  #put into normal ordering
  Hamiltonian_substituted = normal_ordered_form(Hamiltonian_substituted, independent=True)
  #combine and collect like terms
  Hamiltonian_substituted = sp.powdenest(Hamiltonian_substituted, force=True)
  Hamiltonian_substituted = sp.powsimp(Hamiltonian_substituted, force=True)

  return Hamiltonian_substituted


### --- For Transmoon-Coupler-Transmon system Simplified Hamiltonian --- ###

def HardCore(expr, a1, a2, ac):
    """remove the off-diagonal terms in each of the single bosonnic modes,
    which corresponds to the hard-core boson limit
    where each mode can only have 0 or 1 excitation.
    This is a common approximation in the dispersive regime
    where the anharmonicity is large compared to the coupling strength,
    so that higher excitations are energetically suppressed.

    Args:
        expr (): sympy expression containing the bosonic ladder operators a1, a2, ac and their daggers

    Returns:
        _type_: sympy expression with the off-diagonal terms removed
    """
    expr = expr.replace(
        lambda x: x.is_Pow
        and x.exp >= 2
        and (
            x.base == a1
            or x.base == a2
            or x.base == ac
            or x.base == Dagger(a1)
            or x.base == Dagger(a2)
            or x.base == Dagger(ac)
        ),
        lambda x: 0,
    )
    return expr

### --- For Transmon-Coupler-Transmon system Diagonalization --- ###
def SortedFRFSpectrum(evals, evecs_qobj, dim_q1, dim_r, dim_q2, fidelity_threshold=0.5):
    """
    Sort eigenvalues/eigenvectors of a two-qubit–resonator system into branches
    |i_q1, n_r, i_q2> based on total qubit excitation number (i_q1 + i_q2)
    and resonator excitation n_r.

    The algorithm:
    1. Groups qubit states by total excitation N_q = i_q1 + i_q2.
    2. For each N_q branch, starts at n_r = 0 and finds eigenstates with
       maximal fidelity to the bare states |i_q1, 0, i_q2>.
    3. Then "climbs" the resonator ladder using the resonator creation operator
       to locate higher n_r levels.
    4. Returns fixed-size arrays for convenience.

    Parameters
    ----------
    evals : array_like
        Eigenvalues (1D array).
    evecs_qobj : list of Qobj
        Corresponding normalized eigenvectors.
    dim_q1, dim_q2, dim_r : int
        Dimensions of the two qubits and the resonator Hilbert space.
    fidelity_threshold : float, optional
        Minimal fidelity to consider a match valid.

    Returns
    -------
    sorted_evals : np.ndarray, shape (dim_q1, dim_r, dim_q2)
    sorted_evecs : np.ndarray, shape (dim_q1, dim_r, dim_q2), dtype=object
    """

    used = set()
    sorted_evals = np.full((dim_q1, dim_r, dim_q2), np.nan)
    sorted_evecs = np.empty((dim_q1, dim_r, dim_q2), dtype=object)

    # Resonator creation operator
    a_r = qt.destroy(dim_r)
    excit_op = qt.tensor(qt.qeye(dim_q1), a_r.dag(), qt.qeye(dim_q2))

    def find_best_match(target):
        """Find unused eigenstate with maximal fidelity to target."""
        best_idx, best_fid = None, -1.0
        for i, state in enumerate(evecs_qobj):
            if i in used:
                continue
            fid = qt.fidelity(state, target)
            if fid > best_fid:
                best_fid, best_idx = fid, i
        return best_idx, best_fid

    # Maximum total qubit excitation
    max_qubit_exc = (dim_q1 - 1) + (dim_q2 - 1)

    # Loop over total qubit excitation manifolds
    for Nq in range(max_qubit_exc + 1):

        # --- Step 1: find all (iq1, iq2) combos with this total excitation
        qubit_pairs = [
            (iq1, iq2)
            for iq1 in range(dim_q1)
            for iq2 in range(dim_q2)
            if iq1 + iq2 == Nq
        ]

        for iq1, iq2 in qubit_pairs:
            # --- Step 2: find best match for resonator ground state
            target = qt.tensor(
                qt.basis(dim_q1, iq1), qt.basis(dim_r, 0), qt.basis(dim_q2, iq2)
            )
            idx, fid = find_best_match(target)
            if idx is None or fid < fidelity_threshold:
                continue

            used.add(idx)
            sorted_evals[iq1, 0, iq2] = evals[idx]
            # keep phase continuity when possible
            if (evecs_qobj[idx].dag() @ target).real < 0:
                sorted_evecs[iq1, 0, iq2] = -evecs_qobj[idx]
            else:
                sorted_evecs[iq1, 0, iq2] = evecs_qobj[idx]
            current_state = evecs_qobj[idx]

            # --- Step 3: climb resonator ladder for this qubit pair
            for jr in range(1, dim_r):
                excited = (excit_op * current_state).unit()
                idx, fid = find_best_match(excited)
                if idx is None or fid < fidelity_threshold:
                    break
                used.add(idx)
                sorted_evals[iq1, jr, iq2] = evals[idx]
                # keep phase continuity when possible
                if (evecs_qobj[idx].dag() @ excited).real < 0:
                    sorted_evecs[iq1, jr, iq2] = -evecs_qobj[idx]
                else:
                    sorted_evecs[iq1, jr, iq2] = evecs_qobj[idx]
                current_state = evecs_qobj[idx]

    return sorted_evals, sorted_evecs

### ---  For Transmon-Coupler-Transmon system Analytical expressions for energies and coupling matrix elements --- ###
def get_analytical_matrices(dim_q1, dim_c, dim_q2, V0, V1):
    """
    Generate analytical expressions for energies and coupling matrix elements
    for a Transmon-Coupler-Transmon system.

    This function constructs symbolic representations of the energy levels and
    coupling matrix elements based on the dimensions of the qubits and coupler, 
    as well as the provided coupling matrices V0 and V1. The energies are labeled
    according to the basis states |i_q1, n_c, i_q2>, and the coupling elements are 
    labeled according to their corresponding bra-ket notation.
    
    Args:
    dim_q1 (int): Dimension of the first qubit Hilbert space.
    dim_c (int): Dimension of the coupler Hilbert space.
    dim_q2 (int): Dimension of the second qubit Hilbert space.
    V0 (array_like): Static coupling matrix elements (2D array of shape (D, D)).
    V1 (array_like): Drive coupling matrix elements (2D array of shape (D, D)).
    
    Returns:
    E_analytics (np.ndarray): Symbolic expressions for energy levels, shape (D,).
    V0_analytics (np.ndarray): Symbolic expressions for static coupling elements, shape (D, D).
    V1_analytics (np.ndarray): Symbolic expressions for drive coupling elements, shape (D, D).
    
    """
    D = dim_q1 * dim_c * dim_q2

    ## Analytics 3 levels using sympy
    E_analytics = np.zeros(D, dtype=sy.Symbol)
    for n1 in range(dim_q1):
        for nc in range(dim_c):
            for n2 in range(dim_q2):
                n = n1 * dim_c * dim_q2 + nc * dim_q2 + n2
                E_analytics[n] = sy.symbols(f"E_{{\\ket{{{n1}{nc}{n2}}}}}", real=True)

    ## Non-zero matrix elements of static coupling V0 and drive coupling V1
    V0_analytics = np.zeros((D, D), dtype=sy.Symbol)
    V1_analytics = np.zeros((D, D), dtype=sy.Symbol)
    for i1 in range(dim_q1):
        for i2 in range(dim_q2):
            for ic in range(dim_c):
                for jc in range(dim_c):
                    for j1 in range(dim_q1):
                        for j2 in range(dim_q2):
                            i = i1 * dim_c * dim_q2 + ic * dim_q2 + i2
                            j = j1 * dim_c * dim_q2 + jc * dim_q2 + j2
                            if V0[i, j] != 0:
                                V0_analytics[i, j] = sy.symbols(
                                    f"\\bra{{{i1}{ic}{i2}}}V_0{{\\ket{{{j1}{jc}{j2}}}}}"
                                )
                            if V1[i, j] != 0:
                                V1_analytics[i, j] = sy.symbols(
                                    f"\\bra{{{i1}{ic}{i2}}}V_1{{\\ket{{{j1}{jc}{j2}}}}}"
                                )
    return E_analytics, V0_analytics, V1_analytics

def clean_expr(expr):
    expr = sp.expand(expr)            # Produkte auflösen
    expr = sp.simplify(expr)          # grob vereinfachen

    # 🔥 entscheidend: explizit 0-Terme entfernen
    if expr.is_Add:
        expr = sp.Add(*[t for t in expr.args if t != 0])

    return expr


### --- Dynamics prediction using Floquet perturbation theory --- ###

def Psi_t_FloquetPerturb(rH, rW, wd, amp, resonances, E, V_posHarm, initial_state, t_list):
    """
    Calculate time-evolved wavefunction using Floquet perturbation theory.
    
    This function computes the time evolution of a quantum state under a periodically
    driven Hamiltonian using Floquet perturbation theory to a specified order. It 
    calculates both the effective Floquet Hamiltonian and the micromotion operator,
    then evolves the initial state and normalizes the result at each time point.

    Args:
        rH (int): Order of perturbation theory for the effective Hamiltonian.
        rW (int): Order of perturbation theory for the micromotion operator W.
        wd (float): Drive frequency (angular frequency of the periodic modulation).
        resonances (dict): Dictionary mapping state indices to their Floquet sectors,
            e.g., {a:na, b:nb, ...}. Such that E[a]-na*wd ~ E[b]-nb*wd.
        E (array_like): Energy eigenvalues of the undriven system (1D array).
        V_posHarm (array_like or list): Matrix elements of the positive harmonic
            drive coupling (2D array) or list of such matrices for multiple harmonics.
        initial_state (array_like): Initial state vector in the Hilbert space (1D array).
        t_list (array_like): Array of time points at which to evaluate the wavefunction.

    Returns:
        np.ndarray: Time-evolved wavefunction, shape (dim, len(t_list)), where dim is
            the dimension of the Hilbert space. The wavefunction is normalized at each
            time point. Each column psi_t[:, i] represents the state at time t_list[i].
    """
    
    t = sy.symbols("t", real=True)

    ## Substitution time in analytical solution and normalize wavefunction
    def substitue_time_normalize(psi_t_analyt):
        func = sy.lambdify(t, sy.Array(psi_t_analyt))
        psi_t_analyt_sub = np.array([func(tval) for tval in t_list]).T
        ## Normalization of wavefunction
        norm_t = np.sum(np.abs(psi_t_analyt_sub) ** 2, axis=0)
        psi_t_analyt_sub = np.einsum("kt,t->kt", psi_t_analyt_sub, 1 / np.sqrt(norm_t))
        return psi_t_analyt_sub
    
    Heff_matrix = Heff_Floquet_matrix_summed(
        rH, wd, resonances, E, amp / 2 * V_posHarm 
        )
    W_element = W_Floquet_elements(
        rW, wd, resonances, E, amp / 2 * V_posHarm,  
        )
    psi_t = substitue_time_normalize(
        dynamics_eff(
            initial_state, rH, rW,
            Heff_matrix, W_element,
            t, wd, resonances, E,
            )
        )
    return psi_t, W_element



def find_optimal_time(fre, amp, order, i, f, l, E0, V1, t_guess):
    """
    Find optimal time that maximizes fidelity with iSWAP gate.

    Parameters:
    -----------
    fre : float
        Drive frequency
    amp : float
        Drive amplitude
    order : int
        Perturbation theory order
    i, f, l : int
        Indices of states forming the computational subspace
    E0 : array
        Energy eigenvalues
    V1 : array
        Perturbation matrix
    fre_guess : float
        Initial guess for drive frequency
    t_guess : float
        Expected gate time (used to set time window)

    Returns:
    --------
    optimal_t : float
        Time at which maximum fidelity occurs
    optimal_fidelity : float
        Maximum achieved fidelity with iSWAP gate
    optimal_U : array (3x3)
        Subsystem unitary at optimal time
    """
    tlist = np.arange(0, 1.5 * t_guess, 0.01)

    # Define initial states for subsystem basis {i, f, l}
    initial_state_a = np.zeros(len(E0))
    initial_state_a[i] = 1
    initial_state_b = np.zeros(len(E0))
    initial_state_b[f] = 1
    initial_state_c = np.zeros(len(E0))
    initial_state_c[l] = 1
    initial_state_list = [initial_state_a, initial_state_b, initial_state_c]

    # Define ideal iSWAP gate for subsystem {i, f, l}
    # iSWAP: |i⟩→i|f⟩, |f⟩→i|i⟩, |l⟩→|l⟩ (spectator)
    U_ideal = np.array([[0, 1j, 0],
                        [1j, 0, 0],
                        [0, 0, 1]], dtype=complex)

    # Storage for time-evolved unitaries
    lU = np.zeros((len(tlist), 3, 3), dtype=complex)

    # Evolve each basis state and construct unitary matrix
    for j, initial_state in enumerate(initial_state_list):
        psi_t = Psi_t_FloquetPerturb(
            order, 0, fre, amp,
            {i: 1, f: 0, l: 2},
            E0, V1, initial_state, tlist,
        )
        # Extract subsystem components at each time
        # psi_t[k, :] is the evolved state at time tlist[k]
        lU[:, 0, j] = psi_t[i, :]  # Project onto state i
        lU[:, 1, j] = psi_t[f, :]  # Project onto state f
        lU[:, 2, j] = psi_t[l, :]  # Project onto state l

    # Compute gate fidelity at each time: F = |Tr(U_ideal† U(t))|² / 3^2
    fidelities = np.zeros(len(tlist))
    for k in range(len(tlist)):
        trace_overlap = np.trace(U_ideal.conj().T @ lU[k])
        fidelities[k] = np.abs(trace_overlap)**2 / 3**2

    max_fidelity = np.max(fidelities)
    t_max = tlist[np.argmax(fidelities)]
    U_max = lU[np.argmax(fidelities)]

    return t_max, max_fidelity, U_max