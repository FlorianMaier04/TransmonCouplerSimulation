"""
**General resonant Floquet perturbation theory for multiphoton processes**

Module containing the main functions to compute effective Hamiltonian and dynamics for multiphoton processes in a resonantly driven systems. 
The main functions are (see their respective help documentation for details):

- `Heff_Floquet()`: to compute the matrix elements of the effective Hamiltonian at a given order.

- `Heff_Floquet_summed()`: to compute the sum of lowest orders of the matrix elements of the effective Hamiltonian.

- `W_Floquet()`: to compute the non-zero matrix elements at a given order of the transformation `W` mapping the eigenstates of Heff in the degenerate Sambe subspace to the eigenstates of the full Sambe Hamiltonian in the full Sambe space.

- `dynamics_eff()`: to compute the effective evolution of the wavefunction obtained from an effective Hamiltonian and operator `W`. The function takes as input the outputs of the functions `Heff_Floquet_matrix_summed()` and `W_Floquet_elements` (see their respective help documentation).
"""

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm
import itertools as it
import sympy as sy
import os
from functools import lru_cache
from Floquet_perturbation_theory import *

from tqdm import tqdm

## Folder containing the .txt files of DPT coefficients
folder_DPT = os.path.join(os.path.dirname(__file__), 'DPT')

def _res_dict(resonances):
    return resonances if isinstance(resonances, dict) else {k: n for k, n in resonances}

def _harm_list(V_posHarm):
    V = np.asarray(V_posHarm)
    return [V] if V.ndim == 2 else list(V)

def _V_from_Vtilde(V_posHarm, A=None, analytics=False):
    Vtilde = _harm_list(V_posHarm)
    nH = len(Vtilde)
    if A is None:
        A = np.ones(nH, dtype=object if analytics else complex)
    else:
        A = np.array(A, dtype=object if analytics else complex)
        if A.ndim == 0:
            A = np.array([A] * nH, dtype=object if analytics else complex)
    return [A[q] / 2 * Vtilde[q] for q in range(nH)]

@lru_cache(None)
def _dpt_coeffs(order, unitary=True, analytics=False):
    if order == 0:
        return {(): 1}
    if order == 1:
        return {(1,): 1}
    fn = f"{folder_DPT}/DPT{'_unitaryW' if unitary else ''}_coefficients_order_{order}.txt"
    raw = np.atleast_2d(np.loadtxt(fn))
    out = {}
    for row in raw:
        key = tuple(int(x) for x in row[:-1])
        c = row[-1]
        if analytics:
            num, den = c.as_integer_ratio()
            out[key] = sy.Rational(num, den)
        else:
            out[key] = float(c)
    return out

def _V_of(p, V0, Vtilde, Vdag, A):
    if p == 0:
        return V0
    q = abs(p) - 1
    if q >= len(Vtilde):
        return None
    amp = 1 if A is None else A[q]
    return amp/2 * (Vtilde[q] if p > 0 else Vdag[q])

def W_Floquet_elements_fast(rW, wd, resonances, E, V_posHarm,
                            A=None, V0=None, ref_state=None, analytics=False, verbose=False):
    """
    Returns:
        W[(r, l, a)][p] = (Wcoeff, dW_dwd_den, dW_dAvec)
    where
        dW_dwd_den : derivative wrt wd from denominators only
        dW_dAvec   : vector of derivatives wrt amplitudes A_q
    """
    res = _res_dict(resonances)
    E = np.array(E, dtype=object if analytics else complex)
    Vtilde = _harm_list(V_posHarm)
    if Vtilde[0].ndim != 2:
        raise ValueError("V_posHarm must be a matrix or a list of matrices.")
    D = len(E)
    nH = len(Vtilde)

    if A is None:
        A = np.ones(nH, dtype=object if analytics else complex)
    else:
        A = np.array(A, dtype=object if analytics else complex)
        if A.ndim == 0:
            A = np.array([A] * nH, dtype=object if analytics else complex)

    if V0 is None:
        V0 = np.zeros((D, D), dtype=object if analytics else complex)
    else:
        V0 = np.array(V0, dtype=object if analytics else complex)

    if ref_state is None:
        ref_state = min(res.keys())

    for k in res:
        if k != ref_state:
            V0[k, k] = V0[k, k] + E[k] - E[ref_state] - (res[k] - res[ref_state]) * wd
            E[k] = E[ref_state] + (res[k] - res[ref_state]) * wd

    Vdag = [v.conj().T for v in Vtilde]

    # `_V_of` depends only on `p` here -- V0, Vtilde, Vdag, A are fixed for
    # the remainder of this call. Without caching, the (potentially
    # symbolic, expensive-to-assemble) D x D matrix for a given p is
    # rebuilt from scratch every time it is requested, which in the
    # analytics=True case can happen tens of thousands of times for the
    # same handful of distinct p values. This is very likely the single
    # biggest lever on runtime in symbolic mode.
    V_cache = {}
    def V(p):
        if p not in V_cache:
            V_cache[p] = _V_of(p, V0, Vtilde, Vdag, A)
        return V_cache[p]

    W = {}

    for r in range(rW + 1):
        if r == 0:
            zeroA = np.zeros(nH, dtype=object if analytics else complex)
            for l in range(D):
                for a in res:
                    W[(0, l, a)] = {res[a]: (1 if l == a else 0, 0, zeroA.copy())}
            continue

        coeffsDPT = _dpt_coeffs(r, unitary=True, analytics=analytics)

        # Zustandspaare (l, a) vorbereiten
        state_pairs = [(l, a) for l in range(D) for a in res]

        # tqdm pro Ordnung r für die Zustandspaare
        if verbose:
            state_pairs = tqdm(
                state_pairs,
                desc=f"W-Floquet order r={r}/{rW}",
                leave=True,
                disable=not verbose
            )

        for l, a in state_pairs:
            # Collect raw symbolic contributions per p_final in plain
            # Python lists instead of accumulating with `+=` on the fly.
            # sympy's Add re-processes the *entire* growing expression on
            # every `+=`, so incremental accumulation over a large
            # combinatorial loop is effectively quadratic; combining once
            # via sy.Add(*list) at the end is linear (plus one flatten/sort
            # pass) and, for large N, dramatically cheaper.
            terms = {}

            for p_list in it.product(range(-nH, nH + 1), repeat=r):
                p_final = res[a] + sum(p_list)

                for virt in it.product(range(D), repeat=r - 1):
                    chain = [a] + list(virt) + [l]

                    R, Rcomp = [], []
                    for j in range(1, r + 1):
                        if chain[j] in res and res[a] + sum(p_list[:j]) == res[chain[j]]:
                            R.append(j)
                        else:
                            Rcomp.append(j)

                    for exps, cDPT in coeffsDPT.items():
                        m = [None] + list(np.array(exps[::-1], dtype=int))
                        if not all(m[j] == 0 for j in R):
                            continue
                        if not all(m[j] != 0 for j in Rcomp):
                            continue

                        term = cDPT
                        amp_pow = np.zeros(nH, dtype=int)
                        ptot = 0
                        ok = True

                        steps = []

                        for j in range(1, r + 1):
                            pj = p_list[j - 1]
                            ptot += pj
                            eg = E[a] + ptot * wd - E[chain[j]]

                            Vj = V(pj)
                            if Vj is None:
                                ok = False
                                break

                            Vij = Vj[chain[j], chain[j - 1]]

                            term *= Vij * (eg ** (-m[j]))

                            if pj != 0:
                                amp_pow[abs(pj) - 1] += 1

                            steps.append((pj, eg, Vij, m[j], ptot))

                        if not ok:
                            continue

                        # ---------- computing derivatives -----------
                        # Fixed: only accumulate the *completed* chain
                        # product for a given i, once the inner `for j`
                        # loop has finished building it -- not every
                        # intermediate partial product along the way.
                        dterm_dwd_terms = []
                        for i in Rcomp:
                            contrib = cDPT
                            for j, (pj, eg, Vij, mj, ptot) in enumerate(steps, start=1):
                                if j == i:
                                    contrib *= Vij * (-mj * ptot) * (eg ** (-mj - 1))
                                else:
                                    contrib *= Vij * (eg ** (-mj))
                            dterm_dwd_terms.append(contrib)

                        if p_final not in terms:
                            terms[p_final] = [[], [], [[] for _ in range(nH)]]

                        terms[p_final][0].append(term)
                        terms[p_final][1].extend(dterm_dwd_terms)

                        if A is not None:
                            for q in range(nH):
                                terms[p_final][2][q].append(
                                    term * (amp_pow[q] / A[q] if A[q] != 0 else 0)
                                )

            # Combine each p_final's collected terms exactly once.
            W_la = {}
            for p_final, (term_list, dwd_list, dA_lists) in terms.items():
                if analytics:
                    coeff = sy.Add(*term_list)
                    dwd = sy.Add(*dwd_list) if dwd_list else sy.Integer(0)
                    dA_vec = np.array(
                        [sy.Add(*dA_lists[q]) if dA_lists[q] else sy.Integer(0) for q in range(nH)],
                        dtype=object
                    )
                else:
                    coeff = sum(term_list)
                    dwd = sum(dwd_list) if dwd_list else 0
                    dA_vec = np.array(
                        [sum(dA_lists[q]) if dA_lists[q] else 0 for q in range(nH)],
                        dtype=complex
                    )
                W_la[p_final] = (coeff, dwd, dA_vec)

            W[(r, l, a)] = W_la

    if verbose:
        print("✅ W-elements calculation completed.")

    return W

def _fast_sympy_add(terms, chunk_size=50):
    if not terms:
        return sy.S.Zero
    
    # In kleineren Blöcken addieren
    current_level = [sy.Add(*terms[i:i + chunk_size]) for i in range(0, len(terms), chunk_size)]
    
    # Hierarchisch zusammenführen
    while len(current_level) > 1:
        current_level = [
            sy.Add(*current_level[i:i + chunk_size]) 
            for i in range(0, len(current_level), chunk_size)
        ]
        
    return current_level[0]

def Heff_ad_correction_Floquet_fast(r, l, k, wd, A, dwd, dA, t,
                                     resonances, E, V_posHarm, V0=None,
                                     ref_state=None, analytics=False, rW=-1,
                                     W=None, include_geometric=True,
                                     include_micromotion=True,
                                     include_g_correction=True,
                                     verbose=False):
    """
    Total adiabatic correction, projected onto the resonant subspace.
    Combines two contributions in a single (r1, r2, a, p) Sambe-space loop:

      - geometric:   +i * dW^{(r1)†}/dt * W^{(r2)}   (i.e. i dW^†/dt W, NOT
                     -i W^† dW/dt). The gradient acts on the r_{W1} (bra/b)
                     factor, and the FULL time derivative of that factor is
                     conjugated as a whole before multiplying the plain,
                     undifferentiated r_{W2} (ket/a) factor.
                     (toggle: include_geometric)
      - micromotion: -t*dwd*p * W^† p W  (+ G-correction)
    """
    if not (include_geometric or include_micromotion):
        return sy.S.Zero if analytics else 0

    res = _res_dict(resonances)
    if rW < 0:
        rW = r

    dtype = object if analytics else complex
    E = np.array(E, dtype=dtype)
    V_posHarm = _harm_list(V_posHarm)
    nH = len(V_posHarm)

    A = np.array(A if A is not None else np.ones(nH), dtype=dtype)
    if A.ndim == 0:
        A = np.array([A] * nH, dtype=dtype)

    dA = np.array(dA if dA is not None else np.zeros(nH), dtype=dtype)
    if dA.ndim == 0:
        dA = np.array([dA] * nH, dtype=dtype)

    if W is None:
        W = W_Floquet_elements_fast(
            rW, wd, res, E, V_posHarm, A=A, V0=V0, ref_state=ref_state, analytics=analytics, verbose=verbose
        )

    do_micro = include_micromotion and dwd != 0

    # Collect raw contributions in plain Python lists and combine once at
    # the end via sy.Add(*list). Incremental `out += term` inside a large
    # combinatorial (r1, r2, a, p[, p2]) loop forces sympy to re-flatten
    # and re-collect the entire, growing expression on every single
    # addition (effectively O(N^2) in the number of terms N). For
    # analytics=True this dominates runtime far more than the actual
    # arithmetic. sy.Add(*list) performs a single linear-time batch
    # collection instead.
    geom_terms = []
    micro_terms = []

    r_pairs = [(r1, r2) for r1 in range(rW + 1) for r2 in range(rW + 1) if r1 + r2 <= r]
    if verbose:
        print(f"Calculating adiabatic Floquet corrections for element ({l}, {k}) up to order r = {r}...")
        r_pairs = tqdm(r_pairs, desc=f"Heff ad correction (l={l}, k={k})", disable=not verbose)

    for r1, r2 in r_pairs:
        for a in range(len(E)):
            Wal = W[(r1, a, l)]  # doesn't depend on r2 -> fetch once here
            Wak = W[(r2, a, k)]

            for p, (Wap_l, dwd_dWap_l, dA_dWap_l) in Wal.items():
                if p not in Wak:
                    continue
                Wap_k, _, _ = Wak[p]  # plain value only, no derivative needed here

                if include_geometric:
                    dt_dWap_l = dwd * dwd_dWap_l + np.dot(dA, dA_dWap_l)
                    geom_terms.append(1j * np.conj(dt_dWap_l) * Wap_k)

                if do_micro:
                    conj_Wbp = np.conj(Wap_l)
                    micro_terms.append(-t * dwd * p * conj_Wbp * Wap_k)
                    if include_g_correction:
                        for p2, (Wap2, _, _) in Wak.items():
                            if p2 == p:
                                continue
                            phase = np.exp(1j * (p - p2) * wd * t)
                            micro_terms.append(t * dwd * p2 * phase * conj_Wbp * Wap2)

    if analytics:
        out_geom = _fast_sympy_add(geom_terms)
        out_micro = _fast_sympy_add(micro_terms)
    else:
        out_geom = sum(geom_terms) if geom_terms else 0
        out_micro = sum(micro_terms) if micro_terms else 0

    if l == k:
        out_micro = sy.re(out_micro) if analytics else np.real(out_micro)

    if verbose:
        print(f"✅ Adiabatic correction calculation for element ({l}, {k}) completed.")

    return out_geom + out_micro

# def Heff_ad_correction_Floquet_fast(r, b, a, wd, A, dwd, dA, t,
#                                      resonances, E, V_posHarm, V0=None,
#                                      ref_state=None, analytics=False, rW=-1,
#                                      W=None, include_geometric=True,
#                                      include_micromotion=True,
#                                      include_g_correction=True):
#     """
#     Total adiabatic correction, projected onto the resonant subspace.
#     Combines two contributions in a single (r1, r2, a, p) Sambe-space loop:

#       - geometric:   -i * W^† dW/dt                       (toggle: include_geometric)
#       - micromotion: -t*dwd*p * W^† p W  (+ G-correction)  (toggle: include_micromotion,
#                                                               include_g_correction)
#     """
#     if not (include_geometric or include_micromotion):
#         return sy.S.Zero if analytics else 0

#     res = _res_dict(resonances)
#     if rW < 0:
#         rW = r

#     dtype = object if analytics else complex
#     E = np.array(E, dtype=dtype)
#     V_posHarm = _harm_list(V_posHarm)
#     nH = len(V_posHarm)

#     A = np.array(A if A is not None else np.ones(nH), dtype=dtype)
#     if A.ndim == 0:
#         A = np.array([A] * nH, dtype=dtype)

#     dA = np.array(dA if dA is not None else np.zeros(nH), dtype=dtype)
#     if dA.ndim == 0:
#         dA = np.array([dA] * nH, dtype=dtype)

#     if W is None:
#         W = W_Floquet_elements_fast(
#             rW, wd, res, E, V_posHarm, A=A, V0=V0, ref_state=ref_state, analytics=analytics
#         )

#     zero = sy.S.Zero if analytics else 0
#     out_geom, out_micro = zero, zero
#     do_micro = include_micromotion and dwd != 0

#     for r1 in range(rW + 1):
#         for r2 in range(rW + 1):
#             if r1 + r2 > r:
#                 continue
#             for al in range(len(E)):
#                 Wbl = W[(r1, al, b)]
#                 Wak = W[(r2, al, a)]

#                 for p, (Wbp, _, _) in Wbl.items():
#                     if p not in Wak:
#                         continue
#                     Wap, dwd_den, dA_vec = Wak[p]
#                     conj_Wbp = np.conj(Wbp)

#                     if include_geometric:
#                         dWdt = dwd * dwd_den + np.dot(dA, dA_vec)
#                         out_geom += -1j * conj_Wbp * dWdt

#                     if do_micro:
#                         out_micro += -t * dwd * p * conj_Wbp * Wap
#                         if include_g_correction:
#                             for p2, (Wap2, _, _) in Wak.items():
#                                 if p2 == p:
#                                     continue
#                                 phase = np.exp(1j * (p - p2) * wd * t)
#                                 out_micro += t * dwd * p2 * phase * conj_Wbp * Wap2

#     if b == a:
#         out_micro = sy.re(out_micro) if analytics else np.real(out_micro)

#     return out_geom + out_micro


def Heff_Floquet_total_matrix_summed(rH, wd, A, resonances, E, V_posHarm,
                                     V0=None, ref_state=None,
                                     dwd=0, dA=None, t=0,
                                     analytics=False, rW=-1,
                                     include_geometric=True,
                                     include_micromotion=True,
                                     include_g_correction=True, W=None,
                                     verbose=False):
    """
    Base Heff + adiabatic correction.
    V_posHarm should be the normalized harmonics V~_p.
    """
    if type(resonances) == list and type(resonances[0]) in (list, tuple):
        resonances = {k: n for (k, n) in resonances}

    E = np.array(E)
    V_posHarm = np.array(V_posHarm)
    if E.dtype == 'O' or V_posHarm.dtype == 'O':
        analytics = True

    d_res = len(resonances)
    ind_inv = list(resonances.keys())

    dtype = sy.Symbol if analytics else complex
    H = np.zeros((d_res, d_res), dtype=dtype)

    if (include_geometric or include_micromotion) and not W:
        if verbose:
            print("Calculating W-Floquet elements for corrections...")
        W = W_Floquet_elements_fast(
            rW if rW >= 0 else rH, wd, resonances, E, V_posHarm,
            A=A, V0=V0, ref_state=ref_state, analytics=analytics,
            verbose=verbose
        )

    V_full = _V_from_Vtilde(V_posHarm, A=A, analytics=analytics)

    # Indizes-Paare für Matrix-Schleife generieren
    pairs = [(i, bi, j, aj) for i, bi in enumerate(ind_inv) for j, aj in enumerate(ind_inv)]
    if verbose:
        print(f"Calculating effective matrix elements ({d_res}x{d_res})...")
        pairs = tqdm(pairs, desc="H_eff matrix elements", disable=not verbose)

    for i, bi, j, aj in pairs:
        H[i, j] = (
            Heff_Floquet_summed(rH, bi, aj, wd, resonances, E, V_full,
                                V0=V0, ref_state=ref_state,
                                analytics=analytics))
        if include_geometric or include_micromotion:
            H[i, j] += Heff_ad_correction_Floquet_fast(
                rH, bi, aj, wd, A, dwd, dA, t,
                resonances, E, V_posHarm, V0=V0,
                ref_state=ref_state, analytics=analytics, rW=rW,
                W=W, include_geometric=include_geometric,
                include_micromotion=include_micromotion,
                include_g_correction=include_g_correction, verbose=verbose
            )

    if verbose:
        print("✅ H_eff matrix calculation completed.")

    return H

import time
import numpy as np
import sympy as sy
import scipy.linalg as scipy_linalg

def Psi_t_hybrid_piecewise(rH, rW, wd, A, resonances, E, V_posHarm,
                           initial_state, tlist, t_sym, V0=None, ref_state=None,
                           include_geometric=True, include_micromotion=True, 
                           include_g_correction=True, param_dict=None, verbose=False):
    """
    Hybrid time evolution:
    1. Computes Heff and W symbolically (fast).
    2. Lambdifies the matrices to ultra-fast NumPy functions.
    3. Solves the time evolution piecewise (numerically) over tlist.
    """
    def log(msg):
        if verbose: print(msg, flush=True)

    if param_dict is None:
        param_dict = {}

    res = _res_dict(resonances) if callable(globals().get('_res_dict')) else resonances
    if isinstance(resonances, list):
        resonances = dict(resonances)

    D, d_res = len(E), len(resonances)
    ind = {k: i for i, k in enumerate(resonances.keys())}
    kref = min(resonances)

    state = np.argmax(np.abs(initial_state.full().ravel()))
    if state not in ind:
        raise ValueError("Initial state is not contained in the resonant subspace.")
    initial_eff_idx = ind[state]

    # --- 1. Symbolische Vorarbeit ---
    log("[1/3] Computing symbolic transformation W ...")

    log("[2/3] Computing symbolic effective Hamiltonian Heff ...")


    log("[3/3] Constructing full transformation matrix M(t) and lambdifying ...")
    # M_sym ist die Transformationsmatrix, die den effektiven Zustand zurück in den 
    # ursprünglichen Hilbertraum mappt, inklusive der Floquet-Phasen.
    M_sym = sy.zeros(D, d_res)
    for r in range(rW):
        for l in range(D):
            for idx_a, a in enumerate(res):
                W_la = W[r, l, a]
                for p, (W_lp_a, _, _) in W_la.items():
                    phase = sy.exp(-sy.I * (E[kref] + p * wd) * t_sym)
                    M_sym[l, idx_a] += phase * W_lp_a

    # Parameter einsetzen, BEVOR wir lambdify aufrufen
    Heff_sub = sy.Matrix(Heff).subs(param_dict).doit()
    M_sub = M_sym.subs(param_dict).doit()

    # SymPy Ausdrücke in C-schnelle NumPy Funktionen umwandeln
    H_func = sy.lambdify(t_sym, Heff_sub, modules=['numpy', 'scipy'])
    M_func = sy.lambdify(t_sym, M_sub, modules=['numpy', 'scipy'])

    # --- 2. Numerische Zeitentwicklung ---
    log("      Running fast piecewise numerical time evolution ...")
    
    # Array für die Ergebnisse (Zeilen: Zustände, Spalten: Zeitschritte)
    psi_t_num = np.zeros((D, len(tlist)), dtype=complex)
    
    # Startzustand im effektiven Frame (d_res x 1)
    c_eff = np.zeros(d_res, dtype=complex)
    c_eff[initial_eff_idx] = 1.0

    for i, t_val in enumerate(tlist):
        # 1. Speichere den Zustand im ursprünglichen Hilbertraum ab
        M_val = np.array(M_func(t_val), dtype=complex).reshape(D, d_res)
        psi_t_num[:, i] = M_val @ c_eff
        
        # 2. Propagiere den effektiven Zustand c_eff zum nächsten Zeitschritt
        if i < len(tlist) - 1:
            dt = tlist[i+1] - t_val
            # Midpoint-Regel für höhere Genauigkeit (Magnus 2. Ordnung Approximation)
            t_mid = t_val + dt / 2.0
            
            H_mid = np.array(H_func(t_mid), dtype=complex).reshape(d_res, d_res)
            
            # Zeitentwicklungsoperator für den kleinen Schritt dt
            dU = scipy_linalg.expm(-1j * H_mid * dt)
            c_eff = dU @ c_eff

    log("✅ Hybrid calculation completed!")
    return psi_t_num

def Psi_t_from_Heff(
    rH,
    rW,
    signal,
    resonances,
    E,
    V_posHarm,
    initial_state,
    tlist,
    V0=None,
    ref_state=None,
    include_geometric=True,
    include_micromotion=True,
    include_g_correction=True,
    substeps=1,
):

    if isinstance(resonances, list):
        resonances = {k: n for (k, n) in resonances}

    D = len(E)
    d_res = len(resonances)
    ind_inv = list(resonances.keys())
    ind = {k: i for i, k in enumerate(ind_inv)}
    kref = min(resonances)

    psi = np.zeros((D, len(tlist)), dtype=complex)
    dt = tlist[1] - tlist[0]

    # Anfangszustand im Resonanzunterraum
    state = np.argmax(np.abs(initial_state.full().ravel()))
    if state not in ind:
        raise ValueError(
            "Initial state is not contained in the resonant subspace."
        )

    phi = np.zeros(d_res, dtype=complex)
    phi[ind[state]] = 1

    for it, t in tqdm(enumerate(tlist), total=len(tlist)):

        wd, A = signal(t)

        W = W_Floquet_elements_fast(
            rW if rW >= 0 else rH,
            wd,
            resonances,
            E,
            V_posHarm,
            A=A,
            V0=V0,
        )

        psi_t = np.zeros(D, dtype=complex)

        for order in range(0, rW + 1):
            for l in range(D):
                for ia, a in enumerate(ind_inv):
                    for p, (w, _, _) in W[order, l, a].items():
                        psi_t[l] += w * np.exp(-1j * p * wd * t) * phi[ia]

        psi_t *= np.exp(-1j * E[kref] * t)

        n = np.linalg.norm(psi_t)
        if n > 0:
            psi[:, it] = psi_t / n
        else:
            psi[:, it] = psi_t

        if it == len(tlist) - 1:
            break

        sub_dt = dt / substeps

        for s in range(substeps):

            t_sub = t + s * sub_dt

            # Numerische Ableitungen per zentraler Differenz um t_sub
            eps = sub_dt * 1e-4 if sub_dt > 0 else 1e-7
            wd_plus, A_plus = signal(t_sub + eps)
            wd_minus, A_minus = signal(t_sub - eps)

            dwd_val = (wd_plus - wd_minus) / (2 * eps)
            dA_val = (A_plus - A_minus) / (2 * eps)

            wd_sub, A_sub = signal(t_sub)

            Heff = Heff_Floquet_total_matrix_summed(
                rH,
                wd_sub,
                A_sub,
                resonances,
                E,
                V_posHarm,
                V0=V0,
                ref_state=ref_state,
                dwd=dwd_val,
                dA=dA_val,
                t=t_sub,
                rW=rW,
                include_geometric=include_geometric,
                include_micromotion=include_micromotion,
                include_g_correction=include_g_correction,
                W=W,
            )

            phi = expm(-1j * Heff * sub_dt) @ phi

    return psi


import sympy as sy
import numpy as np
import scipy.linalg as scipy_linalg
import matplotlib.pyplot as plt
import time

def commutator(A, B):
    return sy.expand(A * B - B * A)

def magnus_operator(H, t, ti, tf, order=1, numeric_exp=True, verbose=True):
    """Compute 1st/2nd order Magnus expansion U = exp(Omega)."""
    def log(msg, end="\n"):
        if verbose: print(msg, end=end, flush=True)

    if order not in (1, 2):
        raise ValueError("Order must be 1 or 2.")
    
    # .doit() erzwingt die Auswertung eventueller Ableitungen (z.B. wd.diff(t)), 
    # die noch in H stecken könnten.
    H = sy.Matrix(H).doit() if not isinstance(H, sy.MatrixBase) else H.doit()
    rows, cols = H.shape
    t1, t2 = sy.Symbol("t1", real=True), sy.Symbol("t2", real=True)

    # 1st order
    log("  -> [Magnus] Calculating 1st order ...", end="")
    t_start = time.time()
    H1 = H.subs(t, t1).doit()
    Omega = sy.zeros(rows, cols)
    for i in range(rows):
        for j in range(cols):
            expr = sy.expand(H1[i, j])
            # conds='none' verhindert Piecewise-Objekte, .doit() erzwingt die Integration
            Omega[i, j] = -sy.I * sy.integrate(expr, (t1, ti, tf), conds='none').doit()
    log(f" Done ({time.time() - t_start:.2f}s)")

    # 2nd order
    if order == 2:
        log("  -> [Magnus] Calculating 2nd order ...", end="")
        t_start = time.time()
        H2 = H.subs(t, t2).doit()
        Comm = H1 * H2 - H2 * H1
        inner = sy.zeros(rows, cols)
        for i in range(rows):
            for j in range(cols):
                expr = sy.expand(Comm[i, j])
                inner[i, j] = sy.integrate(expr, (t2, ti, t1), conds='none').doit()

        Omega2 = sy.zeros(rows, cols)
        for i in range(rows):
            for j in range(cols):
                expr2 = sy.expand(inner[i, j])
                Omega2[i, j] = -sy.Rational(1, 2) * sy.integrate(expr2, (t1, ti, tf), conds='none').doit()
        Omega += Omega2
        log(f" Done ({time.time() - t_start:.2f}s)")

    log("  -> [Magnus] Simplifying Omega ...", end="")
    t_start = time.time()
    # Auch hier .doit() anhängen, falls trigsimp unaufgelöste Operationen hinterlässt
    Omega = sy.trigsimp(Omega).doit()
    log(f" Done ({time.time() - t_start:.2f}s)")

    # Exponential handling
    if numeric_exp:
        log("  -> [Magnus] Applying Taylor expansion approximation for U ...")
        U = sy.eye(rows) + Omega + sy.Rational(1, 2) * (Omega * Omega)
    else:
        log("  -> [Magnus] Computing exact Matrix.exp() ...", end="")
        t_start = time.time()
        U = Omega.exp()
        log(f" Done ({time.time() - t_start:.2f}s)")

    return U, Omega


def Psi_t_analytical_magnus(rH, rW, wd, A, resonances, E, V_posHarm,
                            initial_state, t, te, V0=None, ref_state=None,
                            include_geometric=True, include_micromotion=True, 
                            include_g_correction=True, param_dict=None, verbose=True):
    """Analytical time evolution via Floquet APT and Magnus expansion."""
    def log(msg, end="\n"):
        if verbose: print(msg, end=end, flush=True)

    res = _res_dict(resonances) if callable(globals().get('_res_dict')) else resonances
    if isinstance(resonances, list):
        resonances = dict(resonances)

    D, d_res = len(E), len(resonances)
    ind = {k: i for i, k in enumerate(resonances.keys())}
    kref = min(resonances)

    state = np.argmax(np.abs(initial_state.full().ravel()))
    if state not in ind:
        raise ValueError("Initial state is not contained in the resonant subspace.")

    log("\n[1/4] Computing transformation W ...")
    t0 = time.time()
    W = W_Floquet_elements_fast(rW, wd, resonances, E, V_posHarm, A=A, V0=V0, analytics=True)
    log(f"      Duration: {time.time() - t0:.2f}s")

    log("\n[2/4] Computing effective Hamiltonian Heff ...")
    t0 = time.time()
    Heff = Heff_Floquet_total_matrix_summed(
        rH, wd, A, resonances, E, V_posHarm, V0=V0, ref_state=ref_state,
        dwd=wd.diff(t), dA=A.diff(t), t=t, analytics=True, rW=rW, 
        include_geometric=include_geometric, include_micromotion=include_micromotion, 
        include_g_correction=include_g_correction, W=W
    )
    log(f"      Duration: {time.time() - t0:.2f}s")

    log("\n[3/4] Running Magnus expansion ...")
    t0 = time.time()
    U, _ = magnus_operator(Heff, t=t, ti=0, tf=te, order=2, numeric_exp=True, verbose=verbose)
    log(f"      Total Magnus duration: {time.time() - t0:.2f}s")

    log("\n[4/4] Constructing coefficients in original Hilbert space ...")
    t0 = time.time()
    psi_t = [0 for _ in range(D)]

    for r in range(rW):
        for l in range(D):
            for idx_a, a in enumerate(res):
                contrib = 0
                for idx_b, b in enumerate(res):
                    W_la = W[r, l, a]
                    for p, (W_lp_a, _, _) in W_la.items():
                        # sy.I anstelle von 1j genutzt, damit der Term vollständig symbolisch bleibt!
                        phase = sy.exp(-sy.I * (E[kref] + p * wd) * t)
                        contrib += phase * U[idx_a, idx_b] * W_lp_a
                psi_t[l] += contrib
                
    # Ein optionales .doit() am Ende kann noch Rest-Operationen abarbeiten
    psi_t = [sy.expand(component).doit() for component in psi_t]
    
    log(f"      Duration: {time.time() - t0:.2f}s\n✅ Analytical evaluation completed!")
    return psi_t