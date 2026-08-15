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

def V_from_Vtilde(V_posHarm, A=None, analytics=False):
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

    V_full = V_from_Vtilde(V_posHarm, A=A, analytics=analytics)

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