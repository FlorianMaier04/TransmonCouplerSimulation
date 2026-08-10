"""
pulse_optimization.py
======================
Ausgelagerte Funktionen fuer die Floquet-Puls-Vorberechnung, die
Infidelity-Berechnung und den Parameter-Sweep via differential_evolution.

Hinweis: `W_Floquet_elements_fast` und `Heff_Floquet_total_matrix_summed`
werden hier vorausgesetzt (importiere sie aus dem Modul, in dem sie
definiert sind) - passe den Import unten entsprechend an.
"""

from time import perf_counter

import numpy as np
import scipy.linalg as scipy_linalg
import sympy as sy
import qutip as qt
from scipy.optimize import differential_evolution, minimize

# TODO: Passe diesen Import an den tatsächlichen Speicherort der beiden
# Funktionen an (z. B. aus deinem Floquet-Kernmodul).
# from floquet_core import W_Floquet_elements_fast, Heff_Floquet_total_matrix_summed


# ------------------------------------------------------------------
# 1. Äußerer Code: Symbolische Vorberechnung (Einmalige Ausführung)
# ------------------------------------------------------------------
def prepare_floquet_functions(
    pulse_shape_builder, p_names, s, rH, rW,
    include_geometric=True, include_micromotion=True, include_g_correction=True, verbose=True
):
    """Berechnet Heff, W und M_sym symbolisch und erzeugt schnell auswertbare NumPy-Lambdas."""
    t_sym = sy.Symbol("t", real=True)
    p_syms = [sy.Symbol(name, real=True) for name in p_names]
    if verbose:
        print("1/2 Erstelle symbolischen Puls & berechne W und Heff ...")
    A_expr, wd_expr = pulse_shape_builder(t_sym, p_syms)
    orig_np_exp = np.exp
    np.exp = sy.exp
    try:
        W = W_Floquet_elements_fast(
            rW, wd_expr, s.resonances, s.E_array, s.V1_dressed_array,
            A=A_expr, V0=None, analytics=True, verbose=True
        )
        Heff = Heff_Floquet_total_matrix_summed(
            rH, wd_expr, A_expr, s.resonances, s.E_array, s.V1_dressed_array, V0=None, ref_state=None,
            dwd=wd_expr.diff(t_sym), dA=A_expr.diff(t_sym), t=t_sym, analytics=True, rW=rW,
            include_geometric=include_geometric, include_micromotion=include_micromotion,
            include_g_correction=include_g_correction, W=W, verbose=verbose
        )
    finally:
        np.exp = orig_np_exp

    resonances = dict(s.resonances) if isinstance(s.resonances, list) else s.resonances
    D, d_res, kref = len(s.E_array), len(resonances), min(resonances)

    M_sym = sy.zeros(D, d_res)
    for r in range(0, rW + 1):
        for l in range(D):
            for idx_a, a in enumerate(resonances):
                for p, (W_lp_a, _, _) in W[r, l, a].items():
                    M_sym[l, idx_a] += sy.exp(-sy.I * (s.E_array[kref] + p * wd_expr) * t_sym) * W_lp_a

    if verbose:
        print("2/2 Erzeuge Lambdified H_func und M_func ...")
    args_list = [t_sym] + p_syms
    return (
        sy.lambdify(args_list, sy.Matrix(Heff).doit(), modules=['numpy', 'scipy']),
        sy.lambdify(args_list, M_sym.doit(), modules=['numpy', 'scipy'])
    )


# ------------------------------------------------------------------
# 2. Hilfsfunktionen: Vektorisierte H-Auswertung & Infidelity
# ------------------------------------------------------------------
def _eval_H_stack(H_func, t_mids, p_vals, d_res):
    try:
        H_all = np.asarray(H_func(t_mids, *p_vals), dtype=complex)
        if H_all.shape == (d_res, d_res, len(t_mids)):
            return H_all
    except Exception:
        pass
    H_all = np.empty((d_res, d_res, len(t_mids)), dtype=complex)
    for k, t_mid in enumerate(t_mids):
        H_all[:, :, k] = np.array(H_func(t_mid, *p_vals), dtype=complex).reshape(d_res, d_res)
    return H_all


def compute_pulse_infidelity(
    p_vals, H_func, M_func, t_mids, dt, tg, D, d_res, res_indices, ind_a, ind_b, state_a, state_b, target_metric, qobj_ideal=None
):
    H_all = _eval_H_stack(H_func, t_mids, p_vals, d_res)

    U_eff = np.eye(d_res, dtype=complex)
    for k in range(len(t_mids)):
        U_eff = scipy_linalg.expm(-1j * H_all[:, :, k] * dt) @ U_eff

    M_final = np.array(M_func(tg, *p_vals), dtype=complex).reshape(D, d_res)
    U_total = M_final @ U_eff
    norms = np.linalg.norm(U_total, axis=0)
    U_total /= norms

    if target_metric == "population":
        fidelity = 0.5 * (np.abs(U_total[state_b, ind_a])**2 + np.abs(U_total[state_a, ind_b])**2)
    elif target_metric == "process":
        fidelity = qt.process_fidelity(qt.Qobj(U_total[res_indices, :]), qobj_ideal)

    return 1.0 - fidelity


# ------------------------------------------------------------------
# 3. Entschlackte Parameter-Sweep Methode
# ------------------------------------------------------------------
def optimize_pulse_parameters(
    H_func,
    M_func,
    param_ranges,
    s,
    tg,
    N_t=200,
    target_metric="population",
    verbose=True,
    popsize=15,
    maxiter=80,
    tol=1e-5,
    polish=True,
    mutation=(0.5, 1.0),
    recombination=0.7,
    seed=1,
    workers=1,
):
    tlist = np.linspace(0, tg, N_t)
    dt, t_mids = tlist[1] - tlist[0], tlist[:-1] + (tlist[1] - tlist[0]) / 2.0

    p_names = list(param_ranges.keys())
    bounds = [(np.min(param_ranges[p]), np.max(param_ranges[p])) for p in p_names]

    resonances = (
        dict(s.resonances) if isinstance(s.resonances, list) else s.resonances
    )
    D, d_res, res_indices = len(s.E_array), len(resonances), list(resonances.keys())
    ind_a, ind_b = res_indices.index(s.state_a), res_indices.index(s.state_b)

    qobj_ideal = None
    if target_metric == "process":
        U_ideal = np.eye(d_res, dtype=complex)
        if d_res >= 2:
            U_ideal[ind_a, ind_b], U_ideal[ind_b, ind_a] = 1j, 1j
            U_ideal[ind_a, ind_a], U_ideal[ind_b, ind_b] = 0, 0
        qobj_ideal = qt.Qobj(U_ideal)

    n_calls = 0
    best_so_far = float("inf")
    t_start = perf_counter()

    # --- DEBUG 1: SETUP ---
    print("=" * 60)
    print("[DEBUG] Starte Puls-Optimierung")
    print(f"[DEBUG] Parameter ({len(p_names)}): {p_names}")
    print(
        f"[DEBUG] Hilbertraum D={D}, Reduzierter Raum d_res={d_res}, N_t={N_t}"
    )
    print(
        f"[DEBUG] Popsize={popsize} -> {popsize * len(p_names)} Evall. pro"
        f" Generation | Maxiter={maxiter} | Workers={workers}"
    )
    print("=" * 60)

    def objective(x):
        nonlocal n_calls, best_so_far
        n_calls += 1
        val = compute_pulse_infidelity(
            tuple(x),
            H_func,
            M_func,
            t_mids,
            dt,
            tg,
            D,
            d_res,
            res_indices,
            ind_a,
            ind_b,
            s.state_a,
            s.state_b,
            target_metric,
            qobj_ideal,
        )

        if val < best_so_far:
            best_so_far = val

        return val

    def callback(xk, convergence):
        """Callback pro Generation in differential_evolution."""
        elapsed = perf_counter() - t_start
        print(
            f"[DEBUG Gen Callback] Aufrufe: {n_calls:6d} | Bisher beste Infidelity:"
            f" {best_so_far:.6e} | Conv: {convergence:.4e} | Zeit: {elapsed:.2f}s"
        )

    print("[DEBUG] Starte differential_evolution...")
    result = differential_evolution(
        objective,
        bounds=bounds,
        strategy="best1bin",
        popsize=popsize,
        mutation=mutation,
        recombination=recombination,
        maxiter=maxiter,
        tol=tol,
        polish=False,  # Polish wird manuell getrackt, siehe unten
        updating="deferred",
        workers=workers,
        disp=verbose,
        seed=seed,
        callback=callback,
    )

    # --- DEBUG 2: POLISH PHASE ---
    if polish:
        print(
            f"\n[DEBUG] Haupt-Optimierung fertig nach {n_calls} Aufrufen. Starte"
            f" L-BFGS-B Polish (kann bei vielen Parametern hängen)..."
        )
        t_polish = perf_counter()
        n_calls_before_polish = n_calls

        res_polish = minimize(
            objective,
            result.x,
            method="L-BFGS-B",
            bounds=bounds,
            tol=tol,
        )
        if res_polish.fun < result.fun:
            result = res_polish

        print(
            f"[DEBUG] Polish abgeschlossen in"
            f" {perf_counter() - t_polish:.2f}s (+{n_calls - n_calls_before_polish} Aufrufe)"
        )

    best_p_vals = tuple(result.x)
    best_params, best_infidelity = dict(zip(p_names, best_p_vals)), result.fun

    # --- DEBUG 3: FINAL DYNAMICS ---
    print("\n[DEBUG] Berechne finale Trajektorie (best_psi_t)...")
    t_sim = perf_counter()

    c_eff, best_psi_t = np.zeros(d_res, dtype=complex), np.zeros(
        (D, N_t), dtype=complex
    )
    c_eff[ind_a] = 1

    for i, t in enumerate(tlist):
        best_psi_t[:, i] = (
            np.asarray(M_func(t, *best_p_vals), dtype=complex).reshape(D, d_res)
            @ c_eff
        )
        if i != N_t - 1:
            H = np.asarray(H_func(t + dt / 2, *best_p_vals), dtype=complex).reshape(
                d_res, d_res
            )
            c_eff = scipy_linalg.expm(-1j * H * dt) @ c_eff

    print(
        f"[DEBUG] Finale Trajektorie fertig in {perf_counter() - t_sim:.2f}s."
        f" Gesamtdauer: {perf_counter() - t_start:.2f}s"
    )
    print("=" * 60)

    if verbose:
        print(
            f"\nFunktionsaufrufe : {n_calls}\nInfidelity      "
            f" : {best_infidelity:.6e}\nBeste Parameter:"
        )
        for k, v in best_params.items():
            print(f"  {k:15s}: {v:.10g}")

    return best_params, best_infidelity, best_psi_t, tlist