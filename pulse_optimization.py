"""
pulse_optimization.py
======================
Ausgelagerte Funktionen fuer die Floquet-Puls-Vorberechnung, die
Infidelity-Berechnung und den Parameter-Sweep via differential_evolution.
"""

from time import perf_counter

import numpy as np
import qutip as qt
import scipy.linalg as scipy_linalg
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import sympy as sy
from FAPT import *


# ------------------------------------------------------------------
# 1. Symbolische Vorbereitung (Einmalige Ausführung)
# ------------------------------------------------------------------

def prepare_floquet_functions(pulse_shape_builder, p_names, s, rH, rW, include_geometric=True, include_micromotion=True, include_g_correction=True, verbose=False):
    """Berechnet Heff, W, M(t) und M^{-1}(0) symbolisch und erzeugt NumPy-Lambdas."""

    t_sym = sy.Symbol("t", real=True)
    p_syms = [sy.Symbol(name, real=True) for name in p_names]

    if verbose:
        print("1/2 Erstelle symbolischen Puls und berechne W und Heff ...")

    A_expr, wd_expr = pulse_shape_builder(t_sym, p_syms)

    orig_np_exp = np.exp
    np.exp = sy.exp

    try:
        W = W_Floquet_elements_fast(rW, wd_expr, s.resonances, s.E_array, s.V1_dressed_array, A=A_expr, V0=None, analytics=True, verbose=verbose)

        Heff = Heff_Floquet_total_matrix_summed(rH, wd_expr, A_expr, s.resonances, s.E_array, s.V1_dressed_array, V0=None, ref_state=None, dwd=wd_expr.diff(t_sym), dA=A_expr.diff(t_sym), t=t_sym, analytics=True, rW=rW, include_geometric=include_geometric, include_micromotion=include_micromotion, include_g_correction=include_g_correction, W=W, verbose=verbose)

    finally:
        np.exp = orig_np_exp

    resonances = dict(s.resonances) if isinstance(s.resonances, list) else s.resonances

    D, d_res, kref = len(s.E_array), len(resonances), min(resonances)

    M_sym = sy.zeros(D, d_res)
    M_inv_0_sym = sy.zeros(d_res, D)

    for r in range(0, rW + 1):
        for l in range(D):
            for idx_a, a in enumerate(resonances):
                for p, (W_lp_a, _, _) in W[r, l, a].items():
                    W_lp_a_0 = W_lp_a.subs(t_sym, 0) if hasattr(W_lp_a, "subs") else W_lp_a
                    M_inv_0_sym[idx_a, l] += sy.conjugate(W_lp_a_0)
                    M_sym[l, idx_a] += sy.exp(-sy.I * (s.E_array[kref] + p * wd_expr) * t_sym) * W_lp_a

    if verbose:
        print("2/2 Erzeuge Lambdified H_func, M_func und M_inv_0_func ...")

    args_list = [t_sym] + p_syms

    H_func = sy.lambdify(args_list, sy.Matrix(Heff).doit(), modules=["numpy", "scipy"])
    M_func = sy.lambdify(args_list, M_sym.doit(), modules=["numpy", "scipy"])
    M_inv_0_func = sy.lambdify(p_syms, M_inv_0_sym.doit(), modules=["numpy", "scipy"])

    return H_func, M_func, M_inv_0_func

def compute_pulse_propagator(p_vals, H_func, M_func, M_inv_0_func, t_mids, dt, tg, s):
    """Berechnet U_eff, U_total und den vollständigen resonanten Block U_sub."""

    d_res = s.d_res
    D = s.D
    res_indices = list(s.resonances.keys())

    H_eval = np.asarray(H_func(t_mids, *p_vals), dtype=complex)
    expected_time_dependent_shape = (d_res, d_res, len(t_mids))
    expected_time_independent_shape = (d_res, d_res)

    if H_eval.shape == expected_time_independent_shape:
        H_all = np.repeat(H_eval[:, :, np.newaxis], len(t_mids), axis=2)

    elif H_eval.shape == expected_time_dependent_shape:
        H_all = H_eval

    else:
        raise ValueError(
            f"H_func liefert bei der Auswertung mit t_mids die Form {H_eval.shape}. "
            f"Erwartet wurde entweder {expected_time_independent_shape} für ein zeitunabhängiges "
            f"Hamiltonian oder {expected_time_dependent_shape} für ein zeitabhängiges Hamiltonian."
        )

    U_eff = np.eye(d_res, dtype=complex)

    for k in range(len(t_mids)):
        U_eff = scipy_linalg.expm(-1j * H_all[:, :, k] * dt) @ U_eff

    M_final = np.asarray(M_func(tg, *p_vals), dtype=complex).reshape(D, d_res)
    M_inv_0 = np.asarray(M_inv_0_func(*p_vals), dtype=complex).reshape(d_res, D)

    U_total = M_final @ U_eff @ M_inv_0

    # normalization
    U_columns = U_total[:, res_indices]
    column_norms = np.linalg.norm(U_columns, axis=0)
    if np.any(column_norms < 1e-14):
        raise ValueError(f"Mindestens eine propagierte Spalte besitzt praktisch Norm null: {column_norms}.")
    U_columns_normalized = U_columns / column_norms[np.newaxis, :]

    # project to resonant subspace image
    U_sub = U_columns_normalized[res_indices, :]

    return U_eff, U_total, U_sub

def compute_pulse_infidelity(p_vals, H_func, M_func, M_inv_0_func, t_mids, dt, tg, target_metric, s):
    d_res = s.d_res

    _, _, U_sub = compute_pulse_propagator(p_vals, H_func, M_func, M_inv_0_func, t_mids, dt, tg, s)

    if target_metric == "population":
        fidelity = 0.5 * (np.abs(U_sub[1, 0]) ** 2 + np.abs(U_sub[0, 1]) ** 2)

    elif target_metric == "process":
        U_target = np.asarray(s.U_ideal, dtype=complex)

        if U_sub.shape != U_target.shape:
            raise ValueError(f"Dimensionsfehler: U_sub hat Form {U_sub.shape}, U_ideal hat Form {U_target.shape}.")

        trace_val = np.trace(U_target.conj().T @ U_sub)
        fidelity = np.abs(trace_val) ** 2 / d_res ** 2

    else:
        raise ValueError(f"Unbekannte target_metric: {target_metric}")

    return 1.0 - float(np.real(fidelity))


def optimize_pulse_parameters(H_func, M_func, M_inv_0_func, param_ranges, s, tg, N_t=200, target_metric="population", verbose=False, popsize=15, maxiter=80, tol=1e-5, polish=True, mutation=(0.5, 1.0), recombination=0.7, seed=1, workers=1):
    tlist = np.linspace(0, tg, N_t)

    dt = tlist[1] - tlist[0]
    t_mids = tlist[:-1] + dt / 2.0

    p_names = list(param_ranges.keys())
    bounds = [(np.min(param_ranges[p]), np.max(param_ranges[p])) for p in p_names]

    n_calls = 0
    best_so_far = float("inf")
    t_start = perf_counter()

    if verbose:
        print("=" * 60)
        print("[DEBUG] Starte Puls-Optimierung")
        print(f"[DEBUG] Parameter ({len(p_names)}): {p_names}")
        print(f"[DEBUG] Hilbertraum D={s.D}, Reduzierter Raum d_res={s.d_res}, N_t={N_t}")
        print(f"[DEBUG] Popsize={popsize} -> {popsize * len(p_names)} Evall. pro Generation | Maxiter={maxiter} | Workers={workers}")
        print("=" * 60)

    def objective(x):
        nonlocal n_calls, best_so_far

        n_calls += 1

        val = compute_pulse_infidelity(tuple(x), H_func, M_func, M_inv_0_func, t_mids, dt, tg, target_metric,s)
        if val < best_so_far:
            best_so_far = val

        return val

    def callback(xk, convergence):
        """Callback pro Generation in differential_evolution."""

        if verbose:
            elapsed = perf_counter() - t_start
            print(f"[DEBUG Gen Callback] Aufrufe: {n_calls:6d} | Bisher beste Infidelity: {best_so_far:.6e} | Conv: {convergence:.4e} | Zeit: {elapsed:.2f}s")

    if verbose:
        print("[DEBUG] Starte differential_evolution...")

    result = differential_evolution(objective, bounds=bounds, strategy="best1bin", popsize=popsize, mutation=mutation, recombination=recombination, maxiter=maxiter, tol=tol, polish=False, updating="deferred", workers=workers, disp=verbose, seed=seed, callback=callback if verbose else None)

    # --- POLISH PHASE ---

    if polish:
        if verbose:
            print(f"\n[DEBUG] Haupt-Optimierung fertig nach {n_calls} Aufrufen. Starte L-BFGS-B Polish...")

        t_polish = perf_counter()
        n_calls_before_polish = n_calls

        res_polish = minimize(objective, result.x, method="L-BFGS-B", bounds=bounds, tol=tol)

        if res_polish.fun < result.fun:
            result = res_polish

        if verbose:
            print(f"[DEBUG] Polish abgeschlossen in {perf_counter() - t_polish:.2f}s (+{n_calls - n_calls_before_polish} Aufrufe)")

    best_p_vals = tuple(result.x)
    best_params = dict(zip(p_names, best_p_vals))
    best_infidelity = result.fun

    return best_params, best_infidelity

def refine_gate_time(best_p_pop, tg_target, wd_val, H_func, M_func, M_inv_0_func, s, steps_per_ns=2.0):
    """Refiniert die Gate-Zeit tg_opt ausgehend von tg_target.

    Das Suchintervall ist streng auf eine Periode T = 2*pi/wd zentriert,
    um die Rotationsphase der Process-Fidelity anzupassen.
    """

    T_period = (2.0 * np.pi) / wd_val
    p_vals_fixed = tuple(best_p_pop[k] for k in best_p_pop)

    def obj_tg(tg_cand):
        N_steps = max(100, int(np.ceil(tg_cand * steps_per_ns)))
        dt = tg_cand / float(N_steps)
        t_mids = (np.arange(N_steps) + 0.5) * dt

        return compute_pulse_infidelity(p_vals_fixed, H_func, M_func, M_inv_0_func, t_mids, dt, tg_cand, "process", s)

    bounds = (tg_target - T_period, tg_target)

    res_tg = minimize_scalar(obj_tg, bounds=bounds, method="bounded")

    return res_tg.x, res_tg.fun


def find_virtual_z_angles(p_vals, H_func, M_func, M_inv_0_func, tg, s, steps_per_ns=20):
    """Bestimmt die beiden Virtual-Z-Winkel aus der Floquet-Prozessmatrix."""

    N_steps = max(100, int(np.ceil(tg * steps_per_ns)))
    dt = tg / float(N_steps)
    t_mids = (np.arange(N_steps) + 0.5) * dt

    p_vals = tuple(p_vals.values()) if isinstance(p_vals, dict) else tuple(p_vals)

    res_keys = list(s.resonances.keys())
    ia = res_keys.index(s.state_a)
    ib = res_keys.index(s.state_b)

    remaining_indices = [i for i in range(s.d_res) if i not in (ia, ib)]

    if len(remaining_indices) != 1:
        raise ValueError(f"Der Couplerindex ist nicht eindeutig: ia={ia}, ib={ib}, d_res={s.d_res}.")

    ic = remaining_indices[0]
    U_target = np.asarray(s.U_ideal, dtype=complex)

    _, _, U_sub = compute_pulse_propagator(p_vals, H_func, M_func, M_inv_0_func, t_mids, dt, tg, s)

    if U_sub.shape != U_target.shape:
        raise ValueError(f"Dimensionsfehler: U_sub hat Form {U_sub.shape}, U_ideal hat Form {U_target.shape}.")

    z = np.sum(U_target.conj() * U_sub, axis=1)
    reference_phase = np.angle(z[ic]) if np.abs(z[ic]) > 1e-14 else np.angle(z[ia])

    theta_1 = reference_phase - np.angle(z[ia]) if np.abs(z[ia]) > 1e-14 else 0.0
    theta_2 = reference_phase - np.angle(z[ib]) if np.abs(z[ib]) > 1e-14 else 0.0

    theta_1 = np.angle(np.exp(1j * theta_1))
    theta_2 = np.angle(np.exp(1j * theta_2))

    return theta_1, theta_2

def find_and_verify_virtual_z(p_vals, H_func, M_func, M_inv_0_func, tg, s, verbose=True, steps_per_ns=20):
    """
    Bestimmt die virtuellen Z-Winkel für Qubit 1 und Qubit 2, welche die
    Process-Fidelity des vollständigen resonanten 3x3-Gates maximieren.

    Verwendete Konvention:

        Rz(theta) = exp(-1j * theta * sigma_z / 2)

    In der resonanten Basis (|100>, |001>, |010>) wirkt die Korrektur
    bis auf eine globale Phase als

        D_Z = diag(exp(1j*theta_1), exp(1j*theta_2), 1).

    Die virtuellen Z-Gates werden nach dem Puls angewendet:

        U_corrected = D_Z @ U_sub.
    """
    N_steps = max(100, int(np.ceil(tg * steps_per_ns)))
    dt = tg / float(N_steps)
    t_mids = (np.arange(N_steps) + 0.5) * dt

    p_vals = tuple(p_vals.values()) if isinstance(p_vals, dict) else tuple(p_vals)

    d_res = s.d_res
    U_target = np.asarray(s.U_ideal, dtype=complex)

    if d_res != 3:
        raise ValueError(f"Die Virtual-Z-Auswertung erwartet d_res=3, erhalten wurde d_res={d_res}.")

    _, _, U_sub = compute_pulse_propagator(p_vals, H_func, M_func, M_inv_0_func, t_mids, dt, tg, s)

    if U_sub.shape != U_target.shape:
        raise ValueError(f"Dimensionsfehler: U_sub hat Form {U_sub.shape}, U_ideal hat Form {U_target.shape}.")

    def process_fidelity(U):
        trace_val = np.trace(U_target.conj().T @ U)
        return float(np.real(np.abs(trace_val) ** 2 / d_res**2))

    # Beiträge der drei Ausgabezustände zum Process-Trace
    z = np.sum(U_target.conj() * U_sub, axis=1)
    z_a, z_b, z_c = z

    # Die Phasen von z_a und z_b an die Phase von z_c angleichen.
    # Falls z_c praktisch null ist, wird z_a als Referenz verwendet.
    reference_phase = np.angle(z_c) if np.abs(z_c) > 1e-14 else np.angle(z_a)

    theta_1 = reference_phase - np.angle(z_a) if np.abs(z_a) > 1e-14 else 0.0
    theta_2 = reference_phase - np.angle(z_b) if np.abs(z_b) > 1e-14 else 0.0

    # Winkel in das Intervall [-pi, pi) bringen
    theta_1 = (theta_1 + np.pi) % (2.0 * np.pi) - np.pi
    theta_2 = (theta_2 + np.pi) % (2.0 * np.pi) - np.pi

    D_Z = np.diag([np.exp(1j * theta_1), np.exp(1j * theta_2), 1.0])
    U_corrected = D_Z @ U_sub

    fidelity_before = process_fidelity(U_sub)
    fidelity_after = process_fidelity(U_corrected)

    # Unabhängige Verifikation
    verification_trace = np.trace(U_target.conj().T @ U_corrected)
    verification_fidelity = float(np.real(np.abs(verification_trace) ** 2 / d_res**2))
    verification_successful = np.isclose(fidelity_after, verification_fidelity, rtol=1e-12, atol=1e-12)

    result = {
        "theta_1_rad": theta_1,
        "theta_2_rad": theta_2,
        "theta_1_deg": np.rad2deg(theta_1),
        "theta_2_deg": np.rad2deg(theta_2),
        "D_Z": D_Z,
        "U_sub": U_sub,
        "U_corrected": U_corrected,
        "trace_contributions": z,
        "fidelity_before": fidelity_before,
        "fidelity_after": fidelity_after,
        "infidelity_before": 1.0 - fidelity_before,
        "infidelity_after": 1.0 - fidelity_after,
        "verification_fidelity": verification_fidelity,
        "verification_successful": verification_successful,
    }

    if verbose:
        print("=" * 65)
        print("Virtuelle Z-Korrektur")
        print("=" * 65)
        print(f"Qubit 1: theta_1 = {theta_1:+.10f} rad = {np.rad2deg(theta_1):+.6f} deg")
        print(f"Qubit 2: theta_2 = {theta_2:+.10f} rad = {np.rad2deg(theta_2):+.6f} deg")
        print(f"Process-Fidelity vorher:    {fidelity_before:.12f}")
        print(f"Process-Fidelity nachher:   {fidelity_after:.12f}")
        print(f"Verifizierte Fidelity:      {verification_fidelity:.12f}")
        print(f"Process-Infidelity nachher: {1.0 - fidelity_after:.6e}")
        print(f"Verifikation konsistent:    {verification_successful}")

        if fidelity_after < 0.99:
            print("Der verbleibende Fehler ist kein reiner, durch virtuelle Z-Gates korrigierbarer Phasenfehler.")

    return result