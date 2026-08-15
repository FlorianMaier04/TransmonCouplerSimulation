"""
pulse_optimization.py
=====================
Allgemeine Funktionen fuer die Floquet-Puls-Vorbereitung, Propagation,
Fidelity-Auswertung und Pulsoptimierung.
"""

from time import perf_counter

import numpy as np
import scipy.linalg as scipy_linalg
from scipy.optimize import differential_evolution, minimize, minimize_scalar
import sympy as sy
from FAPT import *
from utils import * 

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

    D, d_res, kref = len(s.E_array), len(s.resonances), min(s.resonances)

    M_sym = sy.zeros(D, d_res)
    M_inv_0_sym = sy.zeros(d_res, D)

    for r in range(0, rW + 1):
        for l in range(D):
            for idx_a, a in enumerate(s.resonances):
                for p, (W_lp_a, _, _) in W[r, l, a].items():
                    W_lp_a_0 = W_lp_a.subs(t_sym, 0) if hasattr(W_lp_a, "subs") else W_lp_a
                    M_inv_0_sym[idx_a, l] += sy.conjugate(W_lp_a_0)
                    M_sym[l, idx_a] += sy.exp(-sy.I * (s.E_array[kref] + p * wd_expr) * t_sym) * W_lp_a

    if verbose:
        print("2/2 Erzeuge Lambdified H_func, M_func und M_inv_0_func ...")

    args_list = [t_sym] + p_syms

    Heff_sym = sy.Matrix(Heff).doit().applyfunc(sy.factor_terms)
    M_sym = M_sym.doit().applyfunc(sy.factor_terms)
    M_inv_0_sym = M_inv_0_sym.doit().applyfunc(sy.factor_terms)

    if verbose:
        print(f"Operationen in Heff vor CSE: {sy.count_ops(Heff_sym)}")

    H_func = sy.lambdify(args_list, Heff_sym, modules=["numpy", "scipy"], cse=True, docstring_limit=0)
    M_func = sy.lambdify(args_list, M_sym, modules=["numpy", "scipy"], cse=True, docstring_limit=0)
    M_inv_0_func = sy.lambdify(p_syms, M_inv_0_sym, modules=["numpy", "scipy"], cse=True, docstring_limit=0)

    H_element_funcs = [
        [
            sy.lambdify(args_list, Heff_sym[i, j], modules=["numpy", "scipy"], cse=True, docstring_limit=0)
            for j in range(d_res)
        ]
        for i in range(d_res)
    ]

    def evaluate_stack(t_values, *p_vals):
        t_values = np.asarray(t_values, dtype=float)
        H_all = np.empty((d_res, d_res, t_values.size), dtype=complex)

        for i in range(d_res):
            for j in range(d_res):
                values = np.asarray(H_element_funcs[i][j](t_values, *p_vals), dtype=complex)
                H_all[i, j, :] = np.broadcast_to(values, t_values.shape)

        return H_all

    H_func.evaluate_stack = evaluate_stack
    H_func.is_time_dependent = t_sym in Heff_sym.free_symbols    
    return H_func, M_func, M_inv_0_func

def compute_propagator_floquet(p_vals, H_func, M_func, M_inv_0_func, t_mids, dt, tg, s):
    """Berechnet den spaltenweise normierten Propagator im resonanten Unterraum."""

    p_vals = tuple(p_vals)
    d_res = s.d_res
    D = s.D
    if getattr(H_func, "is_time_dependent", True):
        if hasattr(H_func, "evaluate_stack"):
            H_all = H_func.evaluate_stack(t_mids, *p_vals)
        else:
            H_all = np.stack([np.asarray(H_func(t_mid, *p_vals), dtype=complex).reshape(d_res, d_res) for t_mid in t_mids], axis=2)

        U_eff = np.eye(d_res, dtype=complex)

        for k in range(len(t_mids)):
            U_eff = scipy_linalg.expm(-1j * H_all[:, :, k] * dt) @ U_eff
    else:
        H_static = np.asarray(H_func(t_mids[0], *p_vals), dtype=complex).reshape(d_res, d_res)
        U_eff = scipy_linalg.expm(-1j * H_static * len(t_mids) * dt)

    M_final = np.asarray(M_func(tg, *p_vals), dtype=complex).reshape(D, d_res)
    M_inv_0 = np.asarray(M_inv_0_func(*p_vals), dtype=complex).reshape(d_res, D)
    U_total = M_final @ U_eff @ M_inv_0

    # Jede relevante Spalte wird vor der Projektion im vollen D-dimensionalen
    # Raum normiert. Dadurch bleibt Leakage aus dem Resonanzraum sichtbar.
    U_columns = U_total[:, list(s.resonances.keys())]
    column_norms = np.linalg.norm(U_columns, axis=0)

    if np.any(column_norms < 1e-14):
        raise ValueError(f"Mindestens eine propagierte Spalte besitzt praktisch Norm null: {column_norms}.")

    U_columns_normalized = U_columns / column_norms[np.newaxis, :]
    U_sub = U_columns_normalized[list(s.resonances.keys()), :]
    return U_sub

def correct_phase_virtual(U):
    theta = np.angle(U[1,2]) # assuming theta1 and theta2 to be equal
    gamma_0 = np.angle(U[0,0])
    delta = theta - gamma_0 - np.pi/2
    phases = np.array([0,delta,delta,2*delta], dtype=complex)
    D = np.diag(np.exp(-1j*phases))
    return np.exp(-1j*gamma_0) * D @ U # apply virtual z and global shift

def correct_phase_zz(U):
    theta = np.angle(U[1,2]) # assuming theta1 and theta2 to be equal
    gamma_0 = np.angle(U[0,0])
    gamma_e = np.angle(U[3,3])
    delta = theta - gamma_0 - np.pi/2
    # hardcoding the condition on phase matching
    gamma_e_prime = 2*theta-np.pi-gamma_0
    delta_gamma_e = gamma_e_prime - gamma_e
    phases = np.array([0, 0,0,delta_gamma_e], dtype=complex)
    D = np.diag(np.exp(1j*phases))
    return D @ U # apply virtual z and global shift


def infid_process_direct(U, s, correct_zz=False):
    if(correct_zz):
        U = correct_phase_zz(np.array(U, copy=True, dtype=complex))
    U = correct_phase_virtual(U)

    trace_val = np.trace(s.U_ideal.conj().T @ U)
    return 1 - np.abs(trace_val) ** 2 / 4**2

def infid_process_floq(U, s):
    '''
    As floquet Theory can't predict the phase of the computational states 000 and 101 gamma_0 and gamma_e are assumed 0
    '''
    U_comp = np.array([
        [1, 0,        0,        0],
        [0, U[0, 0],  U[0, 1],  0],
        [0, U[1, 0],  U[1, 1],  0],
        [0, 0,        0,        1]], dtype=complex)
    U_comp = correct_phase_virtual(U_comp)
    trace_val = np.trace(s.U_ideal.conj().T @ U_comp)
    return 1 - np.abs(trace_val) ** 2 / 4**2

def infid_population(U):
    direct = U.shape[0]==4 # direct simulation is allreaddy done in the computational subspace
    if not direct:
        U_comp = np.array([
            [1, 0,        0,        0],
            [0, U[0, 0],  U[0, 1],  0],
            [0, U[1, 0],  U[1, 1],  0],
            [0, 0,        0,        1]], dtype=complex, copy=True)
    else: U_comp = np.array(U, dtype=complex, copy=True)
    U_comp = correct_phase_virtual(U_comp)
    return 1 - 1/4 * (np.abs(U_comp[2, 1]) ** 2 + np.abs(U_comp[1, 2]) ** 2 
                      + np.abs(U_comp[0, 0]) ** 2 + np.abs(U_comp[3, 3]) ** 2)

# ------------------------------------------------------------------
# 3. Allgemeine Parameteroptimierung
# ------------------------------------------------------------------

def optimize_pulse_parameters(H_func, M_func, M_inv_0_func, param_ranges, s, tg, N_t=200, verbose=False, popsize=17, maxiter=80, tol=1e-8, polish=True, mutation=(0.5, 1.0), recombination=0.7, seed=1, workers=1, x0=np.array([0,1])):
    tlist = np.linspace(0.0, tg, N_t)
    dt = tlist[1] - tlist[0]
    t_mids = tlist[:-1] + dt / 2.0

    p_names = list(param_ranges.keys())
    bounds = [(np.min(param_ranges[p]), np.max(param_ranges[p])) for p in p_names]

    n_calls = 0
    best_so_far = float("inf")
    t_start = perf_counter()
    
    def objective(x):
        nonlocal n_calls, best_so_far
        n_calls += 1
        U_propagated = compute_propagator_floquet(tuple(x), H_func, M_func, M_inv_0_func, t_mids, dt, tg, s)
        value = infid_population(U_propagated)
        best_so_far = min(best_so_far, value)
        return value

    def callback(xk, convergence):
        if verbose:
            elapsed = perf_counter() - t_start
            print(f"[DEBUG Gen Callback] Aufrufe: {n_calls:6d} | Beste Infidelity: {best_so_far:.6e} | Conv: {convergence:.4e} | Zeit: {elapsed:.2f}s")
    de_kwargs = {"func": objective, "bounds": bounds, "strategy": "best1bin", "popsize": popsize, "mutation": mutation, "recombination": recombination, "maxiter": maxiter, "tol": tol, "polish": False, "updating": "deferred", "workers": workers, "disp": verbose, "seed": seed, "callback": callback if verbose else None}
    if x0 is not None:
        de_kwargs["x0"] = np.asarray(x0, dtype=float)
    result = differential_evolution(**de_kwargs)
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

    best_params = dict(zip(p_names, tuple(result.x)))

    return best_params, float(result.fun)

def optimize_iswap_at_gate_time_cos(tg_target, s, param_ranges, H_func, M_func, M_inv_0_func, pts_per_ns=40.0, N_t=300, popsize=8, maxiter=50, tol=1e-8, polish=True, verbose=False):
    """Optimiert den Populationstransfer und korrigiert anschließend per Virtual Z oder Gatezeit."""
    cos_wd, cos_amp = compute_cos_params(s,tg_target)
    best_params, infid_floq_pop = optimize_pulse_parameters(H_func, M_func, M_inv_0_func, param_ranges, s, tg_target, N_t=N_t, verbose=verbose, popsize=popsize, maxiter=maxiter, tol=tol, polish=polish, x0=None)

    p_vals = tuple(best_params[name] for name in list(param_ranges.keys()))
    wd = cos_wd + best_params["wd_offset"]
    amp = cos_amp * best_params["amp_scale"]

    tg_final, infid_floq_proc = refine_gate_time_phase_aligned(p_vals, tg_target, wd, s, H_func, M_func, M_inv_0_func)
    q_sig = lambda t, args=None: amp * np.cos(wd * t)
    n_pts_mes = max(101, int(np.ceil(tg_final * pts_per_ns)))
    U_direct = np.zeros((4, 4), dtype=complex)
    for j in range(4):
        result = sesolve([s.H0, [s.V1, q_sig]], s.E_states[s.comp_indices[j]], np.linspace(0.0, tg_final, n_pts_mes), e_ops=[])
        amplitudes = np.array([[state.overlap(psi_t) for psi_t in result.states] for state in s.E_states[s.comp_indices]])
        U_direct[:, j] = amplitudes[:, -1]
    infid_direct_pop = infid_population(U_direct)
    infid_direct_proc = infid_process_direct(U_direct,s, correct_zz=False)
    infid_direct_proc_zz_corrected = infid_process_direct(U_direct,s, correct_zz=True)
    print(
        f"tg_final: {tg_final:.4f} ns | "
        f"infid_floq_pop: {infid_floq_pop:.4e} | "
        f"infid_floq_proc: {infid_floq_proc:.4e} | "
        f"infid_direct_pop: {infid_direct_pop:.4e} | "
        f"infid_direct_proc: {infid_direct_proc:.4e}")
    return {"infid_direct_pop":infid_direct_pop, "infid_direct_proc":infid_direct_proc, "infid_floq_proc":infid_floq_proc, "infid_floq_pop":infid_floq_pop, "tg_final":tg_final, " infid_direct_proc_zzcorrected": infid_direct_proc_zz_corrected}

def refine_gate_time_phase_aligned(p_vals, tg_target, wd, s, H_func, M_func, M_inv_0_func, correct_zz = False, steps_per_ns=30.0, grid_points=401):
    """Sucht in [tg_target - 2*pi/|wd|, tg_target] die beste phasenangepasste Process-Fidelity."""
    T_period = 2*2.0 * np.pi / wd
    bounds = (tg_target - T_period, tg_target + T_period)
    N_steps = max(100, int(steps_per_ns*bounds[1]))
    def objective(tg):
        dt_current = tg / float(N_steps)
        t_mids_current = (np.arange(N_steps) + 0.5) * dt_current        
        U = compute_propagator_floquet(p_vals, H_func, M_func, M_inv_0_func, t_mids_current, dt_current, tg, s)
        infid = infid_process_floq(U, s)
        return infid

    tg_grid = np.linspace(bounds[0], bounds[1], grid_points)
    infidelity_grid = np.asarray([objective(tg) for tg in tg_grid])
    best_index = int(np.argmin(infidelity_grid))

    left = tg_grid[max(0, best_index - 1)]
    right = tg_grid[min(grid_points - 1, best_index + 1)]

    if left == right:
        tg_opt = float(tg_grid[best_index])
    else:
        result = minimize_scalar(objective, bounds=(left, right), method="bounded")
        tg_opt = float(result.x)

    return tg_opt, objective(tg_opt)