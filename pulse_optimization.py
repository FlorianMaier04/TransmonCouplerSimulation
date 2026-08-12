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
import cmath

# ------------------------------------------------------------------
# 1. Symbolische Vorbereitung (einmalige Ausfuehrung)
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

    H_func.is_time_dependent = t_sym in Heff_sym.free_symbols

    return H_func, M_func, M_inv_0_func


# ------------------------------------------------------------------
# 2. Propagation und allgemeine Fidelity-Auswertung
# ------------------------------------------------------------------

def compute_pulse_propagator(p_vals, H_func, M_func, M_inv_0_func, t_mids, dt, tg, s):
    """Berechnet den spaltenweise normierten Propagator im resonanten Unterraum."""

    p_vals = tuple(p_vals)
    d_res = s.d_res
    D = s.D
    res_indices = list(s.resonances.keys())

    if getattr(H_func, "is_time_dependent", True):
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
    U_columns = U_total[:, res_indices]
    column_norms = np.linalg.norm(U_columns, axis=0)

    if np.any(column_norms < 1e-14):
        raise ValueError(f"Mindestens eine propagierte Spalte besitzt praktisch Norm null: {column_norms}.")

    U_columns_normalized = U_columns / column_norms[np.newaxis, :]
    U_sub = U_columns_normalized[res_indices, :]

    return U_sub

def compute_gate_infidelity(U, U_target=None, metric="process", swap_indices=(0, 1)):
    """Berechnet Population- oder Process-Infidelity aus einer vorhandenen Prozessmatrix."""

    U = np.asarray(U, dtype=complex)

    if metric == "population":
        ia, ib = swap_indices
        fidelity = 0.5 * (np.abs(U[ib, ia]) ** 2 + np.abs(U[ia, ib]) ** 2)

    elif metric == "process":
        if U_target is None:
            raise ValueError("Fuer die Process-Fidelity muss U_target angegeben werden.")

        U_target = np.asarray(U_target, dtype=complex)

        if U.shape != U_target.shape:
            raise ValueError(f"Dimensionsfehler: U hat Form {U.shape}, U_target hat Form {U_target.shape}.")

        d = U_target.shape[0]
        trace_val = np.trace(U_target.conj().T @ U)
        fidelity = np.abs(trace_val) ** 2 / d**2

    else:
        raise ValueError(f"Unbekannte Fidelity-Metrik: {metric}")

    return 1.0 - float(np.real(fidelity))

def compute_phase_aligned_process_infidelity(U, U_ideal, phase_index=(0, 1), spectator_index=2):
    """Passt eine globale Phase sowie die beliebige Zuschauerphase an und berechnet die Process-Infidelity."""

    U_aligned = np.asarray(U, dtype=complex).copy()
    U_target = np.asarray(U_ideal, dtype=complex).copy()

    i, j = phase_index

    global_phase = np.angle(U_target[i, j]) - np.angle(U_aligned[i, j])
    U_aligned *= np.exp(1j * global_phase)

    gamma = np.angle(U_aligned[spectator_index, spectator_index])
    U_target[spectator_index, spectator_index] = np.exp(1j * gamma)

    infidelity = compute_gate_infidelity(U_aligned, U_target, metric="process")

    return infidelity, U_aligned, U_target, global_phase

def compute_virtual_z_correction(U, U_ideal, ia=0, ib=1):
    """Bestimmt die eine nach Entfernung der globalen Phase verbleibende Virtual-Z-Korrektur."""

    global_phase = np.angle(U_ideal[ia, ib]) - np.angle(U[ia, ib])
    theta_z = np.angle(U_ideal[ib, ia]) - np.angle(U[ib, ia]) - global_phase
    theta_z = np.angle(np.exp(1j * theta_z))

    phases = np.zeros(U.shape[0])
    phases[ib] = theta_z
    D_Z = np.diag(np.exp(1j * phases))

    return theta_z, D_Z

# ------------------------------------------------------------------
# 3. Allgemeine Parameteroptimierung
# ------------------------------------------------------------------

def optimize_pulse_parameters(H_func, M_func, M_inv_0_func, param_ranges, s, tg, N_t=200, target_metric="population", U_target=None, swap_indices=(0, 1), verbose=False, popsize=15, maxiter=80, tol=1e-5, polish=True, mutation=(0.5, 1.0), recombination=0.7, seed=1, workers=1, x0=None):
    """Optimiert Pulsparameter mittels differential_evolution und optionalem L-BFGS-B-Polish."""

    tlist = np.linspace(0.0, tg, N_t)
    dt = tlist[1] - tlist[0]
    t_mids = tlist[:-1] + dt / 2.0

    p_names = list(param_ranges.keys())
    bounds = [(np.min(param_ranges[p]), np.max(param_ranges[p])) for p in p_names]
    U_target = np.asarray(s.U_ideal if U_target is None else U_target, dtype=complex)

    n_calls = 0
    best_so_far = float("inf")
    t_start = perf_counter()

    if verbose:
        print("=" * 60)
        print("[DEBUG] Starte Puls-Optimierung")
        print(f"[DEBUG] Parameter ({len(p_names)}): {p_names}")
        print(f"[DEBUG] Hilbertraum D={s.D}, Resonanzraum d_res={s.d_res}, N_t={N_t}")
        print(f"[DEBUG] Popsize={popsize} | Maxiter={maxiter} | Workers={workers}")
        print("=" * 60)

    def objective(x):
        nonlocal n_calls, best_so_far

        n_calls += 1
        U_sub = compute_pulse_propagator(tuple(x), H_func, M_func, M_inv_0_func, t_mids, dt, tg, s)
        value = compute_gate_infidelity(U_sub, U_target, metric=target_metric, swap_indices=swap_indices)
        best_so_far = min(best_so_far, value)

        return value

    def callback(xk, convergence):
        if verbose:
            elapsed = perf_counter() - t_start
            print(f"[DEBUG Gen Callback] Aufrufe: {n_calls:6d} | Beste Infidelity: {best_so_far:.6e} | Conv: {convergence:.4e} | Zeit: {elapsed:.2f}s")

    if verbose:
        print("[DEBUG] Starte differential_evolution...")

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


# ------------------------------------------------------------------
# 4. Vollstaendiger iSWAP-Ablauf fuer eine Gatezeit
# ------------------------------------------------------------------

def _bind_tg_fixed_floquet_functions(tg_target, cos_wd, cos_amp, param_names, H_base, M_base, M_inv_0_base):
    """Bindet tg und die numerischen Cosinus-Basisparameter an die Basisfunktionen."""

    if set(param_names) != {"wd_offset", "amp_offset"}:
        raise ValueError("param_ranges muss genau die Parameter 'wd_offset' und 'amp_offset' enthalten.")

    def absolute_parameters(offsets):
        values = dict(zip(param_names, offsets))
        return cos_wd + values["wd_offset"], cos_amp + values["amp_offset"]

    def H_func(t, *offsets):
        wd, amp = absolute_parameters(offsets)
        return H_base(t, tg_target, wd, amp)

    def M_func(t, *offsets):
        wd, amp = absolute_parameters(offsets)
        return M_base(t, tg_target, wd, amp)

    def M_inv_0_func(*offsets):
        wd, amp = absolute_parameters(offsets)
        return M_inv_0_base(tg_target, wd, amp)

    H_func.is_time_dependent = getattr(H_base, "is_time_dependent", True)

    return H_func, M_func, M_inv_0_func



def optimize_iswap_at_gate_time(tg_target, s, param_ranges, H_base, M_base, M_inv_0_base, compute_cos_params_func, numerical_propagator, correction_mode="virtual_z", pts_per_ns=40.0, steps_per_ns=20.0, N_t=300, popsize=8, maxiter=50, tol=1e-8, polish=True, verbose=False):
    """Optimiert den Populationstransfer und korrigiert anschließend per Virtual Z oder Gatezeit."""

    cos_wd, cos_amp = compute_cos_params_func(s, tg_target)
    param_names = list(param_ranges.keys())
    res_keys = list(s.resonances.keys())
    d_res = len(res_keys)

    if d_res != 3:
        raise ValueError(f"Fuer diese Auswertung wird ein dreidimensionaler Resonanzraum erwartet, erhalten wurde d_res={d_res}.")

    ia = res_keys.index(s.state_a)
    ib = res_keys.index(s.state_b)
    remaining_indices = [i for i in range(d_res) if i not in (ia, ib)]

    if len(remaining_indices) != 1:
        raise ValueError(f"Der Referenzindex ist nicht eindeutig: ia={ia}, ib={ib}, d_res={d_res}.")

    ic = remaining_indices[0]
    U_ideal = np.array(s.U_ideal, dtype=complex, copy=True)
    H_func, M_func, M_inv_0_func = _bind_tg_fixed_floquet_functions(tg_target, cos_wd, cos_amp, param_names, H_base, M_base, M_inv_0_base)

    best_params, floq_pop = optimize_pulse_parameters(H_func, M_func, M_inv_0_func, param_ranges, s, tg_target, N_t=N_t, target_metric="population", U_target=U_ideal, swap_indices=(ia, ib), verbose=verbose, popsize=popsize, maxiter=maxiter, tol=tol, polish=polish, x0=np.zeros(len(param_names)))

    p_vals = tuple(best_params[name] for name in param_names)
    wd = cos_wd + best_params["wd_offset"]
    amp = cos_amp + best_params["amp_offset"]

    tg_final = tg_target
    N_steps = max(100, int(np.ceil(tg_final * steps_per_ns)))
    dt = tg_final / float(N_steps)
    t_mids = (np.arange(N_steps) + 0.5) * dt

    if correction_mode == "virtual_z":
        U_floq = compute_pulse_propagator(p_vals, H_func, M_func, M_inv_0_func, t_mids, dt, tg_final, s)
        _, D_Z = compute_virtual_z_correction(U_floq, U_ideal, ia=ia, ib=ib)
        floq_proc, U_floq_corrected, U_ideal_floq_corrected, _ = compute_phase_aligned_process_infidelity(D_Z @ U_floq, U_ideal, phase_index=(ia, ib), spectator_index=ic)
    elif correction_mode == "gate_time":
        tg_final, floq_proc, U_floq_corrected, U_ideal_floq_corrected, _ = refine_gate_time_phase_aligned(p_vals, tg_target, wd, H_func, M_func, M_inv_0_func, s, t_mids, dt, phase_index=(ia, ib), spectator_index=ic, steps_per_ns=steps_per_ns)
    else:
        raise ValueError("correction_mode muss 'virtual_z' oder 'gate_time' sein.")
    
    q_sig = lambda t, args=None: amp * np.cos(wd * t)
    n_pts_mes = max(101, int(np.ceil(tg_final * pts_per_ns)))
    t_mes = np.linspace(0.0, tg_final, n_pts_mes)
    U_mes = np.zeros((d_res, d_res), dtype=complex)

    for j in range(d_res):
        amplitudes = numerical_propagator([s.H0, [s.V1, q_sig]], t_mes, j, debug=False, s=s)
        U_mes[:, j] = amplitudes[:, -1]

    mes_pop = compute_gate_infidelity(U_mes, U_ideal, metric="population")
    mes_proc, U_mes_corrected, U_ideal_mes_corrected, _  = compute_phase_aligned_process_infidelity(D_Z @ U_mes if correction_mode=="virtual_z" else U_mes, U_ideal)

    floq_logical_proc, floq_leakage_proc, floq_coherent_proc = compute_logical_error_decomposition(U_floq_corrected, U_ideal_floq_corrected, logical_indices=(ia, ib))
    mes_logical_proc, mes_leakage_proc, mes_coherent_proc = compute_logical_error_decomposition(U_mes_corrected, U_ideal_mes_corrected, logical_indices=(ia, ib))
    return {
        "tg": tg_target,
        "tg_final": tg_final,
        "floq_pop": floq_pop,
        "mes_pop": mes_pop,
        "floq_proc": floq_proc,
        "mes_proc": mes_proc,
        "floq_logical_proc": floq_logical_proc,
        "mes_logical_proc": mes_logical_proc,
        "floq_leakage_proc": floq_leakage_proc,
        "mes_leakage_proc": mes_leakage_proc,
        "floq_coherent_proc": floq_coherent_proc,
        "mes_coherent_proc": mes_coherent_proc,
        "wd_offset": best_params["wd_offset"],
        "amp_offset": best_params["amp_offset"],
    }

def refine_gate_time_phase_aligned(p_vals, tg_target, wd, H_func, M_func, M_inv_0_func, s, t_mids=None, dt=None, phase_index=(0, 1), spectator_index=2, steps_per_ns=20.0, grid_points=401, function_factory=None):
    """Sucht in [tg_target - 2*pi/|wd|, tg_target] die beste phasenangepasste Process-Fidelity."""

    U_ideal = np.asarray(s.U_ideal, dtype=complex)
    T_period = 2.0 * np.pi / np.abs(wd)
    bounds = (max(1e-12, tg_target - T_period), tg_target)

    def evaluate(tg):
        H_current, M_current, M_inv_0_current = function_factory(tg) if function_factory is not None else (H_func, M_func, M_inv_0_func)

        N_steps = max(100, int(np.ceil(tg * steps_per_ns)))
        dt_current = tg / float(N_steps)
        t_mids_current = (np.arange(N_steps) + 0.5) * dt_current

        U = compute_pulse_propagator(p_vals, H_current, M_current, M_inv_0_current, t_mids_current, dt_current, tg, s)
        infidelity, U_aligned, U_ideal_aligned, global_phase = compute_phase_aligned_process_infidelity(U, U_ideal, phase_index=phase_index, spectator_index=spectator_index)

        return infidelity, U_aligned, U_ideal_aligned, global_phase

    def objective(tg):
        return evaluate(tg)[0]

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

    infidelity_opt, U_opt_aligned, U_ideal_aligned, global_phase = evaluate(tg_opt)

    return tg_opt, infidelity_opt, U_opt_aligned, U_ideal_aligned, global_phase

def compute_logical_error_decomposition(U, U_ideal, logical_indices=(0, 1)):
    """Zerlegt die logische Process-Infidelity additiv in Leakage und kohärenten Fehler."""

    logical_indices = list(logical_indices)

    K = np.asarray(U, dtype=complex)[np.ix_(logical_indices, logical_indices)]
    U_target = np.asarray(U_ideal, dtype=complex)[np.ix_(logical_indices, logical_indices)]

    d = len(logical_indices)

    survival = float(np.real(np.trace(K.conj().T @ K) / d))

    trace_val = np.trace(U_target.conj().T @ K)
    logical_fidelity = float(np.real(np.abs(trace_val) ** 2 / d**2))

    logical_infidelity = max(0.0, 1.0 - logical_fidelity)
    leakage_infidelity = max(0.0, 1.0 - survival)
    coherent_infidelity = max(0.0, survival - logical_fidelity)

    return logical_infidelity, leakage_infidelity, coherent_infidelity