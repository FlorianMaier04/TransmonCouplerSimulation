"""Gate-time sweep for the numerical custom-DRAG construction.

The script constructs the pointwise pulse, propagates the ideal DRAG target,
evaluates the cached and explicitly reconstructed Floquet models, performs the
full-system simulation, and optionally optimizes three global pulse parameters.
All relevant sweep data are stored in one compressed NPZ file.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qutip import sesolve
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from scipy.optimize import differential_evolution
from tqdm.auto import tqdm

from custom_drag_optimizer_class import CustomDRAGConfig, CustomDRAGOptimizer
from FAPT import Heff_Floquet_total_matrix_summed, W_Floquet_elements_fast
from pulse_optimization import infid_population
from utils import Simulation


@dataclass
class SweepConfig:
    tg_values: np.ndarray = field(default_factory=lambda: np.array(
        [25.0, 27.5, 30.0, 40.0, 50.0, 65.0, 80.0, 100.0]
    ))
    output_dir: Path = Path("operational_res/chapter5_data")
    data_name: str = "custom_drag_tg_sweep.npz"
    figure_name: str = "custom_drag_infidelity_vs_tg"

    pulse_type: str = "tanh"
    sigma_r: float = 0.1
    drag_order: int = 5
    rH: int = 2
    rW: int = 1
    use_second_harmonic: bool = True
    N_t: int = 200

    full_rH: int = 2
    full_rW: int = 1
    full_geometric: bool = False
    full_micromotion: bool = True
    full_g_correction: bool = False

    run_direct_optimization: bool = True
    direct_bounds: tuple = ((-0.2, 0.2), (0.8, 1.5), (-3.0, 3.0))
    direct_popsize: int = 6
    direct_maxiter: int = 30
    direct_seed: int = 1


class Chapter5DRAGSweep:
    def __init__(self, config: SweepConfig):
        self.c = config
        self.c.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.c.output_dir / self.c.data_name
        self.records: list[dict] = []
        self.simulation = Simulation()

    def optimizer_config(self, tg: float) -> CustomDRAGConfig:
        return CustomDRAGConfig(
            tg=float(tg),
            sigma_r=self.c.sigma_r,
            pulse_type=self.c.pulse_type,
            drag_order=self.c.drag_order,
            rH=self.c.rH,
            rW=self.c.rW,
            use_second_harmonic=self.c.use_second_harmonic,
            run_drag_target_simulation=False,
            N_t=self.c.N_t,
        )

    @staticmethod
    def simulate_drag_target(result):
        H_values = np.zeros((result.c.N_t, 3, 3), dtype=complex)
        for i in range(result.c.N_t):
            H11 = result.targets[i, 0]
            H01 = result.targets[i, 1] + 1j * result.targets[i, 2]
            lam = result.lambda_values[i]
            if abs(lam) < 1e-10:
                lam = result.lambda_values[i - 1] if i else complex(result.c.rrr)
            H12 = lam * H01
            H22 = result.Delta_values[i] + 2.0 * H11
            H_values[i] = (
                (0.0, H01, 0.0),
                (H01.conjugate(), H11, H12),
                (0.0, H12.conjugate(), H22),
            )

        H = CubicSpline(result.tlist, H_values, axis=0)

        def rhs(t, flat_U):
            U = flat_U.reshape(3, 3)
            return (-1j * H(t) @ U).reshape(-1)

        propagation = solve_ivp(
            rhs,
            (0.0, result.c.tg),
            np.eye(3, dtype=complex).reshape(-1),
            t_eval=result.tlist,
            method="DOP853",
            rtol=1e-10,
            atol=1e-12,
        )
        if not propagation.success:
            raise RuntimeError(propagation.message)

        U_values = propagation.y.T.reshape(-1, 3, 3)
        U = U_values[-1]
        fidelity = 0.5 * (abs(U[1, 0])**2 + abs(U[0, 1])**2)
        leakage = 0.5 * (abs(U[2, 0])**2 + abs(U[2, 1])**2)
        return {
            "U_drag_values": U_values,
            "F_drag": float(fidelity),
            "I_drag": float(1.0 - fidelity),
            "L_drag": float(leakage),
        }

    def compute_full_floquet(self, result):
        c, s = result.c, result.s
        d_res, D = len(s.resonances), len(s.E_array)
        res_indices, kref = list(s.resonances), min(s.resonances)
        splines = (
            result.wd_func, result.Ar_func, result.Ai_func,
            result.Ar2_func, result.Ai2_func,
        )[:result.n_controls]

        def floquet_data(t, compute_heff=True):
            controls = np.array([float(f(t)) for f in splines])
            derivatives = np.array([float(f(t, 1)) for f in splines])
            wd, Ar1, Ai1, Ar2, Ai2 = result.unpack_controls(controls)
            dwd, dAr1, dAi1, dAr2, dAi2 = result.unpack_controls(derivatives)
            A1, dA1 = Ar1 + 1j * Ai1, dAr1 + 1j * dAi1

            if c.use_second_harmonic:
                A = np.array([A1, Ar2 + 1j * Ai2])
                dA = np.array([dA1, dAr2 + 1j * dAi2])
                V = np.array([s.V1_dressed_array, s.V1_dressed_array])
            else:
                A, dA, V = A1, dA1, s.V1_dressed_array

            W = W_Floquet_elements_fast(
                self.c.full_rW, wd, s.resonances, s.E_array, V,
                A=A, V0=None, analytics=False, verbose=False,
            )
            if not compute_heff:
                return wd, W, None

            H = Heff_Floquet_total_matrix_summed(
                self.c.full_rH, wd, A, s.resonances, s.E_array, V,
                V0=None, ref_state=getattr(s, "ref_state", None),
                dwd=dwd, dA=dA, t=t, analytics=False, rW=self.c.full_rW,
                include_geometric=self.c.full_geometric,
                include_micromotion=self.c.full_micromotion,
                include_g_correction=self.c.full_g_correction,
                W=W, verbose=False,
            )
            return wd, W, np.asarray(H, dtype=complex)

        def frame_map(wd, W, t):
            M = np.zeros((D, d_res), dtype=complex)
            for r in range(self.c.full_rW + 1):
                for l in range(D):
                    for column, state in enumerate(s.resonances):
                        for p, (coefficient, _, _) in W[r, l, state].items():
                            M[l, column] += np.exp(
                                -1j * (s.E_array[kref] + p * wd) * t
                            ) * coefficient
            return M

        def project(U_eff, M, M_inv_0):
            columns = (M @ U_eff @ M_inv_0)[:, res_indices]
            norms = np.linalg.norm(columns, axis=0)
            if np.any(norms < 1e-14):
                raise ValueError(f"Floquet columns with vanishing norm: {norms}")
            return (columns / norms[np.newaxis, :])[res_indices, :]

        wd0, W0, _ = floquet_data(result.tlist[0], compute_heff=False)
        M_inv_0 = np.zeros((d_res, D), dtype=complex)
        for r in range(self.c.full_rW + 1):
            for l in range(D):
                for row, state in enumerate(s.resonances):
                    for _, (coefficient, _, _) in W0[r, l, state].items():
                        M_inv_0[row, l] += np.conjugate(coefficient)

        shape = (len(result.tlist), d_res, d_res)
        U_cached_values = np.empty(shape, dtype=complex)
        U_explicit_values = np.empty(shape, dtype=complex)
        U_floquet_values = np.empty(shape, dtype=complex)
        U_cached = np.eye(d_res, dtype=complex)
        U_explicit = np.eye(d_res, dtype=complex)
        H_cached = CubicSpline(result.tlist, result.H_eff_values, axis=0)

        U_cached_values[0] = U_cached
        U_explicit_values[0] = U_explicit
        U_floquet_values[0] = project(
            U_explicit, frame_map(wd0, W0, 0.0), M_inv_0
        )

        for k in range(len(result.tlist) - 1):
            t0, t1 = result.tlist[k:k + 2]
            t_mid, dt = 0.5 * (t0 + t1), t1 - t0
            U_cached = expm(-1j * H_cached(t_mid) * dt) @ U_cached
            _, _, H_explicit = floquet_data(t_mid)
            U_explicit = expm(-1j * H_explicit * dt) @ U_explicit
            wd, W, _ = floquet_data(t1, compute_heff=False)

            U_cached_values[k + 1] = U_cached
            U_explicit_values[k + 1] = U_explicit
            U_floquet_values[k + 1] = project(
                U_explicit, frame_map(wd, W, t1), M_inv_0
            )

        return {
            "U_cached_values": U_cached_values,
            "U_explicit_values": U_explicit_values,
            "U_floquet_values": U_floquet_values,
            "I_cached": float(infid_population(U_cached_values[-1])),
            "I_explicit": float(infid_population(U_explicit_values[-1])),
            "I_floquet": float(infid_population(U_floquet_values[-1])),
        }

    @staticmethod
    def direct_propagator(result, parameters=(0.0, 1.0, 1.0)):
        wd_offset, amp1_scale, amp2_scale = parameters
        comp_states = [result.s.E_states[i] for i in result.s.comp_indices]

        def drive(t, args=None):
            wd = float(result.wd_func(t)) + wd_offset
            phi = wd * t
            q1 = (
                float(result.Ar_func(t)) * np.cos(phi)
                + float(result.Ai_func(t)) * np.sin(phi)
            )
            q2 = 0.0
            if result.c.use_second_harmonic:
                q2 = (
                    float(result.Ar2_func(t)) * np.cos(2.0 * phi)
                    + float(result.Ai2_func(t)) * np.sin(2.0 * phi)
                )
            return amp1_scale * q1 + amp2_scale * q2

        U = np.zeros((len(comp_states), len(comp_states)), dtype=complex)
        for column, psi0 in enumerate(comp_states):
            evolution = sesolve(
                [result.s.H0, [result.s.V1, drive]],
                psi0,
                [0.0, result.c.tg],
                options={"nsteps": 100000},
            )
            U[:, column] = [state.overlap(evolution.states[-1]) for state in comp_states]
        return U

    def optimize_direct(self, result):
        if not result.c.use_second_harmonic:
            raise ValueError("The three-parameter optimization requires the second harmonic")

        cache = {}

        def objective(parameters):
            key = tuple(np.round(parameters, 12))
            if key not in cache:
                cache[key] = float(infid_population(self.direct_propagator(result, parameters)))
            return cache[key]

        fit = differential_evolution(
            objective,
            bounds=self.c.direct_bounds,
            x0=np.array([0.0, 1.0, 1.0]),
            strategy="best1bin",
            popsize=self.c.direct_popsize,
            maxiter=self.c.direct_maxiter,
            tol=1e-7,
            atol=1e-10,
            polish=True,
            seed=self.c.direct_seed,
            workers=1,
            updating="immediate",
            disp=False,
        )
        U = self.direct_propagator(result, fit.x)
        return {
            "direct_opt_parameters": np.asarray(fit.x, dtype=float),
            "U_direct_opt": U,
            "I_direct_opt": float(infid_population(U)),
            "direct_opt_nfev": int(fit.nfev),
        }

    def run_gate_time(self, tg: float):
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            result = CustomDRAGOptimizer(self.simulation, self.optimizer_config(tg))
            result.prepare_base_target()
            result.optimize_pointwise()

        drag = self.simulate_drag_target(result)
        floquet = self.compute_full_floquet(result)
        U_direct = self.direct_propagator(result)
        direct = {
            "U_direct": U_direct,
            "I_direct": float(infid_population(U_direct)),
        }
        optimized = self.optimize_direct(result) if self.c.run_direct_optimization else {
            "direct_opt_parameters": np.full(3, np.nan),
            "U_direct_opt": np.full_like(U_direct, np.nan + 0j),
            "I_direct_opt": np.nan,
            "direct_opt_nfev": 0,
        }

        record = {
            "tg": float(tg),
            "f0": float(result.f0),
            "base_solutions": result.base_solutions,
            "ep0": result.ep0,
            "ep0_dot": result.ep0_dot,
            "solutions": result.solutions,
            "control_derivatives": result.control_derivatives,
            "Delta_values": result.Delta_values,
            "dDelta_values": result.dDelta_values,
            "lambda_values": result.lambda_values,
            "targets": result.targets,
            "H_eff_values": result.H_eff_values,
            "optimization_residuals": result.optimization_residuals,
            "total_residuals": result.total_residuals,
            "core_scales": result.core_scales,
            "H01_scale": float(result.H01_scale),
            "H02_scale": float(result.H02_scale),
            "max_core_error": float(np.max(result.core_errors)),
            "max_H02_error": float(np.max(result.H02_errors)),
            "max_lambda_error": float(np.max(result.lambda_errors)),
            "max_Delta_variation": float(np.max(result.Delta_variation)),
            **drag,
            **floquet,
            **direct,
            **optimized,
        }
        return record

    def save(self):
        successful = [record for record in self.records if record.get("success", False)]
        summary_keys = (
            "tg", "f0", "F_drag", "I_drag", "L_drag",
            "I_cached", "I_explicit", "I_floquet", "I_direct", "I_direct_opt",
            "max_core_error", "max_H02_error", "max_lambda_error",
            "max_Delta_variation", "direct_opt_nfev",
        )
        data = {
            "tg_requested": np.asarray(self.c.tg_values, dtype=float),
            "tg_completed": np.asarray([r["tg"] for r in self.records], dtype=float),
            "success": np.asarray([r.get("success", False) for r in self.records], dtype=bool),
            "error_messages": np.asarray([r.get("error", "") for r in self.records], dtype="U1024"),
            "t_normalized": np.linspace(0.0, 1.0, self.c.N_t),
            "pulse_type": np.asarray(self.c.pulse_type),
            "sigma_r": np.asarray(self.c.sigma_r),
            "drag_order": np.asarray(self.c.drag_order),
            "rH": np.asarray(self.c.rH),
            "rW": np.asarray(self.c.rW),
            "full_rH": np.asarray(self.c.full_rH),
            "full_rW": np.asarray(self.c.full_rW),
            "use_second_harmonic": np.asarray(self.c.use_second_harmonic),
        }

        for key in summary_keys:
            data[key] = np.asarray([r.get(key, np.nan) for r in self.records])

        if successful:
            data["pointwise_tg"] = np.asarray([r["tg"] for r in successful])
            array_keys = (
                "base_solutions", "ep0", "ep0_dot", "solutions",
                "control_derivatives", "Delta_values", "dDelta_values",
                "lambda_values", "targets", "H_eff_values",
                "optimization_residuals", "total_residuals", "core_scales",
                "U_drag_values", "U_cached_values", "U_explicit_values",
                "U_floquet_values", "U_direct", "direct_opt_parameters",
                "U_direct_opt",
            )
            for key in array_keys:
                data[key] = np.stack([r[key] for r in successful])
            data["H01_scale"] = np.asarray([r["H01_scale"] for r in successful])
            data["H02_scale"] = np.asarray([r["H02_scale"] for r in successful])

        np.savez_compressed(self.data_path, **data)

    def plot(self):
        successful = [r for r in self.records if r.get("success", False)]
        if not successful:
            return

        tg = np.asarray([r["tg"] for r in successful])
        curves = (
            ("I_drag", "DRAG target", "#9467bd", "-"),
            ("I_cached", "Cached effective", "#7f7f7f", "--"),
            ("I_explicit", "Explicit effective", "#ff7f0e", "-."),
            ("I_floquet", "Transformed Floquet", "#d62728", "-"),
            ("I_direct", "Direct full system", "#1f77b4", "-"),
        )

        plt.rcParams.update({
            "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
            "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5, "lines.linewidth": 1.6,
        })
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
        for key, label, color, linestyle in curves:
            values = np.asarray([r[key] for r in successful])
            ax.semilogy(tg, np.maximum(values, 1e-16), marker="o", color=color,
                        linestyle=linestyle, label=label)

        if self.c.run_direct_optimization:
            values = np.asarray([r["I_direct_opt"] for r in successful])
            ax.semilogy(tg, np.maximum(values, 1e-16), marker="s", color="#17becf",
                        linestyle="--", label="Direct, globally optimized")

        ax.set(xlabel=r"Gate duration $t_g$ [ns]", ylabel=r"Population infidelity $I_{\rm pop}$")
        ax.set_xlim(float(np.min(tg)), float(np.max(tg)))
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(self.c.output_dir / f"{self.c.figure_name}.pdf", bbox_inches="tight")
        fig.savefig(self.c.output_dir / f"{self.c.figure_name}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    def run(self):
        print(f"Chapter-5 custom-DRAG sweep: {len(self.c.tg_values)} gate times")
        print(f"Data file: {self.data_path}")

        for tg in tqdm(self.c.tg_values, desc="Gate-time sweep", unit="gate"):
            try:
                record = self.run_gate_time(float(tg))
                record["success"] = True
                self.records.append(record)
                opt_text = (
                    f", I_direct,opt={record['I_direct_opt']:.3e}"
                    if self.c.run_direct_optimization else ""
                )
                tqdm.write(
                    f"tg={tg:6.1f} ns | f0={record['f0']:.5g} | "
                    f"F_DRAG={record['F_drag']:.10f} | "
                    f"I_Floquet={record['I_floquet']:.3e} | "
                    f"I_direct={record['I_direct']:.3e}{opt_text}"
                )
            except Exception as exc:
                self.records.append({"tg": float(tg), "success": False, "error": repr(exc)})
                tqdm.write(f"tg={tg:6.1f} ns | FAILED: {exc}")
            self.save()

        self.plot()
        print(f"Saved sweep data to {self.data_path}")
        print(f"Saved plot to {self.c.output_dir / (self.c.figure_name + '.pdf')}")
        return self.records


if __name__ == "__main__":
    sweep = Chapter5DRAGSweep(SweepConfig())
    sweep.run()
