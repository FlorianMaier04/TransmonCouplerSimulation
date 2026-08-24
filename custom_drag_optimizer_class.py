from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from qutip import sesolve
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.linalg import expm
from scipy.optimize import differential_evolution, least_squares
from tqdm import tqdm

from pulse_optimization import infid_population
from utils import Simulation
from FAPT import Heff_Floquet_total_matrix_summed, W_Floquet_elements_fast


@dataclass
class CustomDRAGConfig:
    rH: int = 2
    rW: int = 1
    correct: bool = True
    correct_dwd: bool = True
    correct_dA: bool = False
    correct_g: bool = False
    use_second_harmonic: bool = True
    drag_variant: str = "custom"

    tg: float = 100.0
    sigma_r: float = 0.3
    pulse_type: str = "gauss"
    Delta: float = -1.1737
    rrr: float = 1.0 / 0.0998
    drag_order: int = 5

    N_t: int = 201
    wd_half_range: float = 5.0
    amp_bound: float = 20.0
    max_nfev: int = 150
    dwd_backsteps: int = 1
    dA_backsteps: int = 5
    dDelta_lambda_backsteps: int = 5
    c02_weight: float = 1.0
    regularization_weight: float = 0.0

    robust_pointwise: bool = False
    robust_multistart: int = 8
    robust_local_max_nfev: int = 2000
    robust_residual_threshold: float = 1e-7
    robust_jump_threshold: float = 0.15
    robust_acceptance_factor: float = 1.05
    robust_global_fallback: bool = True
    robust_global_popsize: int = 8
    robust_global_maxiter: int = 30
    robust_seed: int = 1
    robust_debug: bool = False
    robust_debug_every: int = 1

    base_amp_guess: float = 0.1
    normalize_base_pulse: bool = True
    target_area: float = np.pi / 2.0
    epsilon_pi_area: float = np.pi
    steps_per_ns: float = 20.0
    run_drag_target_simulation: bool = True


class CustomDRAGOptimizer:
    def __init__(self, simulation, config=None):
        self.s = simulation
        self.c = config or CustomDRAGConfig()

        if self.c.drag_order not in (4, 5):
            raise ValueError("drag_order must be 4 or 5")
        if self.c.sigma_r <= 0:
            raise ValueError("sigma_r must be positive")
        self.c.pulse_type = self.c.pulse_type.lower()
        self.c.drag_variant = self.c.drag_variant.lower()
        if self.c.pulse_type not in ("gauss", "tanh"):
            raise ValueError("pulse_type must be 'gauss' or 'tanh'")
        if self.c.drag_variant not in ("standard", "custom"):
            raise ValueError("drag_variant must be 'standard' or 'custom'")

        keys = list(self.s.resonances)
        self.state_0, self.state_1 = keys[:2]
        n0, n1 = self.s.resonances[self.state_0], self.s.resonances[self.state_1]
        self.wd0 = abs(float(np.real(
            (self.s.E_array[self.state_1] - self.s.E_array[self.state_0]) / (n0 - n1)
        )))
        self.tlist = np.linspace(0.0, self.c.tg, self.c.N_t)
        self.n_controls = 5 if self.c.use_second_harmonic else 3

        print(f"Initial resonance frequency: wd0 = {self.wd0:.8f}")
        print(f"DRAG variant:                {self.c.drag_variant}")
        print(f"Base pulse:                  {self.c.pulse_type}")
        print(f"Second harmonic / H02 cancel: {self.c.use_second_harmonic}")

    # ------------------------------------------------------------------
    # Controls and Floquet Hamiltonian
    # ------------------------------------------------------------------

    def unpack_controls(self, values):
        if self.c.use_second_harmonic:
            return tuple(values)
        wd, Ar1, Ai1 = values
        return wd, Ar1, Ai1, 0.0, 0.0

    def make_controls(self, wd, Ar1, Ai1, Ar2=0.0, Ai2=0.0):
        values = [wd, Ar1, Ai1, Ar2, Ai2] if self.c.use_second_harmonic else [wd, Ar1, Ai1]
        return np.asarray(values, dtype=float)

    def floquet_matrix(self, controls, t=0.0, derivatives=None, apply_corrections=True):
        derivatives = np.zeros(self.n_controls) if derivatives is None else derivatives
        wd, Ar1, Ai1, Ar2, Ai2 = self.unpack_controls(controls)
        dwd, dAr1, dAi1, dAr2, dAi2 = self.unpack_controls(derivatives)

        A1, dA1 = Ar1 + 1j * Ai1, dAr1 + 1j * dAi1
        if self.c.use_second_harmonic:
            A = np.array([A1, Ar2 + 1j * Ai2])
            dA = np.array([dA1, dAr2 + 1j * dAi2])
            V = np.array([self.s.V1_dressed_array, self.s.V1_dressed_array])
        else:
            A, dA, V = A1, dA1, self.s.V1_dressed_array

        H = Heff_Floquet_total_matrix_summed(
            self.c.rH, wd, A, self.s.resonances, self.s.E_array, V,
            V0=None, ref_state=getattr(self.s, "ref_state", None),
            dwd=dwd, dA=dA, t=t, analytics=False, rW=self.c.rW,
            include_geometric=apply_corrections and self.c.correct and self.c.correct_dA,
            include_micromotion=apply_corrections and self.c.correct and self.c.correct_dwd,
            include_g_correction=apply_corrections and self.c.correct and self.c.correct_g and self.c.correct_dwd,
            verbose=False,
        )
        H = np.asarray(H, dtype=complex)
        return H - H[0, 0] * np.eye(H.shape[0])

    # ------------------------------------------------------------------
    # Base pulse and dynamic DRAG target
    # ------------------------------------------------------------------

    def base_shape(self, t):
        c = self.c
        sigma = c.sigma_r * c.tg
        if c.pulse_type == "gauss":
            edge = np.exp(-c.tg**2 / (8.0 * sigma**2))
            return np.exp(-(t - c.tg / 2.0)**2 / (2.0 * sigma**2)) - edge
        return np.tanh(t / sigma) * np.tanh((c.tg - t) / sigma)

    def base_shape_derivative(self, t):
        c = self.c
        sigma = c.sigma_r * c.tg
        if c.pulse_type == "gauss":
            gauss = np.exp(-(t - c.tg / 2.0)**2 / (2.0 * sigma**2))
            return -(t - c.tg / 2.0) * gauss / sigma**2
        x, y = t / sigma, (c.tg - t) / sigma
        return (np.tanh(y) / np.cosh(x)**2 - np.tanh(x) / np.cosh(y)**2) / sigma

    def base_controls(self, t, amplitude):
        return self.make_controls(self.wd0, amplitude * self.base_shape(t), 0.0)

    def base_area(self, amplitude):
        H01 = [
            self.floquet_matrix(self.base_controls(t, amplitude), t=t, apply_corrections=False)[0, 1].real
            for t in self.tlist
        ]
        return np.trapezoid(H01, self.tlist)

    def prepare_target(self):
        if self.c.drag_variant == "standard":
            self.prepare_standard_epsilon_pi()
        else:
            self.prepare_custom_epsilon_pi()

    def prepare_base_target(self):
        self.prepare_target()

    def prepare_custom_epsilon_pi(self):
        c = self.c
        if c.normalize_base_pulse:
            result = least_squares(
                lambda x: [self.base_area(x[0]) - c.target_area],
                [c.base_amp_guess], bounds=(-c.amp_bound, c.amp_bound),
                method="trf", xtol=1e-11, ftol=1e-11, gtol=1e-11,
                max_nfev=c.max_nfev,
            )
            self.f0 = float(result.x[0])
        else:
            self.f0 = float(c.base_amp_guess)

        self._custom_base_controls = np.array([self.base_controls(t, self.f0) for t in self.tlist])
        H0 = np.array([
            self.floquet_matrix(values, t=t, apply_corrections=False)
            for values, t in zip(self._custom_base_controls, self.tlist)
        ])
        self.ep0 = 2.0 * H0[:, 0, 1].real
        self.ep0_dot = CubicSpline(self.tlist, self.ep0)(self.tlist, 1)
        self.Vab = np.nan

        print(f"Base amplitude factor:        f0 = {self.f0:.10f}")
        print(f"Base effective pulse area:    {self.base_area(self.f0):.10f}")

    def prepare_standard_epsilon_pi(self):
        c = self.c
        shape = np.asarray(self.base_shape(self.tlist), dtype=float)
        shape_area = np.trapezoid(shape, self.tlist)
        if abs(shape_area) < 1e-14:
            raise ValueError("The standard-DRAG envelope has vanishing area")

        self.f0 = float(c.epsilon_pi_area / shape_area)
        self.ep0 = self.f0 * shape
        self.ep0_dot = self.f0 * self.base_shape_derivative(self.tlist)
        self.Vab = np.nan

        print(f"Standard E_pi amplitude:      f0 = {self.f0:.10f}")
        print(f"Standard E_pi area:           {np.trapezoid(self.ep0, self.tlist):.10f}")

    def prepare_initial_controls(self):
        if not hasattr(self, "ep0"):
            raise RuntimeError("prepare_target() must be called before prepare_initial_controls()")
        if self.c.drag_variant == "custom":
            self.initial_controls = self._custom_base_controls.copy()
            self.analytical_controls = self.initial_controls.copy()
            self.base_solutions = self.initial_controls
            return

        c = self.c
        Vab_complex = self.s.V1_dressed_array[self.state_0, self.state_1]
        if abs(np.imag(Vab_complex)) > 1e-10 * max(abs(Vab_complex), 1.0):
            raise ValueError(f"Standard DRAG assumes real Vab, obtained {Vab_complex}")
        self.Vab = float(np.real(Vab_complex))
        if abs(self.Vab) < 1e-14:
            raise ValueError("Vab is zero; the standard-DRAG controls are undefined")

        Ar = self.ep0 / self.Vab
        Ai = -self.ep0_dot / (c.Delta * self.Vab)
        wd = self.wd0 + (1.0 - c.rrr**2 / 4.0) * self.ep0**2 / c.Delta
        self.analytical_controls = np.array([self.make_controls(w, ar, ai) for w, ar, ai in zip(wd, Ar, Ai)])

        Ar = np.clip(Ar, -c.amp_bound, c.amp_bound)
        Ai = np.clip(Ai, -c.amp_bound, c.amp_bound)
        wd = np.clip(wd, self.wd0 - c.wd_half_range, self.wd0 + c.wd_half_range)
        self.initial_controls = np.array([self.make_controls(w, ar, ai) for w, ar, ai in zip(wd, Ar, Ai)])
        self.base_solutions = self.initial_controls

    def safe_delta(self, value):
        sign = value if value != 0 else self.c.Delta
        return np.copysign(max(abs(float(value)), 1e-8), sign)

    def current_lambda(self, H, index):
        if abs(H[0, 1]) > 1e-10:
            value = H[1, 2] / H[0, 1]
            if np.isfinite(value.real) and np.isfinite(value.imag):
                return value
        return self.lambda_values[index - 1] if index else complex(self.c.rrr)

    def drag_target(self, index, Delta_eff, dDelta_eff, lambda_eff, dlambda_eff):
        ep, ep_dot = self.ep0[index], self.ep0_dot[index]
        Delta = self.safe_delta(Delta_eff)
        lam, lam2 = lambda_eff, lambda_eff**2

        eps_x = ep
        eps_y = -ep_dot / Delta + ep * dDelta_eff / Delta**2
        delta1 = (lam2 - 4.0) * ep**2 / (4.0 * Delta)

        if self.c.drag_order == 5:
            eps_x += (lam2 - 4.0) * ep**3 / (8.0 * Delta**2)
            eps_x -= (13.0 * lam2**2 - 76.0 * lam2 + 112.0) * ep**5 / (128.0 * Delta**4)
            eps_y += 33.0 * (lam2 - 2.0) * ep**2 * ep_dot / (24.0 * Delta**3)
            eps_y += (6.0 * lam * dlambda_eff - 11.0 * (lam2 - 2.0) * dDelta_eff) * ep**3 / (8.0 * Delta**4)
            delta1 -= (lam2**2 - 7.0 * lam2 + 12.0) * ep**4 / (16.0 * Delta**3)

        H01 = 0.5 * (eps_x - 1j * eps_y)
        return np.array([delta1.real, H01.real, H01.imag])

    def backward_control_derivatives(self, controls, index):
        result = np.zeros(self.n_controls)
        if not self.c.correct or index == 0:
            return result
        if self.c.correct_dwd:
            j = max(0, index - self.c.dwd_backsteps)
            result[0] = (controls[0] - self.solutions[j, 0]) / (self.tlist[index] - self.tlist[j])
        if self.c.correct_dA:
            j = max(0, index - self.c.dA_backsteps)
            result[1:] = (controls[1:] - self.solutions[j, 1:]) / (self.tlist[index] - self.tlist[j])
        return result

    def backward_delta_derivative(self, value, index):
        if index == 0:
            return 0.0
        j = max(0, index - self.c.dDelta_lambda_backsteps)
        return (value - self.Delta_values[j]) / (self.tlist[index] - self.tlist[j])

    def backward_lambda_derivative(self, value, index):
        if index == 0:
            return 0.0
        j = max(0, index - self.c.dDelta_lambda_backsteps)
        return (value - self.lambda_values[j]) / (self.tlist[index] - self.tlist[j])

    # ------------------------------------------------------------------
    # Pointwise pulse-shape optimization
    # ------------------------------------------------------------------

    def optimize_pointwise(self):
        c = self.c
        if c.robust_pointwise:
            return self.optimize_pointwise_robust()
        self.robust_solver_used = False
        if not hasattr(self, "ep0"):
            raise RuntimeError("prepare_target() must be called before optimize_pointwise()")
        if not hasattr(self, "initial_controls"):
            self.prepare_initial_controls()
        eps_y_scale = -self.ep0_dot / self.safe_delta(c.Delta)
        H11_scale = max(np.max(np.abs((c.rrr**2 - 4.0) * self.ep0**2 / (4.0 * c.Delta))), 1e-8)
        H01_scale = max(np.max(np.hypot(0.5 * self.ep0, -0.5 * eps_y_scale)), 1e-8)
        self.core_scales = np.array([H11_scale, H01_scale, H01_scale])
        self.H01_scale = H01_scale
        self.H02_scale = H01_scale

        low = [self.wd0 - c.wd_half_range] + [-c.amp_bound] * (self.n_controls - 1)
        high = [self.wd0 + c.wd_half_range] + [c.amp_bound] * (self.n_controls - 1)
        low, high = np.asarray(low), np.asarray(high)
        parameter_scales = high - low

        N = c.N_t
        self.solutions = np.zeros((N, self.n_controls))
        self.control_derivatives = np.zeros_like(self.solutions)
        self.Delta_values = np.zeros(N)
        self.dDelta_values = np.zeros(N)
        self.lambda_values = np.full(N, complex(c.rrr), dtype=complex)
        self.targets = np.zeros((N, 3))
        self.H_eff_values = np.zeros((N, 3, 3), dtype=complex)
        self.optimization_residuals = np.zeros(N)
        self.total_residuals = np.zeros(N)

        previous = self.initial_controls[0].copy()
        for index, t in enumerate(tqdm(self.tlist, desc=f"Optimizing {c.drag_variant} DRAG pulse")):
            def evaluate(controls):
                derivatives = self.backward_control_derivatives(controls, index)
                H = self.floquet_matrix(controls, t=t, derivatives=derivatives)
                Delta_eff = float((H[2, 2] - 2.0 * H[1, 1]).real)
                dDelta_eff = self.backward_delta_derivative(Delta_eff, index)
                lambda_eff = self.current_lambda(H, index)
                dLambda_eff = self.backward_lambda_derivative(lambda_eff, index)
                target = self.drag_target(index, Delta_eff, dDelta_eff, lambda_eff, dLambda_eff)
                model = np.array([H[1, 1].real, H[0, 1].real, H[0, 1].imag])
                core = (model - target) / self.core_scales
                c02 = c.c02_weight * np.array([H[0, 2].real, H[0, 2].imag]) / self.H02_scale
                return H, derivatives, Delta_eff, dDelta_eff, lambda_eff, target, core, c02

            def residual(controls):
                *_, core, c02 = evaluate(controls)
                parts = [core]
                if c.use_second_harmonic:
                    parts.append(c02)
                values = np.concatenate(parts)
                if c.regularization_weight > 0.0:
                    values = np.r_[values, c.regularization_weight * (controls - previous) / parameter_scales]
                return values

            fit = least_squares(
                residual, previous, bounds=(low, high), method="trf", x_scale="jac",
                ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=c.max_nfev,
            )
            self.solutions[index] = fit.x
            values = evaluate(fit.x)
            H, derivatives, Delta_eff, dDelta_eff, lambda_eff, target, core, c02 = values
            self.control_derivatives[index] = derivatives
            self.H_eff_values[index] = H
            self.Delta_values[index] = Delta_eff
            self.dDelta_values[index] = dDelta_eff
            self.lambda_values[index] = lambda_eff
            self.targets[index] = target
            self.optimization_residuals[index] = np.linalg.norm(residual(fit.x))
            self.total_residuals[index] = np.linalg.norm(np.r_[core, c02])
            previous = fit.x

        self.wd_values, self.Ar_values, self.Ai_values = self.solutions[:, :3].T
        self.dwd_values, self.dAr_values, self.dAi_values = self.control_derivatives[:, :3].T
        if c.use_second_harmonic:
            self.Ar2_values, self.Ai2_values = self.solutions[:, 3:5].T
            self.dAr2_values, self.dAi2_values = self.control_derivatives[:, 3:5].T
        else:
            self.Ar2_values = self.Ai2_values = np.zeros(c.N_t)
            self.dAr2_values = self.dAi2_values = np.zeros(c.N_t)

        self.wd_func = CubicSpline(self.tlist, self.wd_values)
        self.Ar_func = CubicSpline(self.tlist, self.Ar_values)
        self.Ai_func = CubicSpline(self.tlist, self.Ai_values)
        self.Ar2_func = CubicSpline(self.tlist, self.Ar2_values)
        self.Ai2_func = CubicSpline(self.tlist, self.Ai2_values)
        self.dwd_func = CubicSpline(self.tlist, self.dwd_values)
        self.dAr_func = CubicSpline(self.tlist, self.dAr_values)
        self.dAi_func = CubicSpline(self.tlist, self.dAi_values)
        self.dAr2_func = CubicSpline(self.tlist, self.dAr2_values)
        self.dAi2_func = CubicSpline(self.tlist, self.dAi2_values)
        self.compute_diagnostics()

    def optimize_pointwise_robust(self):
        """Follow a continuous solution branch using multistart continuation and a global fallback."""
        c = self.c
        if not hasattr(self, "ep0"):
            raise RuntimeError("prepare_target() must be called before optimize_pointwise()")
        if not hasattr(self, "initial_controls"):
            self.prepare_initial_controls()

        eps_y_scale = -self.ep0_dot / self.safe_delta(c.Delta)
        H11_scale = max(np.max(np.abs((c.rrr**2 - 4.0) * self.ep0**2 / (4.0 * c.Delta))), 1e-8)
        H01_scale = max(np.max(np.hypot(0.5 * self.ep0, -0.5 * eps_y_scale)), 1e-8)
        self.core_scales = np.array([H11_scale, H01_scale, H01_scale])
        self.H01_scale = H01_scale
        self.H02_scale = H01_scale

        low = np.asarray([self.wd0 - c.wd_half_range] + [-c.amp_bound] * (self.n_controls - 1))
        high = np.asarray([self.wd0 + c.wd_half_range] + [c.amp_bound] * (self.n_controls - 1))
        parameter_scales = np.maximum(high - low, 1e-12)
        control_names = ("wd", "Ar1", "Ai1", "Ar2", "Ai2")[:self.n_controls]
        rng = np.random.default_rng(c.robust_seed)

        N = c.N_t
        self.solutions = np.zeros((N, self.n_controls))
        self.control_derivatives = np.zeros_like(self.solutions)
        self.Delta_values = np.zeros(N)
        self.dDelta_values = np.zeros(N)
        self.lambda_values = np.full(N, complex(c.rrr), dtype=complex)
        self.targets = np.zeros((N, 3))
        self.H_eff_values = np.zeros((N, 3, 3), dtype=complex)
        self.optimization_residuals = np.zeros(N)
        self.total_residuals = np.zeros(N)
        self.robust_physical_residuals = np.zeros((N, 5 if c.use_second_harmonic else 3))
        self.robust_branch_jumps = np.zeros(N)
        self.robust_nfev = np.zeros(N, dtype=int)
        self.robust_global_used = np.zeros(N, dtype=bool)
        self.robust_unresolved = np.zeros(N, dtype=bool)
        self.robust_solver_used = True

        previous = np.clip(self.initial_controls[0], low, high)
        iterator = tqdm(self.tlist, desc=f"Robust {c.drag_variant} DRAG continuation")
        for index, t in enumerate(iterator):
            if index < 2:
                predictor = previous.copy()
            else:
                predictor = np.clip(previous + (previous - self.solutions[index - 2]), low, high)

            def evaluate(controls):
                derivatives = self.backward_control_derivatives(controls, index)
                H = self.floquet_matrix(controls, t=t, derivatives=derivatives)
                Delta_eff = float((H[2, 2] - 2.0 * H[1, 1]).real)
                dDelta_eff = self.backward_delta_derivative(Delta_eff, index)
                lambda_eff = self.current_lambda(H, index)
                dLambda_eff = self.backward_lambda_derivative(lambda_eff, index)
                target = self.drag_target(index, Delta_eff, dDelta_eff, lambda_eff, dLambda_eff)
                model = np.array([H[1, 1].real, H[0, 1].real, H[0, 1].imag])
                core = (model - target) / self.core_scales
                c02 = c.c02_weight * np.array([H[0, 2].real, H[0, 2].imag]) / self.H02_scale
                return H, derivatives, Delta_eff, dDelta_eff, lambda_eff, target, core, c02

            def physical_residual(controls):
                *_, core, c02 = evaluate(controls)
                return np.r_[core, c02] if c.use_second_harmonic else core

            def local_residual(controls):
                values = physical_residual(controls)
                if c.regularization_weight > 0.0:
                    values = np.r_[values, c.regularization_weight * (controls - predictor) / parameter_scales]
                return values

            starts = [predictor, previous, np.clip(self.initial_controls[index], low, high)]
            if index > 0:
                starts.append(np.clip(0.5 * (predictor + self.initial_controls[index]), low, high))
            while len(starts) < max(c.robust_multistart, 1):
                level = 0.015 * 2.0**((len(starts) - 3) % 4)
                starts.append(np.clip(predictor + level * parameter_scales * rng.normal(size=self.n_controls), low, high))

            unique_starts = []
            for start in starts:
                if not any(np.allclose(start, existing, rtol=0.0, atol=1e-12) for existing in unique_starts):
                    unique_starts.append(start)

            candidates = []

            def add_local_candidate(start):
                fit = least_squares(
                    local_residual, start, bounds=(low, high), method="trf", jac="3-point",
                    x_scale=parameter_scales, ftol=1e-11, xtol=1e-11, gtol=1e-11,
                    max_nfev=c.robust_local_max_nfev,
                )
                physical = physical_residual(fit.x)
                jump = np.linalg.norm((fit.x - predictor) / parameter_scales) / np.sqrt(self.n_controls)
                candidates.append({"x": fit.x, "physical": physical, "norm": np.linalg.norm(physical), "jump": jump, "nfev": fit.nfev})

            for start in unique_starts:
                add_local_candidate(start)

            def select_candidate():
                minimum = min(candidate["norm"] for candidate in candidates)
                admissible = max(c.robust_residual_threshold, c.robust_acceptance_factor * minimum)
                connected = [candidate for candidate in candidates if candidate["norm"] <= admissible]
                return min(connected, key=lambda candidate: candidate["jump"])

            chosen = select_candidate()
            use_global = c.robust_global_fallback and (
                chosen["norm"] > c.robust_residual_threshold or chosen["jump"] > c.robust_jump_threshold
            )
            if use_global:
                def global_objective(controls):
                    values = physical_residual(controls)
                    return float(np.dot(values, values))

                global_fit = differential_evolution(
                    global_objective, bounds=list(zip(low, high)), x0=predictor, strategy="best1bin",
                    popsize=c.robust_global_popsize, maxiter=c.robust_global_maxiter, tol=1e-8,
                    atol=1e-12, polish=False, seed=c.robust_seed + index, workers=1,
                    updating="immediate", disp=False,
                )
                add_local_candidate(global_fit.x)
                chosen = select_candidate()
                self.robust_global_used[index] = True

            controls = chosen["x"]
            H, derivatives, Delta_eff, dDelta_eff, lambda_eff, target, core, c02 = evaluate(controls)
            physical = np.r_[core, c02] if c.use_second_harmonic else core

            self.solutions[index] = controls
            self.control_derivatives[index] = derivatives
            self.H_eff_values[index] = H
            self.Delta_values[index] = Delta_eff
            self.dDelta_values[index] = dDelta_eff
            self.lambda_values[index] = lambda_eff
            self.targets[index] = target
            self.optimization_residuals[index] = chosen["norm"]
            self.total_residuals[index] = np.linalg.norm(np.r_[core, c02])
            self.robust_physical_residuals[index] = physical
            self.robust_branch_jumps[index] = chosen["jump"]
            self.robust_nfev[index] = sum(candidate["nfev"] for candidate in candidates)
            self.robust_unresolved[index] = chosen["norm"] > c.robust_residual_threshold

            if c.robust_debug and (
                index % max(c.robust_debug_every, 1) == 0
                or self.robust_unresolved[index]
                or self.robust_global_used[index]
            ):
                bound_tol = 1e-5 * parameter_scales
                active = [
                    name for name, value, lower, upper, tol in zip(control_names, controls, low, high, bound_tol)
                    if value - lower <= tol or upper - value <= tol
                ]
                residual_names = ("r11", "Re r01", "Im r01", "Re r02", "Im r02")[:len(physical)]
                residual_text = ", ".join(f"{name}={value:+.2e}" for name, value in zip(residual_names, physical))
                tqdm.write(
                    f"[j={index:03d}, t/tg={t / c.tg:.4f}] |R|={chosen['norm']:.3e}, "
                    f"jump={chosen['jump']:.3e}, global={self.robust_global_used[index]}, "
                    f"bounds={active or '-'} | {residual_text}"
                )

            previous = controls

        self.wd_values, self.Ar_values, self.Ai_values = self.solutions[:, :3].T
        self.dwd_values, self.dAr_values, self.dAi_values = self.control_derivatives[:, :3].T
        if c.use_second_harmonic:
            self.Ar2_values, self.Ai2_values = self.solutions[:, 3:5].T
            self.dAr2_values, self.dAi2_values = self.control_derivatives[:, 3:5].T
        else:
            self.Ar2_values = self.Ai2_values = np.zeros(c.N_t)
            self.dAr2_values = self.dAi2_values = np.zeros(c.N_t)

        for name in ("wd", "Ar", "Ai", "Ar2", "Ai2", "dwd", "dAr", "dAi", "dAr2", "dAi2"):
            setattr(self, f"{name}_func", CubicSpline(self.tlist, getattr(self, f"{name}_values")))

        print("\nRobust continuation diagnostics")
        print(f"Global fallback calls:         {np.count_nonzero(self.robust_global_used)}/{N}")
        print(f"Unresolved time points:        {np.count_nonzero(self.robust_unresolved)}/{N}")
        print(f"Maximum physical residual:     {np.max(self.optimization_residuals):.3e}")
        print(f"Maximum normalized jump:       {np.max(self.robust_branch_jumps):.3e}")
        self.compute_diagnostics()

    def compute_diagnostics(self):
        c = self.c
        model = np.column_stack([
            self.H_eff_values[:, 1, 1].real,
            self.H_eff_values[:, 0, 1].real,
            self.H_eff_values[:, 0, 1].imag,
        ])
        self.element_errors = np.abs(model - self.targets)
        self.normalized_element_errors = self.element_errors / self.core_scales
        self.core_errors = np.linalg.norm(self.normalized_element_errors, axis=1)
        self.H02_values = self.H_eff_values[:, 0, 2]
        self.H02_errors = np.abs(self.H02_values) / self.H02_scale
        lambda_relation = self.H_eff_values[:, 1, 2] - c.rrr * self.H_eff_values[:, 0, 1]
        self.lambda_relation_errors = np.abs(lambda_relation) / self.H01_scale
        self.lambda_errors = np.abs(self.lambda_values - c.rrr) / max(abs(c.rrr), 1e-12)
        self.Delta_variation = np.abs(self.Delta_values - c.Delta) / max(abs(c.Delta), 1e-12)
        mask = np.abs(self.H_eff_values[:, 1, 2]) > 0.05 * np.max(np.abs(self.H_eff_values[:, 1, 2]))

        names = ("H11", "Re(H01)", "Im(H01)")
        print("\nPointwise target-matching diagnostics")
        print(f"Normalization scales: H11={self.core_scales[0]:.3e}, H01/H02={self.H01_scale:.3e}")
        for j, name in enumerate(names):
            print(f"Max normalized {name:7s} mismatch: {np.max(self.normalized_element_errors[:, j]):.3e}")
        print(f"Max combined target mismatch:     {np.max(self.core_errors):.3e}")
        print(f"Max normalized |H02|:             {np.max(self.H02_errors):.3e}")
        print(f"Max absolute |H02|:               {np.max(np.abs(self.H02_values)):.3e}")
        print(f"Mean optimizer objective:         {np.mean(self.optimization_residuals):.3e}")
        if np.any(mask):
            print(f"Max lambda-relation mismatch:     {np.max(self.lambda_relation_errors[mask]):.3e}")
            print(f"Max relative lambda variation:    {np.max(self.lambda_errors[mask]):.3e}")
        print(f"Max relative Delta variation:     {np.max(self.Delta_variation):.3e}")
        if c.use_second_harmonic:
            print(f"Max |A2r| / |A2i|:                {np.max(np.abs(self.Ar2_values)):.3e} / {np.max(np.abs(self.Ai2_values)):.3e}")

    # ------------------------------------------------------------------
    # Direct propagation of the generated DRAG target
    # ------------------------------------------------------------------

    def simulate_drag_target(self, plot=True, verbose=True):
        H_values = np.zeros((self.c.N_t, 3, 3), dtype=complex)
        for i in range(self.c.N_t):
            H11 = self.targets[i, 0]
            H01 = self.targets[i, 1] + 1j * self.targets[i, 2]
            lam = self.lambda_values[i]
            if abs(lam) < 1e-10:
                lam = self.lambda_values[i - 1] if i else complex(self.c.rrr)
            H12 = lam * H01
            H22 = self.Delta_values[i] + 2.0 * H11
            H_values[i] = [[0.0, H01, 0.0], [H01.conjugate(), H11, H12], [0.0, H12.conjugate(), H22]]

        H_spline = CubicSpline(self.tlist, H_values, axis=0)

        def rhs(t, flat_U):
            U = flat_U.reshape(3, 3)
            return (-1j * H_spline(t) @ U).reshape(-1)

        result = solve_ivp(
            rhs, (0.0, self.c.tg), np.eye(3, dtype=complex).reshape(-1),
            t_eval=self.tlist, method="DOP853", rtol=1e-10, atol=1e-12,
        )
        if not result.success:
            raise RuntimeError(result.message)

        self.U_drag_values = result.y.T.reshape(-1, 3, 3)
        self.U_drag = self.U_drag_values[-1]
        self.F_drag = 0.5 * (abs(self.U_drag[1, 0])**2 + abs(self.U_drag[0, 1])**2)
        self.I_drag = 1.0 - self.F_drag
        self.L_drag = 0.5 * (abs(self.U_drag[2, 0])**2 + abs(self.U_drag[2, 1])**2)

        if verbose:
            print("\nPure time-dependent DRAG target:")
            print(f"Population fidelity:           {self.F_drag:.10f}")
            print(f"Population infidelity:         {self.I_drag:.3e}")
            print(f"Mean final leakage:            {self.L_drag:.3e}")
        if plot:
            self.plot_drag_dynamics()
        return {"U_drag_values": self.U_drag_values, "F_drag": self.F_drag, "I_drag": self.I_drag, "L_drag": self.L_drag}

    # ------------------------------------------------------------------
    # Full-system simulation and plots
    # ------------------------------------------------------------------

    def build_drive_signal(self):
        wd = self.wd_func
        Ar1, Ai1 = self.Ar_func, self.Ai_func
        Ar2, Ai2 = self.Ar2_func, self.Ai2_func

        def signal(t, args=None):
            phi = float(wd(t)) * t
            value = float(Ar1(t)) * np.cos(phi) + float(Ai1(t)) * np.sin(phi)
            if self.c.use_second_harmonic:
                value += float(Ar2(t)) * np.cos(2.0 * phi) + float(Ai2(t)) * np.sin(2.0 * phi)
            return value

        return signal

    def compute_print_floq_fidelity(self):
        H_spline = CubicSpline(self.tlist, self.H_eff_values, axis=0)
        d_res = self.H_eff_values.shape[1]
        D = len(self.s.E_array)
        res_indices = list(self.s.resonances)
        kref = min(self.s.resonances)

        U_eff = np.eye(d_res, dtype=complex)
        U_eff_values = np.empty((len(self.tlist), d_res, d_res), dtype=complex)
        U_eff_values[0] = U_eff

        for k in range(len(self.tlist) - 1):
            dt = self.tlist[k + 1] - self.tlist[k]
            t_mid = 0.5 * (self.tlist[k + 1] + self.tlist[k])
            U_eff = expm(-1j * H_spline(t_mid) * dt) @ U_eff
            U_eff_values[k + 1] = U_eff

        def floquet_data(controls):
            wd, Ar1, Ai1, Ar2, Ai2 = self.unpack_controls(controls)
            A1 = Ar1 + 1j * Ai1
            if self.c.use_second_harmonic:
                A = np.array([A1, Ar2 + 1j * Ai2])
                V = np.array([self.s.V1_dressed_array, self.s.V1_dressed_array])
            else:
                A, V = A1, self.s.V1_dressed_array
            W = W_Floquet_elements_fast(
                self.c.rW, wd, self.s.resonances, self.s.E_array, V,
                A=A, V0=None, analytics=False, verbose=False,
            )
            return wd, W

        def frame_map(wd, W, t):
            M = np.zeros((D, d_res), dtype=complex)
            for r in range(self.c.rW + 1):
                for l in range(D):
                    for column, state in enumerate(self.s.resonances):
                        for p, (coefficient, _, _) in W[r, l, state].items():
                            phase = np.exp(-1j * (self.s.E_array[kref] + p * wd) * t)
                            M[l, column] += phase * coefficient
            return M

        wd_initial, W_initial = floquet_data(self.solutions[0])
        M_inv_0 = np.zeros((d_res, D), dtype=complex)
        for r in range(self.c.rW + 1):
            for l in range(D):
                for row, state in enumerate(self.s.resonances):
                    for _, (coefficient, _, _) in W_initial[r, l, state].items():
                        M_inv_0[row, l] += np.conjugate(coefficient)

        U_floquet_values = np.empty_like(U_eff_values)
        for k, (t, controls) in enumerate(tqdm(
            zip(self.tlist, self.solutions), total=len(self.tlist),
            desc="Reconstructing Floquet frame",
        )):
            wd, W = (wd_initial, W_initial) if k == 0 else floquet_data(controls)
            M = frame_map(wd, W, t)
            U_total = M @ U_eff_values[k] @ M_inv_0
            columns = U_total[:, res_indices]
            norms = np.linalg.norm(columns, axis=0)
            if np.any(norms < 1e-14):
                raise ValueError(f"Floquet column with vanishing norm at t={t}: {norms}")
            columns /= norms[np.newaxis, :]
            U_floquet_values[k] = columns[res_indices, :]

        I_eff = infid_population(U_eff_values[-1])
        I_floquet = infid_population(U_floquet_values[-1])
        print("\nFloquet predictions")
        print(f"Effective-space infidelity:    {I_eff:.3e}")
        print(f"Transformed Floquet fidelity:  {1.0 - I_floquet:.10f}")
        print(f"Transformed Floquet infidelity: {I_floquet:.3e}")

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, dpi=110)
        labels = [rf"$|r_{j}\rangle$" for j in range(d_res)]
        for initial, ax in enumerate(axes):
            populations = np.abs(U_floquet_values[:, :, initial])**2
            populations_eff = np.abs(U_eff_values[:, :, initial])**2
            for state, label in enumerate(labels):
                if np.max(populations[:, state]) > 1e-5:
                    ax.plot(self.tlist / self.c.tg, populations[:, state], label=label)
                if np.max(populations_eff[:, state]) > 1e-5:
                    ax.plot(
                        self.tlist / self.c.tg, populations_eff[:, state],
                        "k--", alpha=0.55,
                        label="Effective space only" if state == 0 else None,
                    )
            ax.set_title(f"Initial state {labels[initial]}")
            ax.set(xlabel=r"Normalized time $t/t_g$", xlim=(0.0, 1.0), ylim=(-0.02, 1.02))
            ax.grid(True, linestyle=":", alpha=0.6)
        axes[0].set_ylabel("Population")
        axes[1].legend(frameon=False)
        fig.suptitle("Floquet Dynamics with Frame Reconstruction")
        fig.tight_layout()

        self.U_floquet_eff = U_eff_values[-1]
        self.U_floquet_eff_values = U_eff_values
        self.I_pop_floquet_eff = I_eff
        self.U_floquet = U_floquet_values[-1]
        self.U_floquet_values = U_floquet_values
        self.I_pop_floquet = I_floquet
        return I_floquet, self.U_floquet, U_floquet_values

    def compute_full_floquet(self, full_rH=None, full_rW=None, include_geometric=False, include_micromotion=True, include_g_correction=False, verbose=True):
        full_rH = self.c.rH if full_rH is None else full_rH
        full_rW = self.c.rW if full_rW is None else full_rW
        s, d_res, D = self.s, len(self.s.resonances), len(self.s.E_array)
        res_indices, kref = list(s.resonances), min(s.resonances)
        splines = (self.wd_func, self.Ar_func, self.Ai_func, self.Ar2_func, self.Ai2_func)[:self.n_controls]

        def floquet_data(t, compute_heff=True):
            controls = np.array([float(f(t)) for f in splines])
            derivatives = np.array([float(f(t, 1)) for f in splines])
            wd, Ar1, Ai1, Ar2, Ai2 = self.unpack_controls(controls)
            dwd, dAr1, dAi1, dAr2, dAi2 = self.unpack_controls(derivatives)
            A1, dA1 = Ar1 + 1j * Ai1, dAr1 + 1j * dAi1
            if self.c.use_second_harmonic:
                A = np.array([A1, Ar2 + 1j * Ai2])
                dA = np.array([dA1, dAr2 + 1j * dAi2])
                V = np.array([s.V1_dressed_array, s.V1_dressed_array])
            else:
                A, dA, V = A1, dA1, s.V1_dressed_array

            W = W_Floquet_elements_fast(full_rW, wd, s.resonances, s.E_array, V, A=A, V0=None, analytics=False, verbose=False)
            if not compute_heff:
                return wd, W, None
            H = Heff_Floquet_total_matrix_summed(
                full_rH, wd, A, s.resonances, s.E_array, V, V0=None,
                ref_state=getattr(s, "ref_state", None), dwd=dwd, dA=dA, t=t,
                analytics=False, rW=full_rW, include_geometric=include_geometric,
                include_micromotion=include_micromotion, include_g_correction=include_g_correction,
                W=W, verbose=False,
            )
            return wd, W, np.asarray(H, dtype=complex)

        def frame_map(wd, W, t):
            M = np.zeros((D, d_res), dtype=complex)
            for r in range(full_rW + 1):
                for l in range(D):
                    for column, state in enumerate(s.resonances):
                        for p, (coefficient, _, _) in W[r, l, state].items():
                            M[l, column] += np.exp(-1j * (s.E_array[kref] + p * wd) * t) * coefficient
            return M

        def project(U_eff, M, M_inv_0):
            columns = (M @ U_eff @ M_inv_0)[:, res_indices]
            norms = np.linalg.norm(columns, axis=0)
            if np.any(norms < 1e-14):
                raise ValueError(f"Floquet columns with vanishing norm: {norms}")
            return (columns / norms[np.newaxis, :])[res_indices, :]

        wd0, W0, _ = floquet_data(self.tlist[0], compute_heff=False)
        M_inv_0 = np.zeros((d_res, D), dtype=complex)
        for r in range(full_rW + 1):
            for l in range(D):
                for row, state in enumerate(s.resonances):
                    for _, (coefficient, _, _) in W0[r, l, state].items():
                        M_inv_0[row, l] += np.conjugate(coefficient)

        shape = (len(self.tlist), d_res, d_res)
        U_cached_values = np.empty(shape, dtype=complex)
        U_explicit_values = np.empty(shape, dtype=complex)
        U_floquet_values = np.empty(shape, dtype=complex)
        U_cached = np.eye(d_res, dtype=complex)
        U_explicit = np.eye(d_res, dtype=complex)
        H_cached = CubicSpline(self.tlist, self.H_eff_values, axis=0)
        U_cached_values[0] = U_cached
        U_explicit_values[0] = U_explicit
        U_floquet_values[0] = project(U_explicit, frame_map(wd0, W0, 0.0), M_inv_0)

        iterator = range(len(self.tlist) - 1)
        if verbose:
            iterator = tqdm(iterator, desc="Full Floquet propagation")
        for k in iterator:
            t0, t1 = self.tlist[k:k + 2]
            t_mid, dt = 0.5 * (t0 + t1), t1 - t0
            U_cached = expm(-1j * H_cached(t_mid) * dt) @ U_cached
            _, _, H_explicit = floquet_data(t_mid)
            U_explicit = expm(-1j * H_explicit * dt) @ U_explicit
            wd, W, _ = floquet_data(t1, compute_heff=False)
            U_cached_values[k + 1] = U_cached
            U_explicit_values[k + 1] = U_explicit
            U_floquet_values[k + 1] = project(U_explicit, frame_map(wd, W, t1), M_inv_0)

        output = {
            "U_cached_values": U_cached_values,
            "U_explicit_values": U_explicit_values,
            "U_floquet_values": U_floquet_values,
            "I_cached": float(infid_population(U_cached_values[-1])),
            "I_explicit": float(infid_population(U_explicit_values[-1])),
            "I_floquet": float(infid_population(U_floquet_values[-1])),
        }
        self.floquet_result = output
        if verbose:
            print(f"Cached effective-space infidelity:   {output['I_cached']:.3e}")
            print(f"Explicit effective-space infidelity: {output['I_explicit']:.3e}")
            print(f"Transformed Floquet infidelity:      {output['I_floquet']:.3e}")
        return output

    def direct_propagator(self, parameters=(0.0, 1.0, 1.0)):
        parameters = np.asarray(parameters, dtype=float).ravel()
        if self.c.use_second_harmonic:
            if parameters.size != 3:
                raise ValueError("Expected (wd_offset, amp1_scale, amp2_scale)")
            wd_offset, amp1_scale, amp2_scale = parameters
        else:
            if parameters.size not in (2, 3):
                raise ValueError("Expected (wd_offset, amp1_scale) without the second harmonic")
            wd_offset, amp1_scale = parameters[:2]
            amp2_scale = 0.0
        comp_states = [self.s.E_states[i] for i in self.s.comp_indices]

        def drive(t, args=None):
            wd = float(self.wd_func(t)) + wd_offset
            phi = wd * t
            q1 = float(self.Ar_func(t)) * np.cos(phi) + float(self.Ai_func(t)) * np.sin(phi)
            q2 = 0.0
            if self.c.use_second_harmonic:
                q2 = float(self.Ar2_func(t)) * np.cos(2.0 * phi) + float(self.Ai2_func(t)) * np.sin(2.0 * phi)
            return amp1_scale * q1 + amp2_scale * q2

        U = np.zeros((len(comp_states), len(comp_states)), dtype=complex)
        for column, psi0 in enumerate(comp_states):
            evolution = sesolve([self.s.H0, [self.s.V1, drive]], psi0, [0.0, self.c.tg], options={"nsteps": 100000})
            U[:, column] = [state.overlap(evolution.states[-1]) for state in comp_states]
        return U

    def direct_infidelity(self, parameters=(0.0, 1.0, 1.0)):
        U = self.direct_propagator(parameters)
        return float(infid_population(U)), U

    def optimize_direct_scaling(self, bounds=((-0.2, 0.2), (0.8, 1.5), (-3.0, 3.0)), popsize=6, maxiter=30, seed=1, verbose=False):
        bounds = tuple(bounds)
        if self.c.use_second_harmonic:
            if len(bounds) != 3:
                raise ValueError("Three bounds are required with the second harmonic")
            active_bounds = bounds
            x0 = np.array([0.0, 1.0, 1.0])
        else:
            if len(bounds) < 2:
                raise ValueError("Bounds for wd_offset and amp1_scale are required")
            active_bounds = bounds[:2]
            x0 = np.array([0.0, 1.0])
        cache = {}

        def objective(parameters):
            key = tuple(np.round(parameters, 12))
            if key not in cache:
                cache[key] = float(infid_population(self.direct_propagator(parameters)))
            return cache[key]

        fit = differential_evolution(
            objective, bounds=active_bounds, x0=x0, strategy="best1bin",
            popsize=popsize, maxiter=maxiter, tol=1e-7, atol=1e-10, polish=True,
            seed=seed, workers=1, updating="immediate", disp=verbose,
        )
        U = self.direct_propagator(fit.x)
        full_parameters = np.asarray(fit.x, dtype=float) if self.c.use_second_harmonic else np.array([fit.x[0], fit.x[1], 0.0])
        output = {
            "direct_opt_parameters": full_parameters,
            "U_direct_opt": U,
            "I_direct_opt": float(infid_population(U)),
            "direct_opt_nfev": int(fit.nfev),
        }
        self.direct_optimization_result = output
        return output

    def simulate_direct(self):
        c = self.c
        drive = self.build_drive_signal()
        tlist = np.linspace(0.0, c.tg, max(1001, int(c.tg * c.steps_per_ns) + 1))
        comp_states = [self.s.E_states[i] for i in self.s.comp_indices]
        U = np.zeros((len(comp_states), len(comp_states)), dtype=complex)

        print("\nDirect full-system simulation")
        print(f"Harmonics in drive:            {2 if c.use_second_harmonic else 1}")

        for j, psi0 in enumerate(tqdm(comp_states, desc="Direct simulation")):
            result = sesolve([self.s.H0, [self.s.V1, drive]], psi0, tlist, options={"nsteps": 100000})
            U[:, j] = [state.overlap(result.states[-1]) for state in comp_states]

        infidelity = infid_population(U)
        print(f"Population fidelity:           {1.0 - infidelity:.10f}")
        print(f"Population infidelity:         {infidelity:.3e}")

        dynamics = sesolve(
            [self.s.H0, [self.s.V1, drive]], self.s.E_states[self.state_0], tlist,
            options={"nsteps": 100000},
        )
        populations = np.array([
            [abs(state.overlap(psi))**2 for psi in dynamics.states] for state in self.s.E_states
        ])
        self.direct_result = {
            "infidelity": infidelity, "U": U, "tlist": tlist,
            "populations": populations, "drive": drive,
        }
        self.I_pop_direct, self.U_direct = infidelity, U
        self.tlist_direct, self.direct_populations = tlist, populations
        self.plot_direct_dynamics()
        return infidelity, U

    def plot_drag_dynamics(self):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, dpi=110)
        labels = (r"$|a\rangle$", r"$|b\rangle$", r"$|c\rangle$")
        for initial, ax in enumerate(axes):
            populations = np.abs(self.U_drag_values[:, :, initial])**2
            for state, label in enumerate(labels):
                ax.plot(self.tlist / self.c.tg, populations[:, state], label=label)
            ax.set_title(labels[initial] + " initial")
            ax.set_xlabel(r"$t/t_g$")
            ax.grid(True, linestyle=":", alpha=0.6)
        axes[0].set_ylabel("Population")
        axes[1].legend(frameon=False)
        fig.tight_layout()

    def plot_pulse_shapes(self):
        x = self.tlist / self.c.tg
        fig, axes = plt.subplots(3, 1, figsize=(5.5, 5.0), sharex=True, dpi=110)

        axes[0].plot(x, self.Ar_values, label=r"$A_{r,1}$")
        axes[1].plot(x, self.Ai_values, color="#d62728", label=r"$A_{i,1}$")
        if self.c.use_second_harmonic:
            axes[0].plot(x, self.Ar2_values, "--", label=r"$A_{r,2}$")
            axes[1].plot(x, self.Ai2_values, "--", color="#d62728", label=r"$A_{i,2}$")
        axes[2].plot(x, self.wd_values, color="#2ca02c", label=r"$\omega_d$")
        axes[2].axhline(self.wd0, color="#555555", linestyle=":", label=r"$\omega_{d,0}$")

        axes[0].set_ylabel(r"$A_r$")
        axes[1].set_ylabel(r"$A_i$")
        axes[2].set_ylabel(r"$\omega_d$")
        axes[2].set_xlabel(r"Normalized time $t/t_g$")
        for ax in axes:
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(frameon=False)
        axes[0].set_title(f"Pointwise {self.c.drag_variant.capitalize()} DRAG Controls ({self.c.pulse_type})", fontweight="bold")
        fig.tight_layout()

    def plot_errors(self):
        x = self.tlist / self.c.tg
        fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.0), sharex=True, dpi=100)
        labels = (r"$H_{11}$", r"$\mathrm{Re}\,H_{01}$", r"$\mathrm{Im}\,H_{01}$")
        for j, label in enumerate(labels):
            axes[0].semilogy(x, np.maximum(self.normalized_element_errors[:, j], 1e-16), label=label)
        axes[0].semilogy(x, np.maximum(self.H02_errors, 1e-16), "--", label=r"$|H_{02}|/s_{02}$")
        axes[0].set_ylabel("Normalized target mismatch")
        axes[0].set_title("Pointwise Effective-Hamiltonian Diagnostics", fontweight="bold")

        axes[1].semilogy(x, np.maximum(self.lambda_relation_errors, 1e-16), label=r"$|H_{12}-\lambda_{\rm ref}H_{01}|/s_{01}$")
        axes[1].semilogy(x, np.maximum(self.lambda_errors, 1e-16), "--", label=r"$|\lambda(t)-\lambda_{\rm ref}|/|\lambda_{\rm ref}|$")
        axes[1].semilogy(x, np.maximum(self.Delta_variation, 1e-16), "-.", label=r"$|\Delta_{\rm eff}(t)-\Delta|/|\Delta|$")
        axes[1].set(xlabel=r"Normalized time $t/t_g$", ylabel="Structural variation", xlim=(0.0, 1.0))
        for ax in axes:
            ax.legend(frameon=False)
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
        fig.tight_layout()

    def plot_direct_dynamics(self):
        result = self.direct_result
        fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=110)
        for index, population in enumerate(result["populations"]):
            if np.max(population) > 1e-4:
                ax.plot(result["tlist"] / self.c.tg, population, label=rf"$|E_{{{index}}}\rangle$")
        ax.set(xlabel=r"Normalized time $t/t_g$", ylabel="Population", xlim=(0.0, 1.0), ylim=(-0.02, 1.02))
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(frameon=False, ncol=2)
        ax.set_title("Direct Dynamics of the Pointwise Pulse", fontweight="bold")
        fig.tight_layout()

    def run(self):
        plt.rcParams.update({
            "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
            "axes.titlesize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
            "legend.fontsize": 8.5, "lines.linewidth": 1.6,
        })
        print(f"\n[1/4] Preparing the {self.c.drag_variant} DRAG target")
        self.prepare_target()
        print("\n[2/4] Finding the pointwise DRAG controls")
        self.optimize_pointwise()
        self.plot_pulse_shapes()
        self.plot_errors()
        if self.c.run_drag_target_simulation:
            print("\n[3/4] Propagating the generated three-level DRAG target")
            self.simulate_drag_target()
        print("\n[4/4] Simulating the original pointwise pulse in the full system")
        self.compute_print_floq_fidelity()
        self.simulate_direct()
        plt.show()
        return self


if __name__ == "__main__":
    config = CustomDRAGConfig(
        tg=100.0,
        sigma_r=0.3,
        pulse_type="gauss",  # "gauss" or "tanh"
        drag_order=5,
        use_second_harmonic=True,
        run_drag_target_simulation=True,
    )
    result = CustomDRAGOptimizer(Simulation(), config).run()
