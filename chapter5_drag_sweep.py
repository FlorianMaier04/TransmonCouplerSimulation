"""Gate-time sweep for standard or custom numerical DRAG pulses."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from custom_drag_optimizer_class import CustomDRAGConfig, CustomDRAGOptimizer
from utils import Simulation


DETAIL_KEYS = (
    "base_solutions", "analytical_controls", "ep0", "ep0_dot", "solutions",
    "control_derivatives", "Delta_values", "dDelta_values", "lambda_values",
    "targets", "H_eff_values", "optimization_residuals", "total_residuals",
    "core_scales", "U_drag_values", "U_cached_values", "U_explicit_values",
    "U_floquet_values", "U_direct", "direct_opt_parameters", "U_direct_opt",
)


def load_sweep_data(path):
    """Load the complete sweep archive into an ordinary dictionary."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def inspect_sweep_point(path, sigma_r, tg):
    """Print and return all stored results for one (sigma, tg) point."""
    data = load_sweep_data(path)
    mask = np.isclose(data["pointwise_sigma"], sigma_r) & np.isclose(data["pointwise_tg"], tg)
    indices = np.flatnonzero(mask)
    if len(indices) != 1:
        raise KeyError(f"Expected one successful point for sigma={sigma_r:g}, tg={tg:g}; found {len(indices)}")

    i = int(indices[0])
    point = {key: data[key][i] for key in DETAIL_KEYS if key in data}
    for key in ("H01_scale", "H02_scale"):
        if key in data:
            point[key] = data[key][i]

    completed = np.flatnonzero(
        data["success"]
        & np.isclose(data["sigma_completed"], sigma_r)
        & np.isclose(data["tg_completed"], tg)
    )
    if len(completed) == 1:
        j = int(completed[0])
        for key in data["summary_keys"].astype(str):
            point[key] = data[key][j]

    point.update(sigma_r=float(sigma_r), tg=float(tg))
    print(f"\nStored point: tanh pulse, sigma={sigma_r:g}, tg={tg:g} ns")
    for key in ("F_drag", "I_drag", "I_floquet", "I_direct", "I_direct_opt"):
        if key in point:
            print(f"{key:18s}: {float(point[key]):.6e}")
    return point


@dataclass
class SweepConfig:
    tg_values: np.ndarray = field(default_factory=lambda: np.array([20,22.5,25,27.5,30,
                                                                    40,45,50,55,60,65,
                                                                    70,75,80,85,90,95,100]))
    sigma_values: np.ndarray = field(default_factory=lambda: np.array([0.01,0.05,0.1,0.5,0.8]))
    output_dir: Path = Path("operational_res/chapter5_data")
    data_name: str | None = None
    figure_name: str | None = None

    drag_variant: str = "custom"  # "standard" or "custom"
    pulse_type: str = "tanh"        # "gauss" or "tanh"
    plot_fidelity: str = "optimized_direct"     # "drag", "floquet", "direct" or "optimized_direct"
    epsilon_pi_area: float = np.pi
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

    run_direct_optimization: bool = True  # expensive; enable for optimized direct fidelity
    direct_bounds: tuple = ((-0.2, 0.2), (0.8, 1.5), (-3.0, 3.0))
    direct_popsize: int = 6
    direct_maxiter: int = 30
    direct_seed: int = 1


class Chapter5DRAGSweep:
    def __init__(self, config=None):
        self.c = config or SweepConfig()
        self.c.drag_variant = self.c.drag_variant.lower()
        if self.c.drag_variant not in ("standard", "custom"):
            raise ValueError("drag_variant must be 'standard' or 'custom'")
        if self.c.plot_fidelity not in ("drag", "floquet", "direct", "optimized_direct"):
            raise ValueError("plot_fidelity must be 'drag', 'floquet', 'direct' or 'optimized_direct'")
        if self.c.plot_fidelity == "optimized_direct" and not self.c.run_direct_optimization:
            raise ValueError("plot_fidelity='optimized_direct' requires run_direct_optimization=True")
        self.c.data_name = self.c.data_name or f"{self.c.drag_variant}_drag_{self.c.pulse_type}_sigma_tg_sweep.npz"
        self.c.figure_name = self.c.figure_name or f"{self.c.drag_variant}_drag_{self.c.pulse_type}_population_fidelity_vs_tg"
        self.c.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.c.output_dir / self.c.data_name
        self.s = Simulation()
        self.records = []

    def pointwise_config(self, tg, sigma_r):
        return CustomDRAGConfig(
            tg=float(tg), sigma_r=float(sigma_r), pulse_type=self.c.pulse_type,
            drag_variant=self.c.drag_variant, drag_order=self.c.drag_order,
            rH=self.c.rH, rW=self.c.rW, use_second_harmonic=self.c.use_second_harmonic,
            epsilon_pi_area=self.c.epsilon_pi_area, run_drag_target_simulation=False, N_t=self.c.N_t,
        )

    def run_gate_time(self, tg, sigma_r):
        with open(os.devnull, "w") as devnull, redirect_stdout(devnull), redirect_stderr(devnull):
            result = CustomDRAGOptimizer(self.s, self.pointwise_config(tg, sigma_r))
            result.prepare_target()
            result.optimize_pointwise()

        drag = result.simulate_drag_target(plot=False, verbose=False)
        floquet = result.compute_full_floquet(
            full_rH=self.c.full_rH, full_rW=self.c.full_rW,
            include_geometric=self.c.full_geometric, include_micromotion=self.c.full_micromotion,
            include_g_correction=self.c.full_g_correction, verbose=False,
        )
        I_direct, U_direct = result.direct_infidelity()
        direct = {"I_direct": I_direct, "U_direct": U_direct}

        if self.c.run_direct_optimization:
            optimized = result.optimize_direct_scaling(
                bounds=self.c.direct_bounds, popsize=self.c.direct_popsize,
                maxiter=self.c.direct_maxiter, seed=self.c.direct_seed, verbose=False,
            )
        else:
            optimized = {
                "direct_opt_parameters": np.full(3, np.nan),
                "U_direct_opt": np.full_like(U_direct, np.nan + 0j),
                "I_direct_opt": np.nan,
                "direct_opt_nfev": 0,
            }

        return {
            "tg": float(tg), "sigma_r": float(sigma_r), "f0": float(result.f0), "Vab": float(result.Vab),
            "base_solutions": result.base_solutions,
            "analytical_controls": result.analytical_controls,
            "ep0": result.ep0, "ep0_dot": result.ep0_dot,
            "solutions": result.solutions, "control_derivatives": result.control_derivatives,
            "Delta_values": result.Delta_values, "dDelta_values": result.dDelta_values,
            "lambda_values": result.lambda_values, "targets": result.targets,
            "H_eff_values": result.H_eff_values,
            "optimization_residuals": result.optimization_residuals,
            "total_residuals": result.total_residuals,
            "core_scales": result.core_scales,
            "H01_scale": float(result.H01_scale), "H02_scale": float(result.H02_scale),
            "max_core_error": float(np.max(result.core_errors)),
            "max_H02_error": float(np.max(result.H02_errors)),
            "max_lambda_error": float(np.max(result.lambda_errors)),
            "max_Delta_variation": float(np.max(result.Delta_variation)),
            **drag, **floquet, **direct, **optimized,
        }

    def save(self):
        successful = [record for record in self.records if record.get("success", False)]
        summary_keys = (
            "sigma_r", "tg", "f0", "Vab", "F_drag", "I_drag", "L_drag", "I_cached",
            "I_explicit", "I_floquet", "I_direct", "I_direct_opt",
            "max_core_error", "max_H02_error", "max_lambda_error",
            "max_Delta_variation", "direct_opt_nfev",
        )
        data = {
            "tg_requested": np.asarray(self.c.tg_values, dtype=float),
            "sigma_requested": np.asarray(self.c.sigma_values, dtype=float),
            "tg_completed": np.asarray([r["tg"] for r in self.records]),
            "sigma_completed": np.asarray([r["sigma_r"] for r in self.records]),
            "success": np.asarray([r.get("success", False) for r in self.records]),
            "error_messages": np.asarray([r.get("error", "") for r in self.records], dtype="U1024"),
            "t_normalized": np.linspace(0.0, 1.0, self.c.N_t),
            "drag_variant": np.asarray(self.c.drag_variant),
            "pulse_type": np.asarray(self.c.pulse_type),
            "epsilon_pi_area": np.asarray(self.c.epsilon_pi_area),
            "drag_order": np.asarray(self.c.drag_order),
            "rH": np.asarray(self.c.rH), "rW": np.asarray(self.c.rW),
            "full_rH": np.asarray(self.c.full_rH), "full_rW": np.asarray(self.c.full_rW),
            "use_second_harmonic": np.asarray(self.c.use_second_harmonic),
            "plot_fidelity": np.asarray(self.c.plot_fidelity),
            "summary_keys": np.asarray(summary_keys),
            "detail_keys": np.asarray(DETAIL_KEYS),
        }
        for key in summary_keys:
            data[key] = np.asarray([r.get(key, np.nan) for r in self.records])

        if successful:
            data["pointwise_tg"] = np.asarray([r["tg"] for r in successful])
            data["pointwise_sigma"] = np.asarray([r["sigma_r"] for r in successful])
            for key in DETAIL_KEYS:
                data[key] = np.stack([r[key] for r in successful])
            data["H01_scale"] = np.asarray([r["H01_scale"] for r in successful])
            data["H02_scale"] = np.asarray([r["H02_scale"] for r in successful])
        np.savez_compressed(self.data_path, **data)

    def plot(self):
        records = [record for record in self.records if record.get("success", False)]
        if not records:
            return
        metric = {
            "drag": ("F_drag", "DRAG-space population fidelity"),
            "floquet": ("I_floquet", "Transformed Floquet population fidelity"),
            "direct": ("I_direct", "Direct population fidelity"),
            "optimized_direct": ("I_direct_opt", "Optimized direct population fidelity"),
        }
        key, ylabel = metric[self.c.plot_fidelity]
        plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8.5, "lines.linewidth": 1.6})
        fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
        fidelities = []
        colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(self.c.sigma_values)))
        for color, sigma in zip(colors, self.c.sigma_values):
            subset = sorted((r for r in records if np.isclose(r["sigma_r"], sigma)), key=lambda r: r["tg"])
            if not subset:
                continue
            tg = np.asarray([r["tg"] for r in subset])
            values = np.asarray([r[key] for r in subset])
            fidelity = values if key == "F_drag" else 1.0 - values
            fidelities.extend(fidelity[np.isfinite(fidelity)])
            ax.plot(tg, fidelity, marker="o", color=color, label=rf"$\sigma={sigma:g}$")

        tg_requested = np.asarray(self.c.tg_values, dtype=float)
        if fidelities:
            ymin, ymax = min(fidelities), max(fidelities)
            margin = max(0.01, 0.08 * max(ymax - ymin, 1e-6))
            ax.set_ylim(max(0.0, ymin - margin), min(1.005, ymax + margin))
        ax.set(xlabel=r"Gate duration $t_g$ [ns]", ylabel=ylabel, xlim=(tg_requested.min(), tg_requested.max()))
        ax.set_title(f"{ylabel} — {self.c.pulse_type.capitalize()} {self.c.drag_variant.capitalize()} DRAG")
        ax.grid(True, linestyle=":", alpha=0.55)
        ax.legend(title="Tanh width", frameon=False, ncol=2)
        fig.tight_layout()
        fig.savefig(self.c.output_dir / f"{self.c.figure_name}.pdf", bbox_inches="tight")
        fig.savefig(self.c.output_dir / f"{self.c.figure_name}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    def run(self):
        points = [(float(sigma), float(tg)) for sigma in self.c.sigma_values for tg in self.c.tg_values]
        print(f"Chapter-5 {self.c.drag_variant}-DRAG sweep: {len(self.c.sigma_values)} tanh widths, {len(self.c.tg_values)} gate times")
        for sigma_r, tg in tqdm(points, desc="Sigma/gate-time sweep", unit="point"):
            try:
                record = self.run_gate_time(tg, sigma_r)
                record["success"] = True
                self.records.append(record)
                opt = f", I_direct,opt={record['I_direct_opt']:.3e}" if self.c.run_direct_optimization else ""
                tqdm.write(f"sigma={sigma_r:5.3f} | tg={tg:6.1f} ns | F_DRAG={record['F_drag']:.10f} | I_Floquet={record['I_floquet']:.3e} | I_direct={record['I_direct']:.3e}{opt}")
            except Exception as exc:
                self.records.append({"sigma_r": sigma_r, "tg": tg, "success": False, "error": repr(exc)})
                tqdm.write(f"sigma={sigma_r:5.3f} | tg={tg:6.1f} ns | FAILED: {exc}")
            self.save()
        self.plot()
        print(f"Saved data: {self.data_path}")
        print(f"Saved plot: {self.c.output_dir / (self.c.figure_name + '.pdf')}")
        return self.records


if __name__ == "__main__":
    Chapter5DRAGSweep(SweepConfig()).run()
