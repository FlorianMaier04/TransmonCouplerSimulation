from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MultipleLocator

FILE_PATH_LOW_SIGMA = Path(
    "operational_res/chapter5_data/custom_drag_tanh_sigma_tg_sweep_low_sigma.npz"
)
FILE_PATH_HIGH_SIGMA = Path(
    "operational_res/chapter5_data/custom_drag_tanh_sigma_tg_sweep_high_sigma.npz"
)
FILE_PATH_SINGLE_TONE = Path(
    "operational_res/chapter5_data/custom_drag_tanh_monochromatic.npz"
)
FILE_PATH_SINGLE_TONE_POINT = Path(
    "operational_res/chapter5_data/single_point_monochromatic.npz"
)

INFIDELITY_KEY = "I_direct_opt"
SIGMAS_TO_PLOT = [0.01, 0.1, 0.5,0.8]

SAVE = True
SHOW = True
FIGURE_PATH = Path("Figure/5/sesolve_optimized_tg_sweep.pdf")


def load_data(file_path, required_keys):
    if not file_path.exists():
        print(f"Warning: file {file_path} not found.")
        return None

    with np.load(file_path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}

    missing = set(required_keys).difference(data)
    if missing:
        raise KeyError(f"Missing arrays in {file_path}: {sorted(missing)}")

    return data


def setup_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "text.usetex": False,
    })


def valid_mask(data):
    tg = np.asarray(data["tg_completed"], dtype=float)
    success = np.asarray(data["success"], dtype=bool)

    if success.ndim == 0:
        success = np.full(tg.shape, bool(success))

    return success


def extract_two_tone(datasets, sigma):
    tg_values = []
    infid_values = []

    for data in datasets:
        if data is None:
            continue

        success = valid_mask(data)
        sigma_values = np.asarray(data["sigma_completed"], dtype=float)
        mask = success & np.isclose(sigma_values, sigma)

        tg_values.extend(np.asarray(data["tg_completed"][mask], dtype=float))
        infid_values.extend(np.asarray(data[INFIDELITY_KEY][mask], dtype=float))

    tg = np.asarray(tg_values)
    infid = np.asarray(infid_values)
    valid = np.isfinite(tg) & np.isfinite(infid) & (infid > 1e-15)

    tg, infid = tg[valid], infid[valid]
    order = np.argsort(tg)
    tg, infid = tg[order], infid[order]

    if tg.size:
        _, unique_indices = np.unique(tg, return_index=True)
        tg, infid = tg[unique_indices], infid[unique_indices]

    return tg, infid


def extract_single_tone(datasets):
    tg_values = []
    infid_values = []

    for data in datasets:
        if data is None:
            continue

        success = valid_mask(data)
        tg_values.extend(np.asarray(data["tg_completed"][success], dtype=float))
        infid_values.extend(np.asarray(data[INFIDELITY_KEY][success], dtype=float))

    tg = np.asarray(tg_values)
    infid = np.asarray(infid_values)
    valid = np.isfinite(tg) & np.isfinite(infid) & (infid > 1e-15)

    tg, infid = tg[valid], infid[valid]
    order = np.argsort(tg)
    tg, infid = tg[order], infid[order]

    if tg.size:
        _, unique_indices = np.unique(tg, return_index=True)
        tg, infid = tg[unique_indices], infid[unique_indices]

    return tg, infid


required_sweep_keys = {
    "tg_completed",
    "sigma_completed",
    "success",
    INFIDELITY_KEY,
}
required_single_tone_keys = {
    "tg_completed",
    "success",
    INFIDELITY_KEY,
}

data_low = load_data(FILE_PATH_LOW_SIGMA, required_sweep_keys)
data_high = load_data(FILE_PATH_HIGH_SIGMA, required_sweep_keys)
data_single = load_data(FILE_PATH_SINGLE_TONE, required_single_tone_keys)
data_single_point = load_data(FILE_PATH_SINGLE_TONE_POINT, required_single_tone_keys)

two_tone_datasets = [data_low, data_high]
single_tone_datasets = [data_single, data_single_point]

colors = {
    0.01: "#1f77b4",
    0.05: "#ff7f0e",
    0.1: "#2ca02c",
    0.5: "#d62728",
    0.8: "#7b2cbf",
}
markers = {
    0.01: "o",
    0.05: "s",
    0.1: "D",
    0.5: "^",
    0.8: "v",
}

setup_paper_style()

plot_style = {
    "axes.labelsize": 13.5,
    "xtick.labelsize": 12.0,
    "ytick.labelsize": 12.0,
    "legend.fontsize": 10.0,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
}

with plt.rc_context(plot_style):
    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=100)

    all_tg = []
    all_infid = []

    for sigma in SIGMAS_TO_PLOT:
        tg, infid = extract_two_tone(two_tone_datasets, sigma)

        if not tg.size:
            print(f"Warning: no two-tone data found for sigma={sigma:g}.")
            continue

        all_tg.extend(tg)
        all_infid.extend(infid)

        ax.semilogy(
            tg,
            infid,
            color=colors[sigma],
            marker=markers[sigma],
            mfc="white",
            mec=colors[sigma],
            mew=1.4,
            label=rf"Two-tone, $\sigma={sigma:g}$",
        )

    tg_single, infid_single = extract_single_tone(single_tone_datasets)

    if tg_single.size:
        all_tg.extend(tg_single)
        all_infid.extend(infid_single)

        ax.semilogy(
            tg_single,
            infid_single,
            color="#404040",
            linestyle="--",
            marker="p",
            mfc="white",
            mec="#404040",
            mew=1.4,
            label=r"Single-tone, $\sigma=0.1$",
        )

    ax.set_xlabel(r"Gate duration $t_g$ [ns]")
    ax.set_ylabel(r"Population infidelity $I_{\mathrm{pop}}^{\mathrm{direct}}$")

    if all_tg and all_infid:
        ax.set_xlim(np.min(all_tg), np.max(all_tg))
        ax.set_ylim(np.min(all_infid) * 0.4, min(np.max(all_infid) * 2.5, 1.0))

    ax.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=80))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))

    ax.tick_params(direction="in", which="both")
    ax.tick_params(axis="x", which="minor", length=4)
    ax.grid(True, which="both", linestyle=":", alpha=0.5, linewidth=0.8)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor="gray",
        framealpha=0.9,
        columnspacing=1.1,
        handletextpad=0.5,
    )

    fig.tight_layout()

    if SAVE:
        FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURE_PATH, bbox_inches="tight")

    if SHOW:
        plt.show()
    else:
        plt.close(fig)