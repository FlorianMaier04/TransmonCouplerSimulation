"""
Visualize heff polynomial fits for all coefficient elements.
Creates plots showing the fitted surface for each heff element across
the parameter space (frequency, amplitude real, amplitude imaginary).
Also computes and displays correlation measures.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from scipy.stats import linregress
import seaborn as sns

# Parameters matching heff.ipynb
wd_range = [0.7 * 2*np.pi, 0.712 * 2*np.pi]
amp_real_range = (0, 0.6 * 2*np.pi)
amp_imag_range = (0, 0.1 * 2*np.pi)

def heff_fit(coeffs, wd, amp):
    """Polynomial fit function for heff elements."""
    ar, ai = np.real(amp), np.imag(amp)
    x = np.array([1, wd, ar, ai, wd**2, ar**2, ai**2,
                  wd*ar, wd*ai, ar*ai])
    return x @ coeffs

def load_coefficients(coeff_file):
    """Load precomputed coefficients from pkl file."""
    with open(coeff_file, 'rb') as f:
        coefficients_data = pickle.load(f)
    return coefficients_data

def compute_fit_surface(coeffs, n_points=50):
    """
    Compute fitted heff values across the parameter space.
    
    Returns:
        wd_grid, ar_grid, ai_grid: 1D arrays of parameter values
        fit_values: 3D array of fitted heff values
    """
    wd_vals = np.linspace(wd_range[0], wd_range[1], n_points)
    ar_vals = np.linspace(amp_real_range[0], amp_real_range[1], n_points)
    ai_vals = np.linspace(amp_imag_range[0], amp_imag_range[1], n_points)
    
    # Compute fit values for all parameter combinations
    fit_values = np.zeros((len(wd_vals), len(ar_vals), len(ai_vals)))
    
    for i, wd in enumerate(wd_vals):
        for j, ar in enumerate(ar_vals):
            for k, ai in enumerate(ai_vals):
                amp = ar + 1j*ai
                fit_values[i, j, k] = heff_fit(coeffs, wd, amp)
    
    return wd_vals, ar_vals, ai_vals, fit_values

def compute_correlations(wd_vals, ar_vals, ai_vals, fit_values):
    """
    Compute correlation measures between fitted values and parameters.
    
    Returns:
        Dictionary with correlation statistics
    """
    # Flatten all arrays
    fit_flat = fit_values.flatten()
    
    # Create grids and flatten
    wd_grid, ar_grid, ai_grid = np.meshgrid(wd_vals, ar_vals, ai_vals, indexing='ij')
    wd_flat = wd_grid.flatten()
    ar_flat = ar_grid.flatten()
    ai_flat = ai_grid.flatten()
    
    # Compute correlation coefficients
    corr_wd = np.corrcoef(wd_flat, fit_flat)[0, 1]
    corr_ar = np.corrcoef(ar_flat, fit_flat)[0, 1]
    """
    Compute correlation measures between fitted values and parameters.
    
    Returns:
        Dictionary with correlation statistics
    """
    # Flatten all arrays
    fit_flat = fit_values.flatten()
    
    # Create grids and flatten
    wd_grid, ar_grid, ai_grid = np.meshgrid(wd_vals, ar_vals, ai_vals, indexing='ij')
    wd_flat = wd_grid.flatten()
    ar_flat = ar_grid.flatten()
    ai_flat = ai_grid.flatten()
    
    # Compute correlation coefficients
    corr_wd = np.corrcoef(wd_flat, fit_flat)[0, 1]
    corr_ar = np.corrcoef(ar_flat, fit_flat)[0, 1]
    corr_ai = np.corrcoef(ai_flat, fit_flat)[0, 1]
    
    # Linear regression R-squared values
    slope_wd, intercept_wd, r_wd, _, _ = linregress(wd_flat, fit_flat)
    slope_ar, intercept_ar, r_ar, _, _ = linregress(ar_flat, fit_flat)
    slope_ai, intercept_ai, r_ai, _, _ = linregress(ai_flat, fit_flat)
    
    # Condition number of correlation matrix (measure of independence)
    corr_matrix = np.array([
        [1, corr_wd, corr_ar, corr_ai],
        [corr_wd, 1, np.corrcoef(wd_flat, ar_flat)[0,1], np.corrcoef(wd_flat, ai_flat)[0,1]],
        [corr_ar, np.corrcoef(wd_flat, ar_flat)[0,1], 1, np.corrcoef(ar_flat, ai_flat)[0,1]],
        [corr_ai, np.corrcoef(wd_flat, ai_flat)[0,1], np.corrcoef(ar_flat, ai_flat)[0,1], 1]
    ])
    
    cond_number = np.linalg.cond(corr_matrix)
    
    return {
        'corr_wd': corr_wd,
        'corr_ar': corr_ar,
        'corr_ai': corr_ai,
        'r_squared_wd': r_wd**2,
        'r_squared_ar': r_ar**2,
        'r_squared_ai': r_ai**2,
        'slope_wd': slope_wd,
        'slope_ar': slope_ar,
        'slope_ai': slope_ai,
        'condition_number': cond_number,
        'corr_matrix': corr_matrix
    }

def plot_2d_slices(wd_vals, ar_vals, ai_vals, fit_values, title, fig_num):
    """Plot 2D slices of the fit surface."""
    
    # Middle indices for slicing
    mid_wd = len(wd_vals) // 2
    mid_ar = len(ar_vals) // 2
    mid_ai = len(ai_vals) // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{title} - 2D Slices', fontsize=14, fontweight='bold')
    
    # Slice 1: ar vs wd (at fixed ai)
    im0 = axes[0].contourf(wd_vals, ar_vals, fit_values[:, :, mid_ai].T, levels=20, cmap='viridis')
    axes[0].set_xlabel('Drive Frequency wd (rad/ns)')
    axes[0].set_ylabel('Amplitude Real (rad/ns)')
    axes[0].set_title(f'at ai = {ai_vals[mid_ai]:.4f}')
    plt.colorbar(im0, ax=axes[0])
    
    # Slice 2: ai vs wd (at fixed ar)
    im1 = axes[1].contourf(wd_vals, ai_vals, fit_values[:, mid_ar, :].T, levels=20, cmap='viridis')
    axes[1].set_xlabel('Drive Frequency wd (rad/ns)')
    axes[1].set_ylabel('Amplitude Imag (rad/ns)')
    axes[1].set_title(f'at ar = {ar_vals[mid_ar]:.4f}')
    plt.colorbar(im1, ax=axes[1])
    
    # Slice 3: ai vs ar (at fixed wd)
    im2 = axes[2].contourf(ar_vals, ai_vals, fit_values[mid_wd, :, :].T, levels=20, cmap='viridis')
    axes[2].set_xlabel('Amplitude Real (rad/ns)')
    axes[2].set_ylabel('Amplitude Imag (rad/ns)')
    axes[2].set_title(f'at wd = {wd_vals[mid_wd]:.4f}')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(corr_matrix, title, fig_num):
    """Plot correlation matrix as heatmap."""
    
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['Fit', 'wd', 'ar', 'ai']
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, vmin=-1, vmax=1, square=True, ax=ax,
                xticklabels=labels, yticklabels=labels, cbar_kws={'label': 'Correlation'})
    ax.set_title(f'{title} - Parameter Correlations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    # Load coefficients
    coeff_dir = Path.cwd() / 'TransmonCouplerSimulation' / 'operational_res'
    coeff_file = coeff_dir / 'heff_coefficients.pkl'
    print("coeffdir: ",coeff_dir)
    
    if not coeff_file.exists():
        print(f"Error: Coefficient file not found at {coeff_file}")
        print("Please run heff.ipynb first to compute the coefficients.")
        return
    
    coefficients_data = load_coefficients(coeff_file)
    
    # Element names and their corresponding coefficients
    elements = {
        'H[0,0]': coefficients_data['coeffs00'],
        'H[0,1]': coefficients_data['coeffs01'],
        'H[0,2]': coefficients_data['coeffs02'],
        'H[1,1]': coefficients_data['coeffs11'],
        'H[1,2]': coefficients_data['coeffs12'],
        'H[2,2]': coefficients_data['coeffs22'],
    }
    
    print(f"Loaded {len(elements)} heff matrix elements")
    print("\n" + "="*70)
    print("HEff Polynomial Fit Analysis")
    print("="*70)
    
    all_correlations = {}
    
    # Process each element
    for elem_name, coeffs in elements.items():
        print(f"\n{elem_name}:")
        print("-" * 70)
        
        # Compute fit surface
        print(f"  Computing fit surface...")
        wd_vals, ar_vals, ai_vals, fit_values = compute_fit_surface(coeffs, n_points=30)
        
        # Compute correlations
        print(f"  Computing correlations...")
        correlations = compute_correlations(wd_vals, ar_vals, ai_vals, fit_values)
        all_correlations[elem_name] = correlations
        
        # Print statistics
        print(f"\n  Correlation with fitted values:")
        print(f"    wd (freq):     r = {correlations['corr_wd']:.4f}, R² = {correlations['r_squared_wd']:.4f}")
        print(f"    ar (amp_real): r = {correlations['corr_ar']:.4f}, R² = {correlations['r_squared_ar']:.4f}")
        print(f"    ai (amp_imag): r = {correlations['corr_ai']:.4f}, R² = {correlations['r_squared_ai']:.4f}")
        print(f"\n  Linear fit slopes:")
        print(f"    d(heff)/d(wd)   = {correlations['slope_wd']:.6f}")
        print(f"    d(heff)/d(ar)   = {correlations['slope_ar']:.6f}")
        print(f"    d(heff)/d(ai)   = {correlations['slope_ai']:.6f}")
        print(f"\n  Parameter independence (condition number): {correlations['condition_number']:.2f}")
        print(f"    (lower values → parameters more independent)")
        
        # Create plots
        fig1 = plot_2d_slices(wd_vals, ar_vals, ai_vals, fit_values, elem_name, 1)
        fig2 = plot_correlation_heatmap(correlations['corr_matrix'], elem_name, 2)
    
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    
    # Summary table
    print("\nCorrelation Summary (absolute values):")
    print(f"{'Element':<10} {'|r(wd)|':<10} {'|r(ar)|':<10} {'|r(ai)|':<10} {'Cond#':<10}")
    print("-" * 50)
    for elem_name, corr_data in all_correlations.items():
        print(f"{elem_name:<10} {abs(corr_data['corr_wd']):<10.4f} "
              f"{abs(corr_data['corr_ar']):<10.4f} {abs(corr_data['corr_ai']):<10.4f} "
              f"{corr_data['condition_number']:<10.2f}")
    
    plt.show()

if __name__ == '__main__':
    main()
