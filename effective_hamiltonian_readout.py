import sys
import numpy as np
from qutip import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from Floquet_perturbation_theory import *
from helper_function import *
from IPython.display import display
from scipy.linalg import logm
import json
import os
from datetime import datetime
import threading

def configure_simulation():
    config = {
        # Timing parameters (nanoseconds)
        'time': 200,
        'tstep': 0.01,
        'accuracy': 1,
        
        # Base parameters for sweeps
        'base_amp': 0.1 * 2 * np.pi,
        'base_fre_offset': 0.0,        # offset from static resonant frequency
        
        # Matrix indices to compute
        'i': 0,
        'j': 1,
        
        # Parameter sweep selection
        'parameter_preset': 'parameters_slow',
        # Options: 'parameters_fast', 'parameters_slow', 'parameters_high_range', 'parameters_side', 'parameters_freq'
        
        # Element extraction type
        'extract_func': 'diag_diff',
        # Options: 'omega_ij' (off-diagonal), 'diag_diff' (diagonal difference)

        # Pulse modulation shape (pulse shape parameter sigma = tg*parameter)
        'pulse_mod': 'gaussian 0.5'
        # Options: 'gaussian sigma', None
    }
    return config

class Setup:

    @property 
    def pulse_modulation(self):
        return self._pulse_modulation
    @pulse_modulation.setter
    def pulse_modulation(self, value):
        if(value[0] == 'gaussian'):
            self.sigma = float(value[1])*self.tg
            # B is computed from modulation 0 at time 0 condition
            self.B = np.exp(self.tg**2/(2*self.sigma**2))
            self._pulse_modulation = value[0]
    
    RES_PATH = "/Users/florianmaier/Library/CloudStorage/OneDrive-TUM/ba/res/"

    def __init__(self, dim_q1=3, dim_q2=3, dim_c=3,
                 w1=3.83, w2=3.11, wc=4.29,
                 alpha1=-0.205, alpha2=-0.216, alphac=-0.161,
                 g1c=0.115, g2c=0.110, g12=0.015):
        """
        Initialize simulator with physical parameters.
        
        Args:
            dim_q1, dim_q2, dim_c: Hilbert space dimensions
            w1, w2, wc: Frequencies (in GHz, will be converted to rad/ns)
            alpha1, alpha2, alphac: Anharmonicities (in GHz, will be converted to rad/ns)
            g1c, g2c, g12: Coupling strengths (in GHz, will be converted to rad/ns)
            amp_sweep: Array of amplitudes to sweep (default: 0.001 to 0.05, 10 points)
            fre_sweep: Array of frequency detunings to sweep (default: -0.1 to 0.1, 10 points)
        """
        self.dim_q1, self.dim_q2, self.dim_c = dim_q1, dim_q2, dim_c
        
        # Convert frequencies and couplings to rad/ns frequencies are given in GHz
        self.w1_num = w1 * 2 * np.pi
        self.alpha1_num = alpha1 * 2 * np.pi
        self.w2_num = w2 * 2 * np.pi
        self.alpha2_num = alpha2 * 2 * np.pi
        self.wc_num = wc * 2 * np.pi
        self.alphac_num = alphac * 2 * np.pi
        
        self.g1c_num = g1c * 2 * np.pi
        self.g2c_num = g2c * 2 * np.pi
        self.g12_num = g12 * 2 * np.pi
        
        self._setup_operators()
        self._setup_hamiltonian()
        self.tg = 200
        self.tstep = 0.01
        self.accuracy = 1
        self._pulse_modulation = None
    def _setup_operators(self):
        """Initialize creation/annihilation operators."""
        self.a_q1 = tensor(destroy(self.dim_q1), qeye(self.dim_c), qeye(self.dim_q2))
        self.a_q2 = tensor(qeye(self.dim_q1), qeye(self.dim_c), destroy(self.dim_q2))
        self.a_qc = tensor(qeye(self.dim_q1), destroy(self.dim_c), qeye(self.dim_q2))
    def _setup_hamiltonian(self):
        """Setup Hamiltonian and transform to dressed basis."""
        Hq1 = self.w1_num * self.a_q1.dag() @ self.a_q1 + \
              (self.alpha1_num / 2) * self.a_q1.dag() @ self.a_q1.dag() @ self.a_q1 @ self.a_q1
        Hq2 = self.w2_num * self.a_q2.dag() @ self.a_q2 + \
              (self.alpha2_num / 2) * self.a_q2.dag() @ self.a_q2.dag() @ self.a_q2 @ self.a_q2
        Hc = self.wc_num * self.a_qc.dag() @ self.a_qc + \
             (self.alphac_num / 2) * self.a_qc.dag() @ self.a_qc.dag() @ self.a_qc @ self.a_qc
        V0 = (self.g1c_num * ((self.a_q1 - self.a_q1.dag()) @ (self.a_qc - self.a_qc.dag())) +
              self.g2c_num * ((self.a_q2 - self.a_q2.dag()) @ (self.a_qc - self.a_qc.dag())) +
              self.g12_num * ((self.a_q1 - self.a_q1.dag()) @ (self.a_q2 - self.a_q2.dag())))
        H0 = Hq1 + Hq2 + Hc + V0
        V1 = self.a_qc.dag() @ self.a_qc
        # Diagonalize H0
        evals, evecs = H0.eigenstates()
        self.sorted_evals, self.sorted_evecs = SortedFRFSpectrum(evals, evecs, self.dim_q1, self.dim_c, self.dim_q2)
        self.t_interaction_picture = Qobj(
            np.column_stack([
                self.sorted_evecs[i, j, k].full()
                for i in range(self.dim_q1)
                for j in range(self.dim_c)
                for k in range(self.dim_q2)
            ]),
            dims=[[self.dim_q1, self.dim_c, self.dim_q2], [self.dim_q1, self.dim_c, self.dim_q2]],
        )
        self.H0_dressed = self.t_interaction_picture.dag() @ H0 @ self.t_interaction_picture
        self.V1_dressed = self.t_interaction_picture.dag() @ V1 @ self.t_interaction_picture
        # Resonant frequency
        self.res_freq_static = (self.sorted_evals[1, 0, 0] - self.sorted_evals[0, 0, 1]) / (2 * np.pi)
    def extract_Omega_ij(self, Heff, i, j):
        """Extract Omega_ij = Heff[i,j] from effective Hamiltonian."""
        return Heff[i, j]
    def extract_diagonal_diff(self, Heff, i, j):
        """Extract diagonal difference: Heff[i,i] - Heff[j,j]."""
        return Heff[i, i] - Heff[j, j]

    def compute_U(self, fre, amp, show_progress=True):
        """
        Compute the subsystem unitary matrix.
        
        Args:
            tmax: Max evolution time
            fre: Drive frequency
            amp: Drive amplitude
            tstep: Time step
            show_progress: Show progress bar for states
            
        Returns:
            U_subsystem: 3x3 unitary matrix for subsystem
        """
        tlist = np.arange(0, self.tg, self.tstep)
        
        # State indices
        state_a = self.dim_c * self.dim_q2  # |100>
        state_b = 1  # |001>
        state_c = self.dim_q2  # |010>
        subsystem_indices = [state_a, state_b, state_c]
        
        U = np.zeros((3, 3), dtype=complex)
        
        iterator = tqdm(enumerate(subsystem_indices), total=3, leave=False, disable=not show_progress)
        for col_idx, initial_state_idx in iterator:
            initial_state = np.zeros(27)
            initial_state[initial_state_idx] = 1
            initial_state = Qobj(initial_state, dims=[[self.dim_q1, self.dim_c, self.dim_q2], [1]])
            # the pulse shape can be modulated by a time dependent amplitude
            f = None
            match self.pulse_modulation:
                case None: f = lambda t, args: amp * np.cos(fre * t)
                case 'gaussian': 
                    f = lambda t, args: amp * (np.exp(-(t-self.tg)**2/(2*self.sigma**2))-self.B) * np.cos(fre * t)
            # Evolve
            result = mesolve(
                [self.H0_dressed, [self.V1_dressed, f]],
                initial_state,
                tlist,
                []
            )
            final_state = result.states[-1]
            
            for row_idx, j_state in enumerate(subsystem_indices):
                U[row_idx, col_idx] = final_state.full()[j_state, 0]
        
        return U
        
    def sweep_parameters(self, list, base_val, i, j, mode, extract_func='omega_ij'):
        """
        Sweep over amplitude or frequency and compute matrix element for each.
        
        Args:
            list: Array of parameter values to sweep
            base_val: Base value for the constant parameter
            i, j: Indices for extraction
            mode: 'amplitude' or 'frequency'
            extract_func: 'omega_ij' for off-diagonal or 'diag_diff' for diagonal difference
        Returns:
            values: Extracted values for each parameter
        """        
        if mode == 'amplitude':
            fre_values = [base_val] * len(list)
            amp_values = list
        elif mode == 'frequency':
            amp_values = [base_val] * len(list)
            fre_values = list
        else:
            raise ValueError("mode must be 'amplitude' or 'frequency'")
        
        values = []
        param_label = 'Amplitude' if mode == 'amplitude' else 'Frequency'
        
        for amp, fre in tqdm(zip(amp_values, fre_values), total=len(list), 
                             desc=f"Computing element vs {param_label}", ncols=70, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            U = self.compute_U(fre, amp, show_progress=False)
            Heff = 1j / self.time * logm(U)
            
            if extract_func == 'omega_ij':
                value = self.extract_Omega_ij(Heff, i, j)
            elif extract_func == 'diag_diff':
                value = self.extract_diagonal_diff(Heff, i, j)
            else:
                raise ValueError("extract_func must be 'omega_ij' or 'diag_diff'")
            
            values.append(value)
        return np.array(values)

def parameters_side(left, s):
    lwd = np.linspace(0.48, 0.58, int(80 * s.accuracy)) * 2 * np.pi if left else np.linspace(1.2, 1.3, int(80 * s.accuracy)) * 2 * np.pi
    return False, True, None, lwd
def parameters_fast(s):
    lwd = np.linspace(0.700, 0.712, int(20 * s.accuracy)) * 2 * np.pi
    lamp = np.linspace(0.001, 0.2, int(20 * s.accuracy)) * 2 * np.pi
    return True, True, lamp, lwd
def parameters_slow(s):
    lwd = np.linspace(0.690, 0.722, int(80 * s.accuracy)) * 2 * np.pi
    lamp = np.linspace(0.001, 0.32, int(80 * s.accuracy)) * 2 * np.pi
    return True, True, lamp, lwd
def parameters_freq(s):
    lwd = np.linspace(0.690, 0.722, int(80 * s.accuracy)) * 2 * np.pi
    return False, True, None, lwd
def parameters_high_range(s):
    lwd = np.linspace(0.5, 1.5, int(200 * s.accuracy)) * 2 * np.pi
    lamp = np.linspace(0.001, 1.5, int(200 * s.accuracy)) * 2 * np.pi
    return True, True, lamp, lwd

def run_simulation(simulator, config):
    # Unpack configuration
    time = config['time']
    tstep = config['tstep']
    accuracy = config['accuracy']
    base_amp = config['base_amp']
    pulse_mod = config['pulse_mod']
    i, j = config['i'], config['j']
    extract_func = config['extract_func']
    
    # Setup simulator
    simulator.time = time
    simulator.tstep = tstep
    simulator.accuracy = accuracy
    base_fre = simulator.res_freq_static * 2 * np.pi + config['base_fre_offset']
    simulator.pulse_modulation = pulse_mod.split()

    # Get parameter preset
    preset_func = globals()[config['parameter_preset']]
    sweep_a, sweep_freq, lamp, lwd = preset_func(simulator)
    if(extract_func == 'diag_diff'): sweep_a = False
    
    print(f"\n{'='*70}")
    print(f"Extract function: {extract_func}")
    print(f"Time: {time} ns | Timestep: {tstep} ns | Accuracy factor: {accuracy}")
    print(f"{'='*70}\n")
    
    # Create figure
    fig, axes = plt.subplots(1, 2 if sweep_a and sweep_freq else 1, figsize=(14, 5))
    if sweep_freq and sweep_a:
        ax1, ax2 = axes
    else:
        ax2 = axes
        ax1 = axes
    
    element_label = rf'$\delta_{{{i}}}$-$\delta_{{{j}}}$' if extract_func == 'diag_diff' else rf'Ω_{{{i},{j}}}'
    fig.suptitle(f'{element_label} Parameter Sweeps', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Sweep 1: Amplitude
    if sweep_a:
        print("=" * 70)
        values_amp = simulator.sweep_parameters(lamp, base_fre, i, j, mode='amplitude', extract_func=extract_func)
        
        ax1.plot(lamp / (2 * np.pi), np.real(values_amp), 'o-', linewidth=2, markersize=6, color="#217dbf")
        ax1.set_xlabel('Amplitude (GHz)', fontsize=11)
        ax1.set_ylabel(element_label, fontsize=11)
        ax1.set_title(f'{element_label} vs Amplitude', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Sweep 2: Frequency
    if sweep_freq:
        print("=" * 70)
        values_fre = simulator.sweep_parameters(lwd, base_amp, i, j, mode='frequency', extract_func=extract_func)
        
        ax2.plot(lwd / (2 * np.pi), np.real(values_fre), 's-', linewidth=2, markersize=6, color='#ff7f0e')
        ax2.set_xlabel('Frequency (GHz)', fontsize=11)
        ax2.set_ylabel(element_label, fontsize=11)
        ax2.set_title(f'{element_label} vs Frequency', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add parameter info box
    pulse_text = "cos"
    if simulator.pulse_modulation == 'gaussian':
        pulse_text = "gaussian, sigma: " + pulse_mod.split()[1] + rf"$\cdot t_g$"
    info_text = (
        f"Base Parameters: A={base_amp/(2*np.pi):.4f} GHz | f={base_fre/(2*np.pi):.4f} GHz | "
        f"$t_g$={simulator.time} ns | Δt={simulator.tstep} ns | Pulse-Shape: {pulse_text} | element: ({i},{j})"
    )
    fig.text(0.5, 0.05, info_text, ha='center', va='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             transform=fig.transFigure)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    print("\n" + "=" * 70)
    print("✓ All computations completed!\n")
    
    # Prepare data for saving
    results = {
        'fig': fig,
        'config': config,
        'simulator_params': {
            'w1': simulator.w1_num / (2*np.pi),
            'w2': simulator.w2_num / (2*np.pi),
            'wc': simulator.wc_num / (2*np.pi),
        },
        'sweep_data': {}
    }
    if sweep_a:
        results['sweep_data']['amplitude'] = {
            'parameters': (lamp / (2 * np.pi)).tolist(),
            'values': np.real(values_amp).tolist()
        }
    if sweep_freq:
        results['sweep_data']['frequency'] = {
            'parameters': (lwd / (2 * np.pi)).tolist(),
            'values': np.real(values_fre).tolist()
        }
    
    return results


# ============================================================================
# COMMAND LISTENER (runs in separate thread)
# ============================================================================

class CommandListener:
    """
    Listens for user commands in the terminal and executes them.
    Runs in a separate thread to not block the main GUI.
    """
    
    def __init__(self, results, base_path=Setup.RES_PATH):
        self.results = results
        self.base_path = base_path
        self.running = True
    
    def print_help(self):
        """Print available commands."""
        print("\n" + "="*70)
        print("AVAILABLE COMMANDS:")
        print("  save()  → Save current results to files")
        print("  help()  → Show this help message")
        print("  stop()  → Exit the program")
        print("="*70 + "\n")
    
    def listen(self):
        """Listen for user commands."""
        self.print_help()
        
        while self.running:
            try:
                command = input(">>> ").strip()
                
                if command == 'save()':
                    self.handle_save()
                
                elif command == 'help()':
                    self.print_help()
                
                elif command == 'stop()':
                    self.handle_stop()
                
                elif command == '':
                    continue
                
                else:
                    print(f"❌ Unknown command: '{command}' (type 'help()' for available commands)")
            
            except KeyboardInterrupt:
                print("\n⚠️  Interrupted by user")
                self.handle_stop()
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def handle_save(self):
        """Handle save command."""
        try:
            filename = input("📝 Enter filename (without extension): ").strip()
            if not filename:
                print("❌ Save cancelled - no filename provided")
                return
            
            print(f"💾 Saving as '{filename}'...")
            save_thread = save_with_threading(self.results, self.base_path, filename)
            print("✓ Save process started in background (parallel to this session)\n")
        except Exception as e:
            print(f"❌ Save failed: {e}")
    
    def handle_stop(self):
        """Handle stop command."""
        print("\n" + "="*70)
        print("👋 Shutting down...")
        print("="*70)
        self.running = False
        plt.close('all')
        sys.exit(0)
def save_with_threading(results, base_path=Setup.RES_PATH, filename=None):
    """
    Save figure and data in parallel using threading.
    
    Args:
        results: Dictionary returned from run_simulation()
        base_path: Directory where to save results
        filename: Custom filename (without extension)
    
    Returns:
        Thread object (already started)
    """
    def _save_process():
        save_results(results, base_path, filename)
    
    # Create and start thread
    save_thread = threading.Thread(target=_save_process, daemon=False)
    save_thread.start()
    return save_thread
def save_results(results, base_path=Setup.RES_PATH, filename=None):
    """
    Save figure and data points to files (executed in thread).
    
    Args:
        results: Dictionary returned from run_simulation() containing fig and data
        base_path: Directory where to save results 
        filename: Custom filename (without extension).
    Creates:
        - {filename}_plot.png: The figure image
        - {filename}_data.json: All data points and parameters
    """
    # Create results directory if it doesn't exist
    try:
        os.makedirs(base_path, exist_ok=True)
    except Exception as e:
        print(f"❌ Error creating directory {base_path}: {e}")
        return None, None    
    try:
        # Save figure
        fig = results['fig']
        fig_path = os.path.join(base_path, f'{filename}_plot.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"📊 Figure saved: {fig_path}")
    except Exception as e:
        print(f"❌ Error saving figure: {e}")
        return None, None
    
    # Prepare data for JSON export
    config = results['config']
    sweep_data = results['sweep_data']
    simulator_params = results['simulator_params']
    
    export_data = {
        'filename': filename,
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'time_ns': config['time'],
            'timestep_ns': config['tstep'],
            'accuracy': config['accuracy'],
            'base_amplitude_GHz': config['base_amp'] / (2*np.pi),
            'base_frequency_GHz': config['base_fre_offset'],
            'element_indices': (config['i'], config['j']),
            'extraction_mode': config['extract_func'],
            'parameter_preset': config['parameter_preset'],
            'pulse_modulation': config['pulse_mod']
        },
        'simulator_parameters_GHz': simulator_params,
        'sweep_results': sweep_data
    }
    
    # Save data as JSON
    try:
        data_path = os.path.join(base_path, f'{filename}_data.json')
        with open(data_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"💾 Data saved: {data_path}")
    except Exception as e:
        print(f"❌ Error saving data: {e}")
        return fig_path, None
    
    return fig_path, data_path
if __name__ == "__main__":
    simulator = Setup()
    config = configure_simulation()
    results = run_simulation(simulator, config)
    
    # Show plot non-blocking
    print("\n" + "="*70)
    print("SIMULATION READY - Waiting for commands...")
    print("="*70)
    
    plt.show(block=False)
    
    # Run command listener on main thread
    listener = CommandListener(results, Setup.RES_PATH)
    listener.listen()
    
    # Cleanup
    plt.close('all')
