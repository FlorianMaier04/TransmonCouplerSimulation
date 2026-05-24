import sys
import numpy as np
import sympy as sp
from qutip import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from Floquet_perturbation_theory import *
from helper_function import *
from IPython.display import display
from scipy.linalg import logm
from scipy.optimize import fsolve


class Simulator:
    @property 
    def gt(self):
        return self._gt

    @gt.setter
    def gt(self, value):
        self._gt = value
        self.setup_modulation(self.modulation_mode, self.modulation_args)

    def __init__(self, dim_q1=3, dim_q2=3, dim_c=3,
                 w1=3.83, w2=3.11, wc=4.29,
                 alpha1=-0.205, alpha2=-0.216, alphac=-0.161,
                 g1c=0.115, g2c=0.110, g12=0.015):
        """        
        Args:
            dim_q1, dim_q2, dim_c: Hilbert space dimensions
            w1, w2, wc: Frequencies (in GHz, will be converted to rad/ns)
            alpha1, alpha2, alphac: Anharmonicities (in GHz, will be converted to rad/ns)
            g1c, g2c, g12: Coupling strengths (in GHz, will be converted to rad/ns)
            amp_sweep: Array of amplitudes to sweep (default: 0.001 to 0.05, 10 points)
            fre_sweep: Array of frequency detunings to sweep (default: -0.1 to 0.1, 10 points)
        """
        self.dim_q1, self.dim_q2, self.dim_c = dim_q1, dim_q2, dim_c
        self.d = self.dim_q1 * self.dim_c * self.dim_q2
        self.state_a = dim_q2 * dim_c  # state |100>
        self.state_b = 1  # state |001>
        self.state_c = dim_q2  # state |010>
        self.state_d = 2 # state |002> # this state is at same energy as state_a at the resonant frequency
        self.state_e = 1*(dim_c*dim_q2)+1*(dim_q2)+1 # |111>

        self.order = 2
        self.resonances = {self.state_a: 1, self.state_b: 0, self.state_c: 2}  # E_state_a - wd = E_state_b
        self.sd = len(self.resonances)
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
        self.tg = 200 # in ns
        self._setup_operators()
        self._setup_hamiltonian()

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
        self.E_array = self.sorted_evals.reshape(self.d)
        self.E_states = self.sorted_evecs.reshape(self.d)
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
        self.V1_dressed_array = self.V1_dressed.full().reshape(self.d, self.d).real
        # Resonant frequency
        self.res_freq_static = (self.sorted_evals[1, 0, 0] - self.sorted_evals[0, 0, 1]) / (2 * np.pi)

    def heff_element(self, i, j, wd, amp):
        delta = Heff_Floquet_summed(self.order, i, j, wd, self.resonances, self.E_array, amp/2*self.V1_dressed_array, V0=None)
        return delta
    
    def resonant_condition(self, fre, amp, order, i, f, ):
        fre = float(np.atleast_1d(fre)[0])
        delta_i = Heff_Floquet_summed(
            self.order,
            i,
            i,
            fre,
            self.resonances,  # E_i - wd = E_f
            self.E_array,
            amp / 2 * self.V1_dressed_array,
            V0=None,
            analytics=False,
        )
        delta_f = Heff_Floquet_summed(
            order,
            f,
            f,
            fre,
            self.resonances,  # E_i - wd = E_f
            self.E_array,
            amp / 2 * self.V1_dressed_array,
            V0=None,
            analytics=False,
        )
        diff = delta_f - delta_i
        return float(np.real(diff))

    def find_resonance(self, amp):
        wd_initial_guess = 1*self.res_freq_static * 2*np.pi
        # wd_initial_guess = 0.5*2*np.pi
        resonant_wd_solution= fsolve(self.resonant_condition, wd_initial_guess, 
            args=(amp, self.order, self.state_a, self.state_b,),)
        return resonant_wd_solution
        

    def heff(self, wd, amp):
        d_subspace = len(self.resonances.keys())
        heff = np.zeros((d_subspace, d_subspace))
        states = [*self.resonances.keys()]
        for idx_a, state_a in enumerate(states):
            for idx_b, state_b in enumerate(states):
                heff[idx_a,idx_b] = self.heff_element(state_a, state_b, wd, amp)
        return heff

    
    def config(self, config):
        """Configure the simulator with provided settings."""
        self.tg = config.get('tgate', 200)
        self.order = config.get('order', 3)

    def get_state_name(self, idx):
        match idx:
            case self.state_a:
                return 'a'
            case self.state_b:
                return 'b'
            case self.state_c:
                return 'c'
            case self.state_d:
                return 'd'

config = {
    'showA': True, 
    'showB': True, 
    'showC': True,
    'initial_state': 'b',  # 'a' or 'b' for initial state selection
    'amplitude': 0.1,  # in GHz
    'frequency': 0.7063365266,  # in GHz (res: 0.7063365266894976) (5.41521)
    'order': 2,
    'tgate': 600,  # in ns
    'pulse_shape': 'cos',  # 'cos', 'cossin', or 'gauss'
    'pulse_args': [0.5],  # for gauss: [sigma_fraction]
}

if __name__ == "__main__":
    simulator = Simulator()
    simulator.config(config)
    amp = 0.1 * 2*np.pi
    resonances = simulator.find_resonance(amp)
    resonance_freq = resonances[0] / (2*np.pi)  # in GHz
    print(f"Resonante Frequenz: {resonances/(2*np.pi)} GHz")
    
    # Genauigkeit für den Sweep um die Nullstelle (10er-Potenz, z.B. -3 = 0.001)
    precision_power = -8
    precision = 10**precision_power
    
    # Bereich um die Nullstelle
    sweep_range = 0.01*precision*1e3  # ±10 MHz um die Nullstelle
    freq_range = np.arange(resonance_freq - sweep_range, resonance_freq + sweep_range, precision)
    # Magnitude des Heff-Elements berechnen
    magnitudes_db = []
    for freq in freq_range:
        heff = simulator.heff(freq * 2*np.pi, amp)
        # Magnitude des off-diagonal Elements |Heff[0,1]|
        magnitude = heff[1, 1] - heff[2,2]
        # magnitude_db = 20 * np.log10(magnitude + 1e-10)  # +1e-10 um log(0) zu vermeiden
        magnitudes_db.append(magnitude)
    # Plot um die Nullstelleq
    plt.figure(figsize=(10, 6))
    plt.plot((freq_range - resonance_freq) * 1000, magnitudes_db, 'b-', linewidth=2)  # x-Achse in MHz
    plt.axvline(x=0, color='r', linestyle='--', label='Resonante Frequenz')
    plt.xlabel('Detuning (MHz)')
    plt.ylabel('|Heff[0,1]| (dB)')
    plt.title(f'Heff-Magnitude um die Nullstelle (Genauigkeit: 10^{precision_power})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    # plt.show()