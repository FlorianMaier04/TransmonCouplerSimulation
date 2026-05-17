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

class Simulator:

    N = 10

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
        self.order = 2
        self.resonances = {self.state_a: 1, self.state_b: 0, self.state_c: 2,}  # E_state_a - wd = E_state_b
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
        self.evals, evecs = H0.eigenstates()
        self.sorted_evals, self.sorted_evecs = SortedFRFSpectrum(self.evals, evecs, self.dim_q1, self.dim_c, self.dim_q2)
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
    
    def setup_modulation(self, mode, args):
        V_tot = self.V1_dressed.full().reshape(self.d, self.d).real
        pulse_shape = None
        match mode:
            case 'cossin':
                pulse_shape = lambda t, wd, a, args: a*(np.cos(wd*t)+np.sin(wd*t))
            case 'cos':
                # V(t) hat shape (M, D, D) wenn t shape (M,) hat
                pulse_shape = lambda t, wd, a, args: a * np.cos(wd*t)
            case 'gauss':
                sigma = args[0] * self.tg
                # compute offset to assure pulse 0 at t0 (tgate is assumed to be set at this point)
                B = np.exp(self.tg**2/(2*sigma**2))
                pulse_shape = lambda t, wd, a, args: a*(np.exp(-(t-self.tg)**2/(2*sigma**2))-B) * np.cos(wd*t)
        self.V = lambda t,wd,a,args: V_tot * pulse_shape(t,wd,a,args)[:, None, None]
            
        
    def fourier_coeffs(self, wd, a, N, M=10000):
        f = lambda t: self.V(t, wd, a, args=None)
        T = 2*np.pi / wd
        t = np.linspace(0, T, M, endpoint=False)
        dt = t[1] - t[0]
        ft = f(t)  # shape: (M, D, D)
        coeffs = []
        for n in range(0, N):
            # Berechne V_n = (1/T) * integral_0^T V(t) * exp(-i*n*wd*t) dt
            exp_term = np.exp(-1j*n*wd*t)  # shape: (M,)
            integrand = ft * exp_term[:, None, None]  # shape: (M, D, D)
            coeff = (1/T) * np.sum(integrand, axis=0) * dt  # shape: (D, D)
            coeffs.append(coeff.real)
        return np.array(coeffs)
    
    def delta(self, i, wd, amp):
        fourier_coeffs = self.fourier_coeffs(wd, amp, Simulator.N)
        delta = Heff_Floquet_summed(self.order, i, i, wd, self.resonances, self.evals, fourier_coeffs[1:], V0=fourier_coeffs[0])
        return delta

    def extract_Omega_ij(self, i, j, wd, amp):
        fourier_coeffs = self.fourier_coeffs(wd, amp, Simulator.N)
        omega_ij = Heff_Floquet_summed(self.order, i, j, wd, self.resonances, self.evals, fourier_coeffs[1:], V0=fourier_coeffs[0])
        return omega_ij

    def sweep_amplitude(self, amplitude_values, base_wd, i, j):
        results = []
        iterator = tqdm(enumerate(amplitude_values), total=len(amplitude_values), leave=False, disable=False)
        if i == j:
            # Sweep delta_ii statt Omega_ij
            for idx, amp in iterator:
                value = self.delta(i, base_wd, amp)
                results.append(value)
        else:
            # Sweep Omega_ij
            for idx, amp in iterator:
                value = self.extract_Omega_ij(i, j, base_wd, amp)
                results.append(value)
        return np.array(results)

    def sweep_frequency(self, wd_values, base_amplitude, i, j):
        results = []
        iterator = tqdm(enumerate(wd_values), total=len(wd_values), leave=False, disable=False)
        if i == j:
            # Sweep delta_ii statt Omega_ij
            for idx, wd in iterator:
                value = self.delta(i, wd, base_amplitude)
                results.append(value)
        else:
            # Sweep Omega_ij
            for idx, wd in iterator:
                value = self.extract_Omega_ij(i, j, wd, base_amplitude)
                results.append(value)
        return np.array(results)