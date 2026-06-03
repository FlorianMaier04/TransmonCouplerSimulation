import sys
import numpy as np
import sympy as sp
from qutip import *
import matplotlib.pyplot as plt
from Floquet_perturbation_theory import *
from helper_function import *
from IPython.display import display
from scipy.linalg import logm
from scipy.optimize import fsolve
import scipy.special
from utils_math import *
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.interpolate import interp1d


class Simulation:
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
        self.state_b = dim_q2 * dim_c  # state |100>
        self.state_a = 1  # state |001>
        self.state_c = dim_q2  # state |010>
        self.state_d = 2 # state |002> # this state is at same energy as state_a at the resonant frequency
        self.state_e = 1*(dim_c*dim_q2)+1*(dim_q2)+1 # |111>
        self.order = 3
        self.resonances = {self.state_a: 0, self.state_b: 1, self.state_c: 2}  # E_state_a - wd = E_state_b
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
        self._setup_operators()
        self._setup_hamiltonian()
        self.epsilonx = lambda t: 1
        self.epsilony = lambda t: 0
        self.use_pulse_shape = False

    def _setup_operators(self):
        self.a_q1 = tensor(destroy(self.dim_q1), qeye(self.dim_c), qeye(self.dim_q2))
        self.a_q2 = tensor(qeye(self.dim_q1), qeye(self.dim_c), destroy(self.dim_q2))
        self.a_qc = tensor(qeye(self.dim_q1), destroy(self.dim_c), qeye(self.dim_q2))

    def _setup_hamiltonian(self):
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

    def resonant_condition(self, fre, amp, order, i, f, ):
        fre = float(np.atleast_1d(fre)[0])
        resonances = {self.state_a:0, self.state_b:1, self.state_c:2}
        delta_i = Heff_Floquet_summed(self.order,i,i,
            fre,resonances,self.E_array,amp / 2 * self.V1_dressed_array,V0=None,analytics=False,)
        delta_f = Heff_Floquet_summed(order,f,f,
            fre,resonances,self.E_array,amp / 2 * self.V1_dressed_array,V0=None,analytics=False,)
        diff = delta_f - delta_i
        return float(np.real(diff))

    def find_resonance(self, amp):
        wd_initial_guess = self.res_freq_static * 2*np.pi
        resonant_wd_solution = fsolve(self.resonant_condition, wd_initial_guess, 
            args=(amp, self.order, self.state_a, self.state_b,),)
        return resonant_wd_solution[0]

    def heff_element(self, i, j, wd, amp):
        V_posharm = (amp/2) * self.V1_dressed_array
        V0 = None
        # if(self.use_pulse_shape):
        #     pulse = lambda t: amp*(self.epsilonx + 1j*self.epsilony) * np.cos(wd*t)
        #     coeffs = fourier_coeffs(pulse, wd)
        #     V0 = coeffs[0]
        #     V_posharm = coeffs[1:]
        element = Heff_Floquet_summed(self.order, i, j, wd, self.resonances, self.E_array, V_posHarm=V_posharm, V0=V0)
        return element

    def heff(self, wd, amp):
        heff = np.zeros((self.sd, self.sd), dtype=complex)
        states = [*self.resonances.keys()]
        for idx_a, state_a in enumerate(states):
            for idx_b, state_b in enumerate(states):
                heff[idx_a,idx_b] = self.heff_element(state_a, state_b, wd, amp)
        return heff
    
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

def compute_phi0(s, wd, amp, psi0): 
    rW = 2
    W = W_Floquet_elements(rW, wd, s.resonances, s.E_array, amp/2 * s.V1_dressed_array,
                        None, ref_state = None, analytics = False)
    # express phi0 in the resonant subspace, assuming psi0 is no superposition and lies in the res. subspace
    phi0 = qt.zero_ket(s.sd)
    projections = [None] * s.sd
    for idx, a in enumerate(s.resonances.keys()):
        dic = W[rW, psi0, a]
        n_psi0 = s.resonances[psi0]
        if n_psi0 in dic: c = np.conj(dic[n_psi0])
        else: c = 0
        # if(c!=0): print("idx: ", a, " c: ", c, " richtiger idx: ", idx)
        phi0 += c * qt.basis(s.sd, idx)
        projections[idx] = qt.basis(s.sd, idx).proj()
        # projections[idx] = c * np.conj(c) * qt.basis(simulator.sd, idx).proj()
    if psi0 == s.state_a:
        phi0 = qt.basis(s.sd, 0)
    if psi0 == s.state_b:
      phi0 = qt.basis(s.sd, 1)
    if psi0 == s.state_c:
        phi0 = qt.basis(s.sd, 2)
    # phi0 = phi0.unit()
    # print("phi0: ", phi0)
    return phi0, projections

def find_optimal_gate_time(wd, amp, s, heff):
    Omega_ab = heff[0, 1]
    tg,_,_ = find_optimal_time(wd,amp,s.order,
        s.state_b,s.state_a,s.state_c,s.E_array,s.V1_dressed_array, int(np.pi/(2*Omega_ab.real)),)
    tlist = np.arange(0, tg, 0.01)
    return tg, tlist

def h_target(Delta, lambda_param, tg, detune = 0):
    H = np.zeros((3, 3), dtype=complex)
    H[0][0] = 0
    H[1][1] = 0
    H[2][2] = Delta
    Vx = [None] * (2)
    Vy = [None] * (2)
    epsilonx, epsilony = pulse_functions_gauss(tg, Delta, detune)
    delta1 = delta1_gauss(lambda_param, Delta, epsilonx)
    delta1_mat = [[0,0,0], [0,1,0], [0,0,0]]
    for i in range(0,2):
        lambda_i = 1 if not i == 1 else lambda_param
        Vx[i] = [sigma_x_ij(i, i+1, 3) * lambda_i * 1/2, epsilonx]
        Vy[i] = [sigma_y_ij(i, i+1, 3) * lambda_i * 1/2, epsilony]
    return [Qobj(H), [Qobj(delta1_mat), delta1], *Vx, *Vy], epsilonx, epsilony, delta1

def compute_t_dependency(heff, tg):
    lambda_param = heff[0][1] / heff[1][2]
    Delta = -(heff[0][0] - heff[1][1])/2
    detune = heff[0][0] - heff[1][1]
    return h_target(Delta, lambda_param, tg, delta_a = heff[2][2], delta_c = heff[0][0], detune = detune)

def delta1_gauss(lambda_param, Delta, epsilonx):
    f = lambda t,args=None: (lambda_param**2-4) * epsilonx(t,args)**2/(4*Delta)
    return f

def pulse_functions_gauss(tg, Delta, delta_detune, sigma_ratio=0.5):
    sigma = tg * sigma_ratio
    B_ratio = np.exp(-tg**2 / (8 * sigma**2)) # B = B_ratio * A
    # find the pulse-amplitude
    def pulse_area(A):
        f = lambda t: np.sqrt(delta_detune**2 + A**2 * (np.exp(-(t - tg/2)**2 / (2*sigma**2)) - B_ratio)**2)
        I, _ = quad(f, 0, tg)
        return I
    f = lambda A: pulse_area(A) - np.pi
    sol = root_scalar(
        f,
        bracket=[0, 2],  # anpassen
        method='brentq'
    )
    A = sol.root
    epsilon_x = lambda t, args=None: max(0, np.exp(-(t - tg / 2)**2 / (2 * sigma**2)) - B_ratio) * A
    epsilon_y = lambda t, args=None: 1/Delta * A/sigma**2 * (t-tg/2) * np.exp(-(t-tg/2)**2/(2*sigma**2))
    # epsilon_y = lambda t, args=None: 0
    return epsilon_x, epsilon_y

def compute_cos_params(s, tg):
    # like for the gauss Pulse, also for constant epsilon the integral over epsilon must be pi
    # the amplitude of epsilonx is directly connected to the amplitude via Omega_ab (in this case epsilony=0)
    epsilonx = np.pi/tg    
    amp_min, amp_max = 0.1, 1.5*2*np.pi
    f = lambda A: abs(s.heff_element(s.state_a, s.state_b, s.find_resonance(A), A)) - epsilonx # epsilonx * 1 = Omega_ab
    sol = root_scalar(f, bracket=[amp_min, amp_max])
    return sol.root, s.find_resonance(sol.root)

def print_matrix(A, d=3, th=1e-4):
    f = lambda x: "0" if abs(x) < th else (f"{x:.{d}e}" if abs(x) < 10**(-d) or abs(x) >= 1e4 else f"{x:.{d}f}")
    s = lambda z: (
        f(z.real) if abs(z.imag) < th else
        f"{f(z.imag)}j" if abs(z.real) < th else
        f"{f(z.real)} {'+' if z.imag >= 0 else '-'} {f(abs(z.imag))}j"
    )
    rows = [[s(z) for z in row] for row in A]
    w = [max(len(r[j]) for r in rows) for j in range(len(rows[0]))]
    print('\n'.join('[ ' + '  '.join(x.rjust(wi) for x, wi in zip(r, w)) + ' ]' for r in rows))

config = {
    'amplitude': 0.2,  # in GHz
    'freq': 0.7102, 
    'tgate': 119.951, # in ns
}
if __name__ == "__main__":
    s = Simulation()
    amp = config['amplitude'] * 2 * np.pi
    tg = config['tgate']
    wd = config['freq'] * 2 * np.pi
    
    heff = s.heff(wd, amp)
    lh = compute_t_dependency(heff, tg,)

    Omega_ab = heff[0, 1]
    # tg,f_optimal,U_optimal = find_optimal_time(wd,amp,s.order,
    #     s.state_b,s.state_a,s.state_c,s.E_array,s.V1_dressed_array, int(np.pi/(2*Omega_ab.real)),)
    # print("f_optimal: ", f_optimal)
    # print_matrix(U_optimal)
    tlist = np.linspace(0, tg, 5000)
    print("tg")
    f = s.extract_fidelity(lh, tlist)
    print(" simulated fidelity: ", (f*100),"%")